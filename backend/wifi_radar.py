"""Wi-Fi Wall Radar - detect motion behind walls via signal variance.

Two data sources, both work without extra Python deps or admin rights:

  --source rtt     (DEFAULT, recomendado)
      Pings the default gateway every ~100ms and tracks the variance of the
      round-trip time + packet loss rate. A human body crossing the line of
      sight causes retransmissions -> RTT spikes (1ms -> 8ms) and occasional
      loss. Works in real time, no driver cache.

  --source signal  (legacy)
      Reads `netsh wlan show interfaces` and parses the connected AP's
      Signal %. PROBLEM on Windows: when the connection is idle, the driver
      caches this value and never refreshes it. You'll see a constant 79%
      for minutes even with people walking around. Only useful if you have
      heavy traffic forcing the driver to update.

Standalone:
  python -m backend.wifi_radar live                       # auto = rtt
  python -m backend.wifi_radar live --source rtt
  python -m backend.wifi_radar live --source signal
  python -m backend.wifi_radar live --source rtt --sensitivity 4 --interval 0.1

Integrated with the camera radar:
  python -m backend.camera_detector radar --wifi-radar
"""

from __future__ import annotations

import argparse
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Source 1: netsh signal % (cached on Windows, kept for fallback)
# ---------------------------------------------------------------------------

_SIGNAL_PATTERNS = (
    re.compile(r"Se\S+al\s*:\s*(\d+)\s*%", re.IGNORECASE),
    re.compile(r"Signal\s*:\s*(\d+)\s*%", re.IGNORECASE),
)


def _decode_console(raw: bytes) -> Optional[str]:
    for enc in ("utf-8", "cp1252", "cp850", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def read_wifi_signal() -> Optional[int]:
    """Connected AP signal 0..100 from netsh. CACHED — slow to refresh."""
    try:
        proc = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True, timeout=3.0,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    text = _decode_console(proc.stdout)
    if text is None:
        return None
    for pat in _SIGNAL_PATTERNS:
        m = pat.search(text)
        if m:
            return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Source 2: ping RTT to default gateway (live, no caching)
# ---------------------------------------------------------------------------

_RTT_PATTERN = re.compile(r"(?:tiempo|time)\s*[=<]\s*(\d+)\s*ms", re.IGNORECASE)


def get_default_gateway() -> Optional[str]:
    """Find the default IPv4 gateway via `route print 0.0.0.0` on Windows."""
    try:
        proc = subprocess.run(
            ["route", "print", "0.0.0.0"],
            capture_output=True, timeout=3.0,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    text = _decode_console(proc.stdout)
    if text is None:
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "0.0.0.0" and parts[1] == "0.0.0.0":
            gw = parts[2]
            octets = gw.split(".")
            if len(octets) == 4 and all(o.isdigit() and 0 <= int(o) <= 255 for o in octets):
                return gw
    return None


def read_ping_rtt_ms(host: str, timeout_ms: int = 300) -> Optional[float]:
    """Ping host once. Return RTT in ms, or None on loss/timeout."""
    try:
        proc = subprocess.run(
            ["ping", "-n", "1", "-w", str(timeout_ms), host],
            capture_output=True, timeout=(timeout_ms / 1000.0) + 1.0,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    text = _decode_console(proc.stdout)
    if text is None:
        return None
    m = _RTT_PATTERN.search(text)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Generic sliding-window variance detector
# ---------------------------------------------------------------------------

@dataclass
class RadarState:
    state: str = "init"            # init|calibrating|warming|quiet|motion|no_link
    source: str = "rtt"            # rtt | signal
    raw: Optional[float] = None    # last reading (ms for rtt, % for signal)
    baseline_median: Optional[float] = None
    baseline_mad: Optional[float] = None
    anomaly_rate: float = 0.0      # fraction of recent samples flagged as anomalous
    loss_rate: float = 0.0
    threshold: float = 0.30        # anomaly_rate that fires motion
    k_sigma: float = 4.0
    samples: int = 0
    last_loss: bool = False
    # Back-compat aliases (the camera banner used these names previously)
    variance: float = 0.0
    baseline_var: Optional[float] = None
    ratio: float = 0.0


class WallRadar:
    """Robust motion detector using median + MAD + anomaly-rate.

    Why not plain variance: a single noisy 5s baseline can poison the threshold
    permanently. Median + MAD ignore extreme samples during calibration, so
    even if 30-40% of calibration samples were already motion, the baseline
    stays useful.

    Per-sample rule (after calibration):
      anomalous = (rtt > median + k*MAD)  OR  (sample is a loss)
    Motion fires when anomaly_rate in the recent window exceeds `threshold`.
    Threshold = 0.30 means "30% of the last N samples were anomalous".
    """

    def __init__(self, window: int = 30, baseline_secs: float = 10.0,
                 k_sigma: float = 4.0, anomaly_threshold: float = 0.30,
                 source: str = "rtt"):
        self.window_size = window
        self.win: deque = deque(maxlen=window)
        self.loss_win: deque = deque(maxlen=window)
        self.anom_win: deque = deque(maxlen=window)
        self.k_sigma = k_sigma
        self.anomaly_threshold = anomaly_threshold
        self.source = source
        self.baseline_until = time.time() + baseline_secs
        self.baseline_samples: List[float] = []
        self.baseline_median: Optional[float] = None
        self.baseline_mad: Optional[float] = None

    def _finalize_baseline(self) -> None:
        if not self.baseline_samples:
            self.baseline_median = 5.0
            self.baseline_mad = 5.0
            return
        arr = np.array(self.baseline_samples, dtype=float)
        med = float(np.median(arr))
        mad_raw = float(np.median(np.abs(arr - med)))
        # 1.4826 converts MAD to a stddev-equivalent under Gaussian noise
        mad = max(1.0, 1.4826 * mad_raw)
        self.baseline_median = med
        self.baseline_mad = mad

    def push(self, value: Optional[float]) -> RadarState:
        now = time.time()
        was_loss = value is None

        st = RadarState(
            source=self.source, raw=value,
            threshold=self.anomaly_threshold, k_sigma=self.k_sigma,
            last_loss=was_loss,
        )

        # Phase 1: calibration
        if now < self.baseline_until:
            if not was_loss:
                self.baseline_samples.append(float(value))
            st.state = "calibrating"
            st.samples = len(self.baseline_samples)
            return st

        # Phase 2: lock in baseline once
        if self.baseline_median is None:
            self._finalize_baseline()

        st.baseline_median = self.baseline_median
        st.baseline_mad = self.baseline_mad
        st.baseline_var = self.baseline_median  # alias for old banner

        # Per-sample anomaly classification
        if was_loss:
            is_anom = True
        else:
            self.win.append(float(value))
            z = (value - self.baseline_median) / max(1.0, self.baseline_mad)
            is_anom = z > self.k_sigma
        self.loss_win.append(1 if was_loss else 0)
        self.anom_win.append(1 if is_anom else 0)

        # Need a half-full window to judge
        if len(self.anom_win) < max(5, self.window_size // 2):
            st.state = "warming"
            st.samples = len(self.anom_win)
            return st

        st.loss_rate = float(np.mean(self.loss_win))
        st.anomaly_rate = float(np.mean(self.anom_win))
        st.ratio = st.anomaly_rate / max(0.05, self.anomaly_threshold)
        # back-compat: also report current short-window variance for display
        if self.win:
            st.variance = float(np.var(np.array(self.win, dtype=float)))
        st.samples = len(self.anom_win)
        st.state = "motion" if st.anomaly_rate >= self.anomaly_threshold else "quiet"
        return st


# ---------------------------------------------------------------------------
# Background thread
# ---------------------------------------------------------------------------

class WallRadarThread:
    """Background thread that polls a reader callable and updates RadarState."""

    def __init__(self, reader: Callable[[], Optional[float]], source: str = "rtt",
                 window: int = 30, baseline_secs: float = 10.0,
                 k_sigma: float = 4.0, anomaly_threshold: float = 0.30,
                 interval: float = 0.15):
        self.reader = reader
        self.radar = WallRadar(window=window, baseline_secs=baseline_secs,
                               k_sigma=k_sigma, anomaly_threshold=anomaly_threshold,
                               source=source)
        self.interval = interval
        self._state = RadarState(source=source, threshold=anomaly_threshold,
                                 k_sigma=k_sigma)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_state(self) -> RadarState:
        with self._lock:
            return self._state

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self.reader()
            st = self.radar.push(value)
            with self._lock:
                self._state = st
            self._stop.wait(self.interval)


# ---------------------------------------------------------------------------
# Source resolution helpers
# ---------------------------------------------------------------------------

def make_reader(source: str) -> Tuple[Callable[[], Optional[float]], str, str]:
    """Return (reader, resolved_source, info_string)."""
    if source == "rtt" or source == "auto":
        gw = get_default_gateway()
        if gw is not None:
            return (lambda: read_ping_rtt_ms(gw)), "rtt", f"ping -> {gw}"
        if source == "rtt":
            raise RuntimeError("No se detecto gateway por defecto. Esta conectado a la red?")
        # auto fallback
        return (lambda: _signal_as_float()), "signal", "netsh signal % (gateway no detectado)"
    if source == "signal":
        return (lambda: _signal_as_float()), "signal", "netsh signal % (puede estar cacheado!)"
    raise ValueError(f"Source desconocido: {source}")


def _signal_as_float() -> Optional[float]:
    v = read_wifi_signal()
    return float(v) if v is not None else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_STATE_TAG = {
    "init":        ("...", "\033[90m"),
    "calibrating": ("CAL", "\033[36m"),
    "warming":     ("WRM", "\033[33m"),
    "quiet":       ("OK ", "\033[32m"),
    "motion":      ("!!!", "\033[91m"),
    "no_link":     ("XX ", "\033[91m"),
}
_RESET = "\033[0m"


def _format_raw(st: RadarState) -> str:
    if st.raw is None:
        return "  LOSS"
    if st.source == "rtt":
        return f"{st.raw:5.1f} ms"
    return f"{int(st.raw):3d} %"


def cmd_live(args) -> None:
    try:
        reader, resolved, info = make_reader(args.source)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return

    radar = WallRadar(window=args.window, baseline_secs=args.baseline,
                      k_sigma=args.k_sigma, anomaly_threshold=args.threshold,
                      source=resolved)

    # Optional dashboard broadcast
    from backend.event_broadcast import init_broadcaster, publish as bc_publish
    init_broadcaster(args.broadcast)
    last_state_for_anchor = None
    print(f"Wi-Fi Wall Radar | source={resolved} ({info})")
    print(f"  window={args.window} baseline={args.baseline}s  k={args.k_sigma}sigma  "
          f"anomaly_thr={args.threshold*100:.0f}%  interval={args.interval}s")
    print(f"  Ctrl+C para salir\n")
    print(f"=== CALIBRANDO {args.baseline}s ===")
    print(f"  La sala TIENE QUE estar vacia y la persona quieta.")
    print(f"  Si arranca con LOSS o picos altos, abortar (Ctrl+C) y reintentar.\n")

    last_motion = False
    baseline_printed = False
    try:
        while True:
            value = reader()
            st = radar.push(value)
            tag, color = _STATE_TAG.get(st.state, ("???", ""))
            raw_str = _format_raw(st)

            # First non-calibrating tick: print the resolved baseline
            if not baseline_printed and st.baseline_median is not None:
                med = st.baseline_median
                mad = st.baseline_mad or 0
                anom_thresh_ms = med + args.k_sigma * mad
                quality = "OK" if mad < 20 else ("ALTA" if mad < 50 else "MUY ALTA")
                print(f"\n>>> Baseline fijado: mediana={med:.1f}ms  MAD={mad:.1f}ms  "
                      f"(ruido: {quality})")
                print(f">>> Umbral de anomalia: RTT > {anom_thresh_ms:.0f}ms o LOSS.")
                if mad >= 20:
                    print(f">>> AVISO: baseline ruidoso. Si querias sala vacia, "
                          f"reinicia ahora con la sala REALMENTE quieta.\n")
                else:
                    print()
                baseline_printed = True

            metrics = ""
            if st.baseline_median is not None:
                metrics = (f"  anom={st.anomaly_rate*100:4.1f}%  "
                           f"loss={st.loss_rate*100:4.1f}%  "
                           f"med={st.baseline_median:.1f}+/-{st.baseline_mad:.1f}ms")
            print(f"{color}[{tag}]{_RESET} {raw_str}{metrics}")
            if st.state == "motion" and not last_motion:
                print(f"     {color}>>> ALGUIEN AL OTRO LADO DEL MURO <<<{_RESET}")
            last_motion = (st.state == "motion")

            # Broadcast on every state change (and on motion ticks) so the
            # dashboard heatmap pulses smoothly without spamming on quiet.
            if args.broadcast and (st.state != last_state_for_anchor or st.state == "motion"):
                bc_publish(
                    anchor_id=args.anchor_id,
                    source=f"wifi_{resolved}",
                    kind="motion" if st.state == "motion" else "state",
                    state=st.state,
                    value=st.raw if st.raw is not None else 0,
                    extra={"anomaly_rate": st.anomaly_rate, "loss_rate": st.loss_rate},
                )
                last_state_for_anchor = st.state

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("live", help="Live oscilloscope + motion alarm")
    pl.add_argument("--source", choices=["auto", "rtt", "signal"], default="auto",
                    help="Data source. auto = rtt si hay gateway, sino signal.")
    pl.add_argument("--window", type=int, default=30,
                    help="Samples in sliding window (default 30 = ~4.5s @ 150ms)")
    pl.add_argument("--baseline", type=float, default=10.0,
                    help="Seconds to calibrate empty-room baseline (default 10)")
    pl.add_argument("--k-sigma", type=float, default=4.0,
                    help="Anomaly = sample > median + k*MAD. Higher = less sensitive (default 4)")
    pl.add_argument("--threshold", type=float, default=0.30,
                    help="Fire motion when >=threshold of window is anomalous (default 0.30 = 30%%)")
    pl.add_argument("--interval", type=float, default=0.15,
                    help="Polling period in seconds (0.15 = ~7 Hz)")
    pl.add_argument("--broadcast", default=None,
                    help="URL del dashboard FastAPI (ej http://127.0.0.1:8000). "
                         "Publica eventos al WebSocket para visualizar en 3D.")
    pl.add_argument("--anchor-id", default="starlink_hotspot",
                    help="ID del anchor (en data/anchors.json) al que se atribuyen los eventos.")
    pl.set_defaults(func=cmd_live)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
