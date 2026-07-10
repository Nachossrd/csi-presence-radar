"""ESP32-S3 CSI reader: serial -> motion detection -> dashboard events.

Reads CSV lines from the ESP32 sketch (`esp32/csi_radar/csi_radar.ino`):
   ts_ms,rssi,n_subc,amp_mean,amp_var,channel,src_mac

Detects motion using robust statistics on the *sliding-window variance of
amp_mean*. Same approach as wifi_radar / multi_ap_radar / ble_radar so the
dashboard treatment is uniform:
  - Calibrate baseline median + MAD from the first ~10s of quiet samples
  - Per-sample anomaly: |amp_mean - baseline_median| > k_sigma * MAD
  - Motion state: anomaly_rate over recent window > threshold

CLI:
  python -m backend.csi_reader live                          # show stream + state
  python -m backend.csi_reader live --port COM7 --broadcast http://127.0.0.1:8000

Anchor binding: events are published with anchor_id="esp32_csi_radar".
Add a matching entry to data/anchors.json so the dashboard renders it.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import serial


DEFAULT_PORT = "COM7"
DEFAULT_BAUD = 115200
ANCHOR_ID = "esp32_csi_radar"

# Detector parameters tuned for CSI from a multi-MAC stream (vs single-anchor RTT)
WINDOW_SIZE       = 30      # samples (~1s at 30 Hz)
BASELINE_SECS     = 10.0    # initial calibration time
K_SIGMA           = 2.5     # anomaly = sample > median + K * MAD
ANOMALY_THRESHOLD = 0.15    # 15% of window anomalous -> motion (was 30%)
MAD_FLOOR         = 0.5     # protects against zero-variance edge case

# Default motion metric:
#   amp_mean: average amplitude across 64 subcarriers (per packet)
#   amp_var:  variance across subcarriers within one packet (= multipath
#             complexity — direct motion indicator due to frequency-selective
#             fading when a body partially obstructs the path)
DEFAULT_METRIC = "amp_var"

CSV_LINE_RE = re.compile(
    r"^(\d+),(-?\d+),(\d+),([0-9.\-]+),([0-9.\-]+),(\d+),"
    r"([0-9a-fA-F:]{17})$"
)


@dataclass
class CSISample:
    ts_ms: int
    rssi: int
    n_subc: int
    amp_mean: float
    amp_var: float
    channel: int
    src_mac: str


@dataclass
class State:
    state: str = "init"
    samples: int = 0
    rate_hz: float = 0.0
    amp_now: float = 0.0
    amp_median: Optional[float] = None
    amp_mad: Optional[float] = None
    anomaly_rate: float = 0.0
    unique_macs: int = 0


def parse_line(line: str) -> Optional[CSISample]:
    m = CSV_LINE_RE.match(line.strip())
    if not m:
        return None
    return CSISample(
        ts_ms=int(m.group(1)),
        rssi=int(m.group(2)),
        n_subc=int(m.group(3)),
        amp_mean=float(m.group(4)),
        amp_var=float(m.group(5)),
        channel=int(m.group(6)),
        src_mac=m.group(7).lower(),
    )


class CSIRadar:
    """Sliding-window robust motion detector on amp_mean or amp_var."""

    def __init__(self, window: int = WINDOW_SIZE,
                 baseline_secs: float = BASELINE_SECS,
                 k_sigma: float = K_SIGMA,
                 anomaly_threshold: float = ANOMALY_THRESHOLD,
                 metric: str = DEFAULT_METRIC,
                 source_mac: Optional[str] = None):
        self.win: deque = deque(maxlen=window)
        self.k = k_sigma
        self.thr = anomaly_threshold
        self.metric = metric
        self.source_mac = source_mac.lower() if source_mac else None
        self.baseline_until = time.time() + baseline_secs
        self.baseline_buf: list[float] = []
        self.median: Optional[float] = None
        self.mad: Optional[float] = None
        self.mac_seen: set[str] = set()
        self._t_first: float = 0.0
        self._n_total: int = 0
        self._skipped: int = 0

    def _metric_value(self, sample: CSISample) -> float:
        return sample.amp_var if self.metric == "amp_var" else sample.amp_mean

    def push(self, sample: CSISample) -> Optional[State]:
        # Filter by source MAC if requested
        if self.source_mac and sample.src_mac != self.source_mac:
            self._skipped += 1
            return None
        now = time.time()
        if self._t_first == 0:
            self._t_first = now
        self._n_total += 1
        value = self._metric_value(sample)
        self.win.append(value)
        self.mac_seen.add(sample.src_mac)

        st = State(samples=self._n_total,
                   amp_now=value,
                   unique_macs=len(self.mac_seen))
        elapsed = max(0.001, now - self._t_first)
        st.rate_hz = self._n_total / elapsed

        # Phase 1: calibration
        if now < self.baseline_until:
            self.baseline_buf.append(sample.amp_mean)
            st.state = "calibrating"
            return st

        # Phase 2: lock baseline
        if self.median is None:
            if len(self.baseline_buf) >= 5:
                self.median = statistics.median(self.baseline_buf)
                deviations = [abs(x - self.median) for x in self.baseline_buf]
                mad_raw = statistics.median(deviations)
                self.mad = max(MAD_FLOOR, 1.4826 * mad_raw)
            else:
                self.median = 0.0
                self.mad = MAD_FLOOR

        st.amp_median = self.median
        st.amp_mad = self.mad

        # Phase 3: warming up the window
        if len(self.win) < max(8, self.win.maxlen // 3):
            st.state = "warming"
            return st

        anomalies = sum(1 for x in self.win if abs(x - self.median) > self.k * self.mad)
        anom_rate = anomalies / len(self.win)
        st.anomaly_rate = anom_rate
        st.state = "motion" if anom_rate >= self.thr else "quiet"
        return st


def cmd_live(args) -> None:
    print(f"CSI reader | port={args.port} baud={args.baud} window={args.window} "
          f"k={args.k_sigma} thr={args.threshold*100:.0f}%  metric={args.metric}"
          f"{'  source_mac='+args.source_mac if args.source_mac else ''}")
    print(f"Conectando a {args.port}...")
    try:
        s = serial.Serial(args.port, args.baud, timeout=1.0)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    print("OK. Calibrando ~10s con la sala vacia. NO te muevas...\n")

    radar = CSIRadar(window=args.window, baseline_secs=args.baseline,
                     k_sigma=args.k_sigma, anomaly_threshold=args.threshold,
                     metric=args.metric, source_mac=args.source_mac)

    publisher = None
    if args.broadcast:
        from backend.event_broadcast import init_broadcaster, publish as bc_publish
        init_broadcaster(args.broadcast)
        publisher = bc_publish

    last_state = None
    last_print = 0.0
    motion_count_since_print = 0
    try:
        while True:
            line = s.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            sample = parse_line(line)
            if sample is None:
                # passthrough boot/init/stats lines
                if line.startswith("[") and time.time() - last_print > 1.0:
                    print(f"  {line}")
                continue

            st = radar.push(sample)
            if st is None:  # filtered out by source_mac
                continue
            if st.state == "motion":
                motion_count_since_print += 1

            now = time.time()
            if now - last_print >= 1.0:
                color = {
                    "calibrating": "\033[36m",
                    "warming":     "\033[33m",
                    "quiet":       "\033[32m",
                    "motion":      "\033[91m",
                }.get(st.state, "")
                reset = "\033[0m"
                m_str = f"med={st.amp_median:5.1f}+/-{st.amp_mad:4.1f}" if st.amp_median is not None else "med=  -  +/-  - "
                print(f"{color}[{st.state.upper():11s}]{reset} "
                      f"n={st.samples:6d} rate={st.rate_hz:5.1f}Hz  amp={st.amp_now:6.1f}  "
                      f"{m_str}  anom={st.anomaly_rate*100:4.1f}%  macs={st.unique_macs}")
                last_print = now
                motion_count_since_print = 0

            # Broadcast every state change AND every ~1s during motion
            should_broadcast = False
            if st.state != last_state:
                should_broadcast = True
            elif st.state == "motion" and now - last_print < 0.05:
                should_broadcast = True
            if publisher and should_broadcast:
                publisher(
                    anchor_id=ANCHOR_ID,
                    source="esp32_csi",
                    kind="motion" if st.state == "motion" else "state",
                    state=st.state,
                    value=float(sample.rssi),
                    extra={
                        "amp_mean": sample.amp_mean,
                        "amp_var": sample.amp_var,
                        "anomaly_rate": st.anomaly_rate,
                        "amp_median": st.amp_median,
                        "n_subc": sample.n_subc,
                        "channel": sample.channel,
                        "src_mac": sample.src_mac,
                        "rate_hz": st.rate_hz,
                    },
                )
            last_state = st.state
    except KeyboardInterrupt:
        print()
    finally:
        try:
            s.close()
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("live", help="Stream CSI from ESP32, detect motion, publish events")
    pl.add_argument("--port", default=DEFAULT_PORT)
    pl.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    pl.add_argument("--window", type=int, default=WINDOW_SIZE)
    pl.add_argument("--baseline", type=float, default=BASELINE_SECS)
    pl.add_argument("--k-sigma", type=float, default=K_SIGMA)
    pl.add_argument("--threshold", type=float, default=ANOMALY_THRESHOLD)
    pl.add_argument("--broadcast", default=None,
                    help="URL del dashboard FastAPI (ej http://127.0.0.1:8000)")
    pl.add_argument("--metric", choices=["amp_mean", "amp_var"], default=DEFAULT_METRIC,
                    help="Metrica para detectar motion. amp_var = variance entre subcarriers "
                         "(default, mas sensible). amp_mean = amplitud media.")
    pl.add_argument("--source-mac", default=None,
                    help="Filtra a paquetes de UN solo MAC (ej. el BSSID de tu router). "
                         "Da signal mucho mas limpio. Ej: aa:bb:cc:dd:ee:ff (BSSID de tu router).")
    pl.set_defaults(func=cmd_live)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
