"""Multi-AP Wi-Fi radar - per-router directional motion sensing.

Single-AP RTT radar measures ONE invisible line (laptop -> router). For
directional sensing without specialized hardware, we periodically scan all
visible APs and track signal variance per BSSID. Each AP is a different
"ray" through different walls. By watching WHICH AP's signal fluctuates,
you infer which wall the motion is behind.

Limitations:
  - Windows rate-limits Wi-Fi scans (~1 scan per 3-4 seconds in practice).
  - Signal resolution is 1% (coarse vs RTT in ms).
  - You map BSSIDs to physical directions yourself (no compass on the PC).
  - Will be replaced by ESP32-S3 CSI when hardware arrives (proper AoA).

Standalone:
  python -m backend.multi_ap_radar list                  # one-shot, see all APs
  python -m backend.multi_ap_radar live                  # continuous, top 6
  python -m backend.multi_ap_radar live --top 10 --interval 3

Integrated with camera:
  python -m backend.camera_detector radar --wifi-radar --multi-ap
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from backend.wifi_radar import _decode_console

_REPO = Path(__file__).resolve().parent.parent
AP_LABELS_FILE = _REPO / "data" / "ap_labels.json"


def load_ap_labels() -> Dict[str, str]:
    if not AP_LABELS_FILE.exists():
        return {}
    try:
        raw = json.loads(AP_LABELS_FILE.read_text(encoding="utf-8"))
        return {str(k).lower(): str(v) for k, v in raw.items()}
    except Exception:
        return {}


def save_ap_labels(labels: Dict[str, str]) -> None:
    AP_LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    AP_LABELS_FILE.write_text(
        json.dumps(labels, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# Parse `netsh wlan show networks mode=bssid` output.
# Block layout:
#   SSID 1 : Movistar_ABCD
#       Network type ...
#       Authentication ...
#       Encryption ...
#       BSSID 1                 : 12:34:56:78:9a:bc
#            Signal             : 79%
#            Radio type ...

_SSID_LINE = re.compile(r"^SSID\s+\d+\s*:\s*(.*?)$", re.IGNORECASE)
_BSSID_LINE = re.compile(r"^BSSID\s+\d+\s*:\s*([0-9a-fA-F:]{17})$", re.IGNORECASE)
_SIGNAL_LINE = re.compile(r"^(?:Se\S+al|Signal)\s*:\s*(\d+)\s*%", re.IGNORECASE)


def scan_aps() -> Dict[str, dict]:
    """Run a Wi-Fi scan and return {bssid_lower: {ssid, signal_pct}}.

    Empty dict on error or no APs. Windows caches results ~3-4s; calling
    faster than that gives you the same data twice.
    """
    try:
        proc = subprocess.run(
            ["netsh", "wlan", "show", "networks", "mode=bssid"],
            capture_output=True, timeout=8.0,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}
    text = _decode_console(proc.stdout)
    if text is None:
        return {}

    aps: Dict[str, dict] = {}
    current_ssid = "(hidden)"
    current_bssid: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        m_ssid = _SSID_LINE.match(line)
        if m_ssid:
            current_ssid = m_ssid.group(1).strip() or "(hidden)"
            current_bssid = None
            continue
        m_bssid = _BSSID_LINE.match(line)
        if m_bssid:
            current_bssid = m_bssid.group(1).lower()
            aps[current_bssid] = {"ssid": current_ssid, "signal_pct": None}
            continue
        m_sig = _SIGNAL_LINE.match(line)
        if m_sig and current_bssid:
            aps[current_bssid]["signal_pct"] = int(m_sig.group(1))

    # Drop APs missing a signal reading
    return {b: v for b, v in aps.items() if v["signal_pct"] is not None}


@dataclass
class APState:
    bssid: str
    ssid: str
    signal_pct: int
    median: Optional[float] = None
    mad: Optional[float] = None
    anomaly_rate: float = 0.0
    state: str = "init"        # init|calibrating|warming|quiet|motion
    samples: int = 0
    last_seen: float = 0.0
    label: str = ""             # user-assigned room/wall label


class MultiAPRadar:
    """Per-BSSID robust motion detector (median + MAD on signal %)."""

    def __init__(self, window: int = 10, baseline_samples: int = 6,
                 k_sigma: float = 2.0, anomaly_threshold: float = 0.30,
                 mad_floor: float = 2.0):
        self.window = window
        self.baseline_samples_target = baseline_samples
        self.k_sigma = k_sigma
        self.threshold = anomaly_threshold
        self.mad_floor = mad_floor
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.baseline_buf: Dict[str, List[int]] = defaultdict(list)
        self.median: Dict[str, float] = {}
        self.mad: Dict[str, float] = {}
        self.labels: Dict[str, str] = load_ap_labels()

    def push(self, aps: Dict[str, dict]) -> Dict[str, APState]:
        now = time.time()
        out: Dict[str, APState] = {}

        for bssid, info in aps.items():
            sig = info["signal_pct"]
            if sig is None:
                continue
            self.history[bssid].append(sig)

            st = APState(
                bssid=bssid, ssid=info["ssid"], signal_pct=sig,
                samples=len(self.history[bssid]), last_seen=now,
                label=self.labels.get(bssid, ""),
            )

            # Phase 1: calibration
            if bssid not in self.median:
                self.baseline_buf[bssid].append(sig)
                if len(self.baseline_buf[bssid]) >= self.baseline_samples_target:
                    arr = np.array(self.baseline_buf[bssid], dtype=float)
                    m = float(np.median(arr))
                    md = max(self.mad_floor, 1.4826 * float(np.median(np.abs(arr - m))))
                    self.median[bssid] = m
                    self.mad[bssid] = md
                st.state = "calibrating"
                out[bssid] = st
                continue

            st.median = self.median[bssid]
            st.mad = self.mad[bssid]

            # Phase 2: warming
            if len(self.history[bssid]) < max(3, self.window // 2):
                st.state = "warming"
                out[bssid] = st
                continue

            # Phase 3: detect
            arr = np.array(self.history[bssid], dtype=float)
            deviations = np.abs(arr - self.median[bssid])
            anom = float(np.mean(deviations > self.k_sigma * self.mad[bssid]))
            st.anomaly_rate = anom
            st.state = "motion" if anom >= self.threshold else "quiet"
            out[bssid] = st
        return out


class MultiAPRadarThread:
    """Background thread that scans + updates the per-AP states."""

    def __init__(self, scan_interval: float = 4.0, **detector_kwargs):
        self.detector = MultiAPRadar(**detector_kwargs)
        self.scan_interval = scan_interval
        self._states: Dict[str, APState] = {}
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
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_states(self) -> Dict[str, APState]:
        with self._lock:
            return dict(self._states)

    def _run(self) -> None:
        while not self._stop.is_set():
            aps = scan_aps()
            if aps:
                new_states = self.detector.push(aps)
                with self._lock:
                    self._states = new_states
            self._stop.wait(self.scan_interval)


# ---------- CLI ----------

_STATE_COLORS = {
    "init":        "\033[90m",
    "calibrating": "\033[36m",
    "warming":     "\033[33m",
    "quiet":       "\033[32m",
    "motion":      "\033[91m",
}
_RESET = "\033[0m"


def cmd_list(_args) -> None:
    aps = scan_aps()
    if not aps:
        print("Sin APs visibles (o sin adaptador Wi-Fi).")
        return
    print(f"\n{'BSSID':<20} {'Signal':>7}  SSID")
    print("-" * 70)
    for bssid, info in sorted(aps.items(), key=lambda x: -x[1]["signal_pct"]):
        print(f"{bssid:<20} {info['signal_pct']:>6}%  {info['ssid']}")
    print(f"\nTotal: {len(aps)} APs visibles.\n")
    print("Para que estos sean 'rayos direccionales' utiles, anota mentalmente")
    print("donde esta cada router fisicamente (cocina, garaje, pieza vecina, etc).")
    print("Luego corre 'live' y mira que AP dispara motion cuando alguien cruza\n"
          "la pared correspondiente.\n")


def cmd_live(args) -> None:
    radar = MultiAPRadar(window=args.window, k_sigma=args.k_sigma,
                         anomaly_threshold=args.threshold,
                         baseline_samples=args.baseline_samples)
    print(f"Multi-AP Wi-Fi Radar | scan cada ~{args.interval:.0f}s "
          f"| top {args.top} APs | Ctrl+C para salir")
    print(f"Calibrando {args.baseline_samples} muestras por AP "
          f"(~{args.baseline_samples*args.interval:.0f}s). NO te muevas en la "
          f"casa durante este tiempo.\n")

    # Optional dashboard broadcast
    from backend.event_broadcast import init_broadcaster, publish as bc_publish
    bssid_to_anchor: Dict[str, str] = {}
    if args.broadcast:
        init_broadcaster(args.broadcast)
        try:
            anchors_path = _REPO / "data" / "anchors.json"
            if anchors_path.exists():
                cfg = json.loads(anchors_path.read_text(encoding="utf-8"))
                for a in cfg.get("anchors", []):
                    bssid = (a.get("match") or {}).get("bssid")
                    if bssid:
                        bssid_to_anchor[bssid.lower()] = a["id"]
                print(f"[broadcast] mapeados {len(bssid_to_anchor)} BSSIDs -> anchors\n")
        except Exception as exc:
            print(f"[broadcast] no se pudo cargar anchors.json: {exc}")
    last_published_state: Dict[str, str] = {}

    try:
        while True:
            t0 = time.time()
            aps = scan_aps()
            if not aps:
                print("[ERROR] No hay APs visibles. Esta el Wi-Fi activo?")
                time.sleep(args.interval)
                continue

            states = radar.push(aps)
            ordered = sorted(states.values(),
                             key=lambda s: -s.signal_pct)[:args.top]

            # Broadcast every known anchor's current signal_pct (drives auto-distance)
            if args.broadcast:
                for s in states.values():
                    aid = bssid_to_anchor.get(s.bssid.lower())
                    if not aid:
                        continue
                    is_motion = s.state == "motion"
                    prev = last_published_state.get(s.bssid)
                    if is_motion or s.state != prev:
                        bc_publish(
                            anchor_id=aid,
                            source="wifi_multi_ap",
                            kind="motion" if is_motion else "state",
                            state=s.state,
                            value=int(s.signal_pct),
                            extra={"ssid": s.ssid, "anomaly_rate": s.anomaly_rate,
                                   "label": s.label},
                        )
                        last_published_state[s.bssid] = s.state

            print(f"\n[{time.strftime('%H:%M:%S')}]  {len(states)} APs  (top {len(ordered)})")
            for s in ordered:
                color = _STATE_COLORS.get(s.state, "")
                tag = s.state.upper()[:4].ljust(4)
                m = f"{s.median:5.1f}" if s.median is not None else "  -- "
                mad = f"{s.mad:4.1f}" if s.mad is not None else " -- "
                # Show label if set, else SSID
                if s.label:
                    name_disp = f"** {s.label} **  ({s.ssid[:18]})"
                else:
                    name_disp = (s.ssid[:25] + "...") if len(s.ssid) > 28 else s.ssid
                marker = "  >>>" if s.state == "motion" else "     "
                print(f"  {color}[{tag}]{_RESET} {s.bssid}  "
                      f"sig={s.signal_pct:3d}%  base={m}+/-{mad}  "
                      f"anom={s.anomaly_rate*100:4.1f}%  {marker} {name_disp}")

            # Respect scan interval but compensate for time spent in scan
            elapsed = time.time() - t0
            time.sleep(max(0.5, args.interval - elapsed))
    except KeyboardInterrupt:
        print()


def cmd_label(args) -> None:
    labels = load_ap_labels()
    bssid = args.bssid.lower()
    name = (args.name or "").strip()
    if not name:
        if bssid in labels:
            removed = labels.pop(bssid)
            print(f"[ap] Eliminada etiqueta de {bssid} (era '{removed}').")
        else:
            print(f"[ap] {bssid} no tenia etiqueta.")
    else:
        prev = labels.get(bssid)
        labels[bssid] = name
        if prev:
            print(f"[ap] {bssid}: '{prev}' -> '{name}'")
        else:
            print(f"[ap] {bssid} -> '{name}'")
    save_ap_labels(labels)
    print(f"[ap] Guardado en {AP_LABELS_FILE}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="One-shot scan, show all APs")
    pl.set_defaults(func=cmd_list)

    pe = sub.add_parser("label", help="Asignar / quitar nombre amigable a un AP (BSSID)")
    pe.add_argument("bssid", help="BSSID (AA:BB:CC:DD:EE:FF)")
    pe.add_argument("name", nargs="?", default="",
                    help="Nombre amigable (ej 'Pared Cocina'). Vacio = borrar.")
    pe.set_defaults(func=cmd_label)

    pv = sub.add_parser("live", help="Continuous per-AP motion detection")
    pv.add_argument("--interval", type=float, default=4.0,
                    help="Seconds between scans (Windows caches ~3-4s)")
    pv.add_argument("--top", type=int, default=6,
                    help="Show only top-N strongest APs (default 6)")
    pv.add_argument("--window", type=int, default=10,
                    help="Sliding window size per AP (default 10 = ~40s of history)")
    pv.add_argument("--baseline-samples", type=int, default=6)
    pv.add_argument("--k-sigma", type=float, default=2.0,
                    help="Anomaly when signal differs from median by k*MAD (default 2.0)")
    pv.add_argument("--threshold", type=float, default=0.30,
                    help="Fraction of window with anomalies that triggers motion (0.30 = 30%%)")
    pv.add_argument("--broadcast", default=None,
                    help="URL del dashboard FastAPI (ej http://127.0.0.1:8000) para publicar "
                         "signal_pct por anchor labeled. Manda 'value' = 0..100.")
    pv.set_defaults(func=cmd_live)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
