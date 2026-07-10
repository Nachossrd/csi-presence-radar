"""BLE Radar - passive Bluetooth Low Energy presence + motion detector.

Modern phones broadcast BLE advertisements every 1-3s (Apple Continuity,
Google Fast Pair, etc.) even with the Bluetooth toggle "off". The laptop
listens passively (no pairing, no admin) and measures RSSI per MAC. When
someone with a phone in their pocket walks closer to the wall in front of
the laptop, the BLE RSSI rises sharply because BLE 2.4GHz attenuates only
~5-15 dB through typical drywall.

Why this complements Wi-Fi + camera:
  - BLE tells you WHO is close (per-device, optionally per-person via labels).
  - Wi-Fi tells you THAT motion happens along a specific ray (anchor-based).
  - Camera tells you WHAT you actually see (faces, identities, positions).

Caveats:
  - iOS/Android randomize BLE MAC every ~15 min by default. A "Tata" label
    will eventually go stale; re-label or pair the phone for a fixed MAC.
  - Some devices broadcast a local name we can use as a stable hint.
  - Detection sensitivity scales with the device's advertising rate.

Install:
  pip install bleak

Standalone:
  python -m backend.ble_radar scan                 # one-shot 10s scan
  python -m backend.ble_radar live                 # continuous monitoring
  python -m backend.ble_radar label <mac> "Tata"   # label a MAC
  python -m backend.ble_radar label <mac>          # remove label

Integrated with camera:
  python -m backend.camera_detector radar --ble-radar
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
LABELS_FILE = _REPO / "data" / "ble_devices.json"


def load_labels() -> Dict[str, str]:
    if not LABELS_FILE.exists():
        return {}
    try:
        raw = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
        return {str(k).lower(): str(v) for k, v in raw.items()}
    except Exception:
        return {}


def save_labels(labels: Dict[str, str]) -> None:
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    LABELS_FILE.write_text(
        json.dumps(labels, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


@dataclass
class BLEState:
    mac: str
    name: str = ""              # device-broadcast name
    label: str = ""             # user-assigned friendly name
    rssi: int = 0
    median: Optional[float] = None
    mad: Optional[float] = None
    delta: float = 0.0          # current rssi - baseline median (+ = closer)
    anomaly_rate: float = 0.0
    state: str = "init"         # init|calibrating|warming|quiet|moving|approaching
    samples: int = 0
    last_seen: float = 0.0


class BLERadar:
    """Per-MAC robust motion detector on BLE RSSI (dBm)."""

    def __init__(self, window: int = 20, baseline_samples: int = 10,
                 k_sigma: float = 3.0, anomaly_threshold: float = 0.30,
                 mad_floor: float = 2.0):
        self.window_size = window
        self.baseline_target = baseline_samples
        self.k_sigma = k_sigma
        self.anomaly_threshold = anomaly_threshold
        self.mad_floor = mad_floor

        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.baseline_buf: Dict[str, List[int]] = defaultdict(list)
        self.median: Dict[str, float] = {}
        self.mad: Dict[str, float] = {}
        self.names: Dict[str, str] = {}
        self.labels: Dict[str, str] = load_labels()

    def push(self, mac: str, rssi: int, name: str = "") -> BLEState:
        now = time.time()
        mac = mac.lower()
        self.history[mac].append(rssi)
        if name and (not self.names.get(mac)):
            self.names[mac] = name

        st = BLEState(
            mac=mac,
            name=self.names.get(mac, ""),
            label=self.labels.get(mac, ""),
            rssi=rssi,
            samples=len(self.history[mac]),
            last_seen=now,
        )

        # Phase 1: calibration
        if mac not in self.median:
            self.baseline_buf[mac].append(rssi)
            if len(self.baseline_buf[mac]) >= self.baseline_target:
                arr = np.array(self.baseline_buf[mac], dtype=float)
                m = float(np.median(arr))
                md = max(self.mad_floor, 1.4826 * float(np.median(np.abs(arr - m))))
                self.median[mac] = m
                self.mad[mac] = md
            st.state = "calibrating"
            return st

        st.median = self.median[mac]
        st.mad = self.mad[mac]
        st.delta = float(rssi) - self.median[mac]

        # Phase 2: warming
        if len(self.history[mac]) < max(5, self.window_size // 2):
            st.state = "warming"
            return st

        # Phase 3: detect
        arr = np.array(self.history[mac], dtype=float)
        dev = arr - self.median[mac]
        approaching_rate = float(np.mean(dev > self.k_sigma * self.mad[mac]))
        moving_rate = float(np.mean(np.abs(dev) > self.k_sigma * self.mad[mac]))
        st.anomaly_rate = moving_rate

        if approaching_rate >= self.anomaly_threshold:
            st.state = "approaching"
        elif moving_rate >= self.anomaly_threshold:
            st.state = "moving"
        else:
            st.state = "quiet"
        return st


class BLERadarThread:
    """Background thread running bleak's asyncio scanner."""

    def __init__(self, **detector_kwargs):
        self.radar = BLERadar(**detector_kwargs)
        self._states: Dict[str, BLEState] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.error: Optional[str] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def get_states(self) -> Dict[str, BLEState]:
        with self._lock:
            return dict(self._states)

    def reload_labels(self) -> None:
        self.radar.labels = load_labels()

    def _thread_main(self) -> None:
        try:
            from bleak import BleakScanner
        except ImportError:
            self.error = "bleak no esta instalado. Corre: pip install bleak"
            print(f"[ble] {self.error}")
            return

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main(BleakScanner))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[ble] Scanner error: {self.error}")
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    async def _async_main(self, BleakScanner) -> None:
        def callback(device, advertising_data):
            try:
                rssi = int(advertising_data.rssi)
            except (TypeError, ValueError, AttributeError):
                return
            mac = device.address.lower()
            name = device.name or ""
            if not name:
                try:
                    name = advertising_data.local_name or ""
                except Exception:
                    name = ""
            with self._lock:
                st = self.radar.push(mac, rssi, name)
                self._states[mac] = st

        try:
            async with BleakScanner(callback) as scanner:
                _ = scanner  # keep reference
                while not self._stop_event.is_set():
                    await asyncio.sleep(0.5)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[ble] Scanner stopped: {self.error}")


# ---------------- CLI ----------------

_STATE_COLORS = {
    "init":        "\033[90m",
    "calibrating": "\033[36m",
    "warming":     "\033[33m",
    "quiet":       "\033[32m",
    "moving":      "\033[91m",
    "approaching": "\033[1;91m",
}
_RESET = "\033[0m"
_STATE_TAGS = {
    "init":        "INIT",
    "calibrating": "CAL ",
    "warming":     "WARM",
    "quiet":       "OK  ",
    "moving":      "MOVE",
    "approaching": "NEAR",
}


def cmd_scan(args) -> None:
    """One-shot N-second scan, list all visible devices."""
    try:
        from bleak import BleakScanner
    except ImportError:
        print("bleak no esta instalado. Corre: pip install bleak")
        return

    print(f"Escaneando BLE durante {args.duration:.0f}s...\n")

    async def do_scan():
        devices = {}

        def cb(device, ad):
            devices[device.address.lower()] = (device, ad)

        async with BleakScanner(cb):
            await asyncio.sleep(args.duration)
        return devices

    try:
        devices = asyncio.run(do_scan())
    except Exception as exc:
        print(f"[ble] ERROR: {type(exc).__name__}: {exc}")
        return

    labels = load_labels()
    if not devices:
        print("Sin dispositivos BLE detectados.")
        print("Posibles causas: Bluetooth apagado en la laptop, drivers, sin dispositivos cerca.")
        return

    ordered = sorted(devices.values(), key=lambda x: -(x[1].rssi or -100))
    print(f"{'MAC':<19} {'RSSI':>6}  {'Etiqueta':<18} Nombre / datos")
    print("-" * 80)
    for device, ad in ordered:
        mac = device.address.lower()
        label = labels.get(mac, "")
        name = device.name or ad.local_name or ""
        rssi = ad.rssi if ad.rssi is not None else -100
        print(f"{mac:<19} {rssi:>4}dB  {label:<18} {name}")

    print(f"\nTotal: {len(devices)} dispositivos BLE visibles.")
    print(f"\nPara etiquetar: python -m backend.ble_radar label <mac> \"Nombre\"")


def cmd_live(args) -> None:
    th = BLERadarThread(
        window=args.window, baseline_samples=args.baseline_samples,
        k_sigma=args.k_sigma, anomaly_threshold=args.threshold,
    )
    th.start()
    print(f"BLE Radar | window={args.window} k={args.k_sigma}sigma "
          f"thr={args.threshold*100:.0f}% | Ctrl+C para salir\n")
    print("Esperando muestras. Cada dispositivo se calibra solo cuando junta "
          f"{args.baseline_samples} muestras (depende de su tasa de advertising).\n")

    # Optional dashboard broadcast: map MAC -> anchor_id via data/anchors.json
    from backend.event_broadcast import init_broadcaster, publish as bc_publish
    mac_to_anchor: Dict[str, str] = {}
    if args.broadcast:
        init_broadcaster(args.broadcast)
        try:
            import json as _json
            from pathlib import Path as _Path
            anchors_path = _Path(__file__).resolve().parent.parent / "data" / "anchors.json"
            if anchors_path.exists():
                cfg = _json.loads(anchors_path.read_text(encoding="utf-8"))
                for a in cfg.get("anchors", []):
                    mac = (a.get("match") or {}).get("mac")
                    if mac:
                        mac_to_anchor[mac.lower()] = a["id"]
                print(f"[broadcast] mapeados {len(mac_to_anchor)} MACs -> anchors\n")
        except Exception as exc:
            print(f"[broadcast] no se pudo cargar anchors.json: {exc}")
    last_published_state: Dict[str, str] = {}

    try:
        while True:
            time.sleep(args.interval)
            if th.error:
                print(f"[!] {th.error}")
                return
            states = th.get_states()
            if not states:
                continue

            ordered = sorted(states.values(), key=lambda s: -s.rssi)[:args.top]

            # Broadcast every known anchor's current RSSI (drives auto-distance ring)
            if args.broadcast:
                for s in states.values():
                    aid = mac_to_anchor.get(s.mac.lower())
                    if not aid:
                        continue
                    is_motion = s.state in ("motion", "moving", "approaching")
                    prev = last_published_state.get(s.mac)
                    # publish on every poll if motion, else only on state changes
                    if is_motion or s.state != prev:
                        bc_publish(
                            anchor_id=aid,
                            source="ble",
                            kind="motion" if is_motion else "state",
                            state=s.state,
                            value=int(s.rssi),
                            delta=s.delta,
                            extra={"label": s.label, "anomaly_rate": s.anomaly_rate},
                        )
                        last_published_state[s.mac] = s.state

            print(f"\n[{time.strftime('%H:%M:%S')}]  {len(states)} disp. BLE  (top {len(ordered)})")
            for s in ordered:
                color = _STATE_COLORS.get(s.state, "")
                tag = _STATE_TAGS.get(s.state, "????")
                base_str = (f"{s.median:5.1f}+/-{s.mad:4.1f}"
                            if s.median is not None else "  --  +/-  --")
                delta_str = f"{s.delta:+5.1f}" if s.median is not None else "    -"
                marker = "  >>>" if s.state in ("approaching", "moving") else "     "
                display_name = s.label or s.name or "(sin nombre)"
                if s.label:
                    display_name = f"** {s.label} **"
                print(f"  {color}[{tag}]{_RESET} {s.mac}  "
                      f"rssi={s.rssi:4d}dB  base={base_str}  delta={delta_str}dB  "
                      f"anom={s.anomaly_rate*100:3.0f}%  {marker}  {display_name}")
    except KeyboardInterrupt:
        print()
    finally:
        th.stop()


def cmd_label(args) -> None:
    labels = load_labels()
    mac = args.mac.lower()
    name = (args.name or "").strip()
    if not name:
        if mac in labels:
            removed = labels.pop(mac)
            print(f"[ble] Eliminada etiqueta de {mac} (era '{removed}').")
        else:
            print(f"[ble] {mac} no tenia etiqueta.")
    else:
        prev = labels.get(mac)
        labels[mac] = name
        if prev:
            print(f"[ble] {mac}: '{prev}' -> '{name}'")
        else:
            print(f"[ble] {mac} -> '{name}'")
    save_labels(labels)
    print(f"[ble] Guardado en {LABELS_FILE}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("scan", help="Escaneo unico (N segundos), lista dispositivos")
    ps.add_argument("--duration", type=float, default=10.0)
    ps.set_defaults(func=cmd_scan)

    pv = sub.add_parser("live", help="Monitoreo continuo con deteccion de motion/proximidad")
    pv.add_argument("--top", type=int, default=8, help="Cuantos dispositivos mostrar")
    pv.add_argument("--window", type=int, default=20)
    pv.add_argument("--baseline-samples", type=int, default=10)
    pv.add_argument("--k-sigma", type=float, default=3.0,
                    help="Anomalia = rssi se desvia mas que k*MAD. Default 3.0")
    pv.add_argument("--threshold", type=float, default=0.30,
                    help="Fraccion del window con anomalias para gatillar motion (default 0.30)")
    pv.add_argument("--interval", type=float, default=1.5,
                    help="Cada cuanto refrescar la pantalla (segundos)")
    pv.add_argument("--broadcast", default=None,
                    help="URL del dashboard FastAPI (ej http://127.0.0.1:8000) para publicar "
                         "RSSI por anchor labeled. Manda 'value' = dBm.")
    pv.set_defaults(func=cmd_live)

    pl = sub.add_parser("label", help="Asignar / quitar nombre amigable a un MAC")
    pl.add_argument("mac", help="MAC del dispositivo (AA:BB:CC:DD:EE:FF)")
    pl.add_argument("name", nargs="?", default="",
                    help="Nombre amigable. Vacio para borrar la etiqueta.")
    pl.set_defaults(func=cmd_label)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
