"""Per-anchor RSSI calibration.

Default path-loss params (TX power, n=3) are generic averages. For real
accuracy, measure RSSI at a *known* distance for THIS specific device and
compute its actual TX power.

  python -m backend.calibrate_anchor tv_living --distance 5.0 --duration 60
  python -m backend.calibrate_anchor starlink_hotspot --distance 1.0 --duration 90

The script:
  1. Loads the anchor from data/anchors.json (by id)
  2. Listens for RSSI samples for `duration` seconds (BLE or Wi-Fi depending on type)
  3. Takes the median RSSI (robust to outliers)
  4. Solves the path-loss equation backwards for tx_power_at_1m
  5. Writes that field back into data/anchors.json
  6. The frontend uses it automatically next time the dashboard refreshes.

Note this calibrates ONE distance only. Indoor n (path-loss exponent) is
still assumed to be 3.0 — good enough most of the time, but if you want
the model more accurate, run the calibration with the anchor at TWO known
distances and we can fit n too. Tell me when you do.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent.parent
ANCHORS_FILE = _REPO / "data" / "anchors.json"
PATH_LOSS_N = 3.0


def _load_anchors() -> dict:
    return json.loads(ANCHORS_FILE.read_text(encoding="utf-8"))


def _save_anchors(cfg: dict) -> None:
    ANCHORS_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                            encoding="utf-8")


async def _collect_ble_rssi(target_mac: str, duration: float) -> list[int]:
    """Listen to BleakScanner advertisements for `duration` seconds.

    Returns the list of RSSI values seen for the given MAC.
    """
    try:
        from bleak import BleakScanner
    except ImportError:
        print("[calibrate] bleak no instalado. Corre: pip install bleak")
        return []

    target = target_mac.lower()
    samples: list[int] = []

    def cb(device, ad):
        if device.address.lower() != target:
            return
        try:
            r = int(ad.rssi)
            samples.append(r)
            print(f"  BLE sample #{len(samples)}: {r} dBm")
        except Exception:
            pass

    print(f"[calibrate] Escuchando BLE durante {duration:.0f}s para MAC {target}")
    async with BleakScanner(cb):
        await asyncio.sleep(duration)
    return samples


def _collect_wifi_rssi(target_bssid: str, duration: float) -> list[int]:
    """Scan Wi-Fi periodically for `duration` seconds.

    Returns the list of RSSI (dBm) values for the given BSSID.
    Note: each scan takes ~4s on Windows; you'll get duration/4 samples max.
    """
    from backend.wifi_scanner import WiFiScanner
    scanner = WiFiScanner()
    target = target_bssid.lower()
    samples: list[int] = []
    t_end = time.time() + duration
    while time.time() < t_end:
        try:
            aps = scanner.scan_networks()
        except Exception as exc:
            print(f"  scan error: {exc}")
            time.sleep(2.0)
            continue
        for ap in aps:
            if ap.bssid.lower() == target:
                # Convert % to dBm using the same formula as the frontend
                rssi = int(max(-100, min(-30, ap.signal_percent / 2 - 100)))
                samples.append(rssi)
                print(f"  Wi-Fi sample #{len(samples)}: signal={ap.signal_percent}%  rssi={rssi} dBm")
                break
        time.sleep(0.5)
    return samples


def _compute_tx_power(median_rssi: float, distance_m: float) -> float:
    """Solve path-loss for tx_power_at_1m given a known distance."""
    return median_rssi + 10 * PATH_LOSS_N * math.log10(distance_m)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("anchor_id", help="anchor id de data/anchors.json")
    p.add_argument("--distance", type=float, required=True,
                   help="Distancia real laptop->anchor (metros)")
    p.add_argument("--duration", type=float, default=60.0,
                   help="Tiempo de muestreo en segundos (default 60)")
    p.add_argument("--dry-run", action="store_true",
                   help="No guarda anchors.json, solo muestra el valor calculado")
    args = p.parse_args()

    cfg = _load_anchors()
    anchor = next((a for a in cfg.get("anchors", []) if a.get("id") == args.anchor_id), None)
    if anchor is None:
        print(f"[calibrate] Anchor '{args.anchor_id}' no esta en anchors.json")
        sys.exit(1)

    a_type = anchor.get("type", "")
    match = anchor.get("match") or {}
    print(f"[calibrate] Anchor: {anchor.get('name')}  type={a_type}  match={match}")
    print(f"[calibrate] Distancia conocida: {args.distance:.2f} m")
    print(f"[calibrate] PONE el dispositivo a esa distancia EXACTA del laptop AHORA "
          f"y no lo muevas durante {args.duration:.0f}s\n")
    time.sleep(2.0)

    samples: list[int] = []
    if a_type.startswith("ble"):
        mac = match.get("mac", "")
        if not mac:
            print("[calibrate] El anchor BLE no tiene match.mac")
            sys.exit(1)
        samples = asyncio.run(_collect_ble_rssi(mac, args.duration))
    elif a_type == "wifi_ap":
        bssid = match.get("bssid", "")
        if not bssid:
            print("[calibrate] El anchor wifi_ap no tiene match.bssid")
            sys.exit(1)
        samples = _collect_wifi_rssi(bssid, args.duration)
    else:
        print(f"[calibrate] Tipo de anchor no soportado: {a_type}")
        sys.exit(1)

    if len(samples) < 5:
        print(f"\n[calibrate] Solo {len(samples)} samples. Necesito al menos 5 para "
              f"un median confiable. El anchor no esta visible o el scan no funciono.")
        sys.exit(1)

    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0
    tx_power = _compute_tx_power(median, args.distance)

    print(f"\n=== RESULTADO ===")
    print(f"  Samples capturados: {len(samples)}")
    print(f"  RSSI median:  {median:6.1f} dBm")
    print(f"  RSSI mean:    {mean:6.1f} dBm  (stdev {stdev:.2f})")
    print(f"  Computed tx_power_at_1m: {tx_power:6.1f} dBm  (para path-loss n={PATH_LOSS_N})")
    default_tx = -55.0 if a_type.startswith("ble") else -40.0
    print(f"  (Default generico era {default_tx} dBm. Diferencia: {tx_power - default_tx:+.1f} dB)")

    if args.dry_run:
        print(f"\n[dry-run] No guarde nada. Para aplicar correr sin --dry-run.")
        return

    anchor["tx_power_at_1m"] = round(tx_power, 1)
    _save_anchors(cfg)
    print(f"\n[calibrate] Guardado en anchors.json: {args.anchor_id}.tx_power_at_1m = {round(tx_power, 1)}")
    print(f"[calibrate] Refresca el browser para usar el valor calibrado.")


if __name__ == "__main__":
    main()
