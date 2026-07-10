"""Camera direction tags - link a wall the camera is pointing at to its anchors.

A "direction" is a named view that the camera is pointing at, paired with the
set of Wi-Fi APs and/or BLE devices that act as anchors behind that view.
When a direction is active, only those anchors count for the wall-motion
alert; the rest are dimmed in their panels.

Workflow:
  1. Point camera at a wall.
  2. Generate motion (ask someone to cross that wall).
  3. Press 't' while an anchor is firing.
  4. The active anchors are captured. Type a name in the terminal.
  5. Press 'n' later to cycle to that direction; the alert filters to it.

Saved as data/camera_directions.json:
{
  "Living-TV": {
    "ap_bssids": ["aa:bb:cc:dd:ee:01"],
    "ble_macs":  ["aa:bb:cc:dd:ee:02"]
  }
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
DIRECTIONS_FILE = _REPO / "data" / "camera_directions.json"


def load_directions() -> Dict[str, Dict[str, List[str]]]:
    if not DIRECTIONS_FILE.exists():
        return {}
    try:
        raw = json.loads(DIRECTIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, List[str]]] = {}
    for name, info in raw.items():
        if not isinstance(info, dict):
            continue
        ap_bssids = [str(b).lower() for b in info.get("ap_bssids", []) if isinstance(b, str)]
        ble_macs = [str(m).lower() for m in info.get("ble_macs", []) if isinstance(m, str)]
        out[str(name)] = {"ap_bssids": ap_bssids, "ble_macs": ble_macs}
    return out


def save_directions(dirs: Dict[str, Dict[str, List[str]]]) -> None:
    DIRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DIRECTIONS_FILE.write_text(
        json.dumps(dirs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
