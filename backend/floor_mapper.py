"""Map foot-pixel position to real-world floor coordinates (meters).

Two modes:
  - Calibrated: 4-point homography via cv2.getPerspectiveTransform. Accurate.
  - Naive:      X = linear from horizontal center; Y = 1 - py/H (foot bajo = cerca).
                (Esto es exactamente lo que describiste como "Homografía Básica".)

Calibration file: data/floor_calibration.json
{
  "image_points": [[x1,y1], ...],   # 4 pixel points clicked on a captured frame
  "world_points": [[X1,Y1], ...],   # the same 4 points in meters on the floor
  "width_m":  4.0,
  "length_m": 6.0
}

The 4 points must be in the SAME ORDER in both arrays (e.g. TL, TR, BR, BL).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
CALIBRATION_FILE = _REPO / "data" / "floor_calibration.json"


@dataclass
class FloorMapper:
    width_m: float = 4.0
    length_m: float = 6.0
    homography: Optional[np.ndarray] = None

    def pixel_to_world(
        self, px: float, py: float, frame_w: int, frame_h: int
    ) -> Tuple[float, float]:
        if self.homography is not None:
            pt = np.array([[[px, py]]], dtype=np.float32)
            warped = cv2.perspectiveTransform(pt, self.homography)
            return float(warped[0, 0, 0]), float(warped[0, 0, 1])
        # Naive: foot at bottom of frame = closer to camera (Y small)
        x = (px / max(1, frame_w)) * self.width_m
        y = (1.0 - py / max(1, frame_h)) * self.length_m
        return x, y

    @classmethod
    def load(cls, path: Path = CALIBRATION_FILE) -> "FloorMapper":
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        img_pts = np.array(data["image_points"], dtype=np.float32)
        wld_pts = np.array(data["world_points"], dtype=np.float32)
        H = cv2.getPerspectiveTransform(img_pts, wld_pts)
        return cls(
            width_m=float(data.get("width_m", 4.0)),
            length_m=float(data.get("length_m", 6.0)),
            homography=H,
        )

    @staticmethod
    def save_calibration(
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]],
        width_m: float,
        length_m: float,
        path: Path = CALIBRATION_FILE,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "image_points": [list(p) for p in image_points],
                    "world_points": [list(p) for p in world_points],
                    "width_m": width_m,
                    "length_m": length_m,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[floor] Calibracion guardada -> {path}")

    # ------------- visualization -------------

    def render_map(
        self,
        positions: Dict[str, Dict],
        size_px: int = 520,
        bg: Tuple[int, int, int] = (15, 15, 22),
        title: str = "Mapa cartesiano",
    ) -> np.ndarray:
        canvas = np.full((size_px, size_px, 3), bg, dtype=np.uint8)

        # grid
        n = 10
        for i in range(n + 1):
            v = int(i * (size_px - 1) / n)
            cv2.line(canvas, (v, 0), (v, size_px), (40, 40, 55), 1)
            cv2.line(canvas, (0, v), (size_px, v), (40, 40, 55), 1)

        # frame
        cv2.rectangle(canvas, (0, 0), (size_px - 1, size_px - 1), (110, 110, 130), 2)

        # axes labels
        cv2.putText(canvas, title, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 230), 1)
        cv2.putText(canvas, f"X: 0 -> {self.width_m:.1f} m",
                    (10, size_px - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 180), 1)
        cv2.putText(canvas, f"Y: 0 -> {self.length_m:.1f} m",
                    (10, size_px - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 180), 1)

        # camera marker (origin assumed at bottom-center of mapped area)
        cam = (size_px // 2, size_px - 6)
        cv2.drawMarker(canvas, cam, (0, 200, 255),
                       markerType=cv2.MARKER_TRIANGLE_UP, markerSize=14, thickness=2)

        # plot persons
        for label, info in positions.items():
            x, y = info["xy"]
            color = info.get("color", (180, 180, 180))
            cx = int(np.clip(x / max(0.001, self.width_m), 0, 1) * (size_px - 1))
            cy = int((1.0 - np.clip(y / max(0.001, self.length_m), 0, 1)) * (size_px - 1))

            # halo + dot
            cv2.circle(canvas, (cx, cy), 14, (255, 255, 255), 1)
            cv2.circle(canvas, (cx, cy), 10, color, -1)
            cv2.putText(canvas, label, (cx + 14, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"({x:.1f},{y:.1f})",
                        (cx + 14, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 170), 1, cv2.LINE_AA)
        return canvas
