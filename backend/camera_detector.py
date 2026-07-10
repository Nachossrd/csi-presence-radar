"""Webcam-based person detector using YOLOv8 + IoU tracker + face ID + radar map.

The raw YOLO output has false positives on:
  - Hands/arms close to lens that geometrically resemble a torso
  - Static objects (blankets, chair backs) that match person priors at mid conf

The tracker filters these by requiring N consecutive frames of a box at the
same location (IoU > threshold) before counting it as a "confirmed" person.

Subcommands:
  live      -- live window with bounding boxes (green=confirmed, yellow=candidate)
  count     -- headless count of CONFIRMED persons every N seconds
  test      -- 10-frame smoke test
  enroll    -- enroll faces from data/faces/<Name>/*.jpg into embeddings.npz
  calibrate -- click 4 floor points on a captured frame to set the homography
  radar     -- live view + 2D map of identified persons (Tata / Abuela / Yo)
"""

import argparse
import math
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PERSON_CLASS_ID = 0
DEFAULT_MODEL = "yolov8n.pt"  # nano: 6MB, ~250ms on CPU. Use yolov8s for accuracy.

# BGR colors per identity. Unknown / unmatched falls back to gray.
LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Tata":   (255, 140,   0),   # azul-cian
    "Abuela": (200,   0, 200),   # magenta
    "Yo":     ( 30, 220,  80),   # verde
    "Karen":  (  0, 220, 255),   # amarillo
}
DEFAULT_COLOR = (180, 180, 180)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class PersonTracker:
    """Light IoU-based multi-object tracker. Confirms a track after N hits."""

    def __init__(self, iou_threshold: float = 0.3, min_hits: int = 3, max_lost: int = 5):
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_lost = max_lost
        self.tracks: List[dict] = []
        self._next_id = 0

    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> None:
        matched_ids = set()
        unmatched_dets: List[Tuple[int, int, int, int, float]] = []

        for det in detections:
            best_track = None
            best_iou = self.iou_threshold
            for t in self.tracks:
                if t["id"] in matched_ids:
                    continue
                iou = _iou(det[:4], t["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track = t
            if best_track is not None:
                best_track["bbox"] = det[:4]
                best_track["conf"] = det[4]
                best_track["hits"] += 1
                best_track["lost"] = 0
                matched_ids.add(best_track["id"])
            else:
                unmatched_dets.append(det)

        for t in self.tracks:
            if t["id"] not in matched_ids:
                t["lost"] += 1

        for det in unmatched_dets:
            self.tracks.append({
                "id": self._next_id,
                "bbox": det[:4],
                "conf": det[4],
                "hits": 1,
                "lost": 0,
                # identity slots (populated by FaceIdentifier in radar mode)
                "label": None,
                "label_score": -1.0,
            })
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t["lost"] <= self.max_lost]

    def confirmed(self) -> List[dict]:
        return [t for t in self.tracks if t["hits"] >= self.min_hits and t["lost"] == 0]

    def candidates(self) -> List[dict]:
        return [t for t in self.tracks if t["hits"] < self.min_hits and t["lost"] == 0]


def _load_model(name: str):
    from ultralytics import YOLO
    print(f"Loading YOLO model ({name})...")
    return YOLO(name)


def _open_camera(preferred: int, max_probe: int = 4,
                 width: int = 0, height: int = 0) -> Optional[cv2.VideoCapture]:
    """Try preferred index first; if it can open but not read a frame, probe others.

    If width/height > 0, request that resolution from the driver. Most webcams
    will honor it. Lower resolution = much faster YOLO + face inference.
    """
    order = [preferred] + [i for i in range(max_probe) if i != preferred]
    for idx in order:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        if width > 0 and height > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        time.sleep(0.4)
        ok, frame = cap.read()
        if ok:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if idx != preferred:
                print(f"[INFO] Camera index {preferred} did not deliver frames. Using index {idx} instead.")
            print(f"[INFO] Camera abierta @ {actual_w}x{actual_h}")
            return cap
        cap.release()
    print(f"[ERROR] No working camera found in indices 0..{max_probe-1}")
    return None


def _detect(model, frame, conf: float, imgsz: int = 640) -> List[Tuple[int, int, int, int, float]]:
    results = model(frame, classes=[PERSON_CLASS_ID], conf=conf,
                    imgsz=imgsz, verbose=False)
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            c = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, c))
    return boxes


def cmd_live(args) -> None:
    model = _load_model(args.model)
    tracker = PersonTracker(min_hits=args.min_hits, max_lost=args.max_lost)
    cap = _open_camera(args.camera)
    if cap is None:
        return
    print(f"Camera open. conf>={args.conf}, min_hits={args.min_hits}, max_lost={args.max_lost}")
    print("'q' to quit.")

    fps_t0 = time.time()
    fps_frames = 0
    fps_disp = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            raw = _detect(model, frame, args.conf)
            tracker.update(raw)
            confirmed = tracker.confirmed()
            candidates = tracker.candidates()

            for t in candidates:
                x1, y1, x2, y2 = t["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
                cv2.putText(frame, f"? {t['conf']:.2f} ({t['hits']}/{args.min_hits})",
                            (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

            for t in confirmed:
                x1, y1, x2, y2 = t["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"#{t['id']} {t['conf']:.2f}",
                            (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            n_conf = len(confirmed)
            n_cand = len(candidates)
            color = (0, 255, 255) if n_conf else (120, 120, 120)
            cv2.putText(frame, f"Personas: {n_conf}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            if n_cand:
                cv2.putText(frame, f"(+{n_cand} candidato/s)", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            fps_frames += 1
            if time.time() - fps_t0 >= 1.0:
                fps_disp = fps_frames / (time.time() - fps_t0)
                fps_t0 = time.time()
                fps_frames = 0
            cv2.putText(frame, f"{fps_disp:.1f} fps  {args.model}",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Radar - Camera Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def cmd_count(args) -> None:
    model = _load_model(args.model)
    tracker = PersonTracker(min_hits=args.min_hits, max_lost=args.max_lost)
    cap = _open_camera(args.camera)
    if cap is None:
        return
    print(f"Counting confirmed persons every {args.interval}s. Ctrl+C to stop.\n")

    try:
        last_print = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw = _detect(model, frame, args.conf)
            tracker.update(raw)

            now = time.time()
            if now - last_print >= args.interval:
                n = len(tracker.confirmed())
                cand = len(tracker.candidates())
                ts = time.strftime("%H:%M:%S")
                print(f"  [{ts}] confirmados={n}  candidatos={cand}")
                last_print = now
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


def cmd_test(args) -> None:
    """Capture 10 frames at 250ms intervals, run tracker, report final state."""
    model = _load_model(args.model)
    tracker = PersonTracker(min_hits=args.min_hits, max_lost=args.max_lost)
    cap = _open_camera(args.camera)
    if cap is None:
        return
    print(f"Capturing 10 frames @ 250ms (conf>={args.conf}, min_hits={args.min_hits})")
    try:
        for i in range(10):
            ok, frame = cap.read()
            if not ok:
                continue
            raw = _detect(model, frame, args.conf)
            tracker.update(raw)
            conf_n = len(tracker.confirmed())
            cand_n = len(tracker.candidates())
            print(f"  Frame {i+1:2d}: raw={len(raw)}  confirmados={conf_n}  candidatos={cand_n}")
            time.sleep(0.25)
        print(f"\nFinal: {len(tracker.confirmed())} persona/s confirmadas")
        for t in tracker.confirmed():
            print(f"  id={t['id']}  bbox={t['bbox']}  conf={t['conf']:.2f}  hits={t['hits']}")
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Enroll faces  (data/faces/<Name>/*.jpg  ->  data/faces/embeddings.npz)
# ---------------------------------------------------------------------------

def cmd_enroll(_args) -> None:
    from backend.face_identifier import FaceIdentifier
    fid = FaceIdentifier()
    counts = fid.enroll_from_folders()
    if not counts:
        print("[enroll] No se registro ninguna identidad. Nada que guardar.")
        return
    fid.save()
    total = sum(counts.values())
    print(f"\n[enroll] OK. {total} embeddings de {len(counts)} identidades:")
    for name, n in counts.items():
        print(f"  - {name}: {n} fotos")


# ---------------------------------------------------------------------------
# Calibrate floor:  click 4 floor points on a frame, type real-world meters
# ---------------------------------------------------------------------------

def cmd_calibrate(args) -> None:
    from backend.floor_mapper import FloorMapper

    cap = _open_camera(args.camera)
    if cap is None:
        return

    # Discard a few frames so autoexposure settles
    for _ in range(8):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("[calibrate] No pude capturar un frame.")
        return

    snapshot = frame.copy()
    clicks: List[Tuple[int, int]] = []
    window = "Calibracion - haz click en 4 puntos del piso (orden: TL, TR, BR, BL)"

    def on_click(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_click)

    instructions = [
        "Clic 1: esquina superior-izquierda del rectangulo del piso",
        "Clic 2: esquina superior-derecha",
        "Clic 3: esquina inferior-derecha (mas cerca de la camara)",
        "Clic 4: esquina inferior-izquierda (mas cerca de la camara)",
        "Despues responde el tamano real del rectangulo en la consola.",
        "ESC para cancelar.",
    ]

    print("\n=== Calibracion de piso ===")
    for line in instructions:
        print(" - " + line)

    while True:
        view = snapshot.copy()
        for i, (px, py) in enumerate(clicks):
            cv2.circle(view, (px, py), 6, (0, 255, 255), -1)
            cv2.putText(view, str(i + 1), (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if len(clicks) >= 2:
            cv2.polylines(view, [np.array(clicks, dtype=np.int32)],
                          isClosed=(len(clicks) == 4), color=(0, 255, 0), thickness=1)

        cv2.putText(view, f"Puntos: {len(clicks)}/4  (ENTER para continuar, ESC para abortar)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(window, view)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # ESC
            cv2.destroyAllWindows()
            print("[calibrate] Cancelado.")
            return
        if k in (13, 10) and len(clicks) == 4:  # ENTER
            break

    cv2.destroyAllWindows()

    try:
        width_m = float(input("Ancho real del rectangulo (eje X, metros): ").strip())
        length_m = float(input("Largo real del rectangulo (eje Y, metros): ").strip())
    except ValueError:
        print("[calibrate] Numero invalido. Abortado.")
        return

    # World coordinates in the same order as clicks:
    # TL = (0, length), TR = (width, length), BR = (width, 0), BL = (0, 0)
    world_points = [(0.0, length_m), (width_m, length_m), (width_m, 0.0), (0.0, 0.0)]
    FloorMapper.save_calibration(
        image_points=clicks,
        world_points=world_points,
        width_m=width_m,
        length_m=length_m,
    )
    print("[calibrate] Listo. Ahora ejecuta:  python -m backend.camera_detector radar")


# ---------------------------------------------------------------------------
# Radar:  YOLO + face ID + 2D cartesian map (top-down)
# ---------------------------------------------------------------------------

def _identify_track(face_id, frame, track, frame_w, frame_h, retry_every: int = 8) -> None:
    """Run face recognition on the person crop, with throttling for ALL tracks.

    Bug fix from prior version: the old throttle only applied to tracks that
    already had a confident label. Unknown tracks were re-identified every
    frame, which on HD camera input could add 100ms+ per person per frame.
    Now we throttle uniformly; confident tracks re-check 3x less often.
    """
    track["frames_since_id"] = track.get("frames_since_id", retry_every) + 1

    has_confident = (track.get("label") is not None
                     and track.get("label_score", 0.0) >= 0.5)
    threshold = retry_every * 3 if has_confident else retry_every
    if track["frames_since_id"] < threshold:
        return

    x1, y1, x2, y2 = track["bbox"]
    # Crop the upper 2/3 of the bounding box where the face usually is
    cy_split = y1 + int((y2 - y1) * 0.65)
    cx1, cy1, cx2, cy2 = (
        max(0, x1), max(0, y1),
        min(frame_w, x2), min(frame_h, cy_split),
    )
    if cx2 - cx1 < 30 or cy2 - cy1 < 30:
        return
    crop = frame[cy1:cy2, cx1:cx2]

    name, score, _ = face_id.identify_in_crop(crop)
    track["frames_since_id"] = 0
    if name is not None and score > track.get("label_score", -1.0):
        track["label"] = name
        track["label_score"] = score


def _draw_anchor_listen_panel(frame, anchor_states, n_visible: int,
                              connected: bool, anchors_meta: Optional[dict] = None) -> bool:
    """Overlay listing live anchor states from the dashboard WebSocket.

    Returns True if "through-wall motion" was detected (= any motion + no
    person visible in camera), so the caller can paint a big alert.
    """
    h, w = frame.shape[:2]
    panel_w = 320
    row_h = 22
    n_rows = max(1, len(anchor_states))
    panel_h = 32 + n_rows * row_h + 8
    # Bottom-right
    x0 = w - panel_w - 12
    y0 = h - panel_h - 12

    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 18, 28), -1)
    border = (60, 220, 60) if connected else (90, 90, 110)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), border, 1)

    header = "Anchors (vivo)" if connected else "Anchors (desconectado)"
    cv2.putText(frame, header, (x0 + 10, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 230, 240), 1, cv2.LINE_AA)

    motion_anchors = []
    for i, st in enumerate(anchor_states):
        y = y0 + 40 + i * row_h
        is_motion = st.state in ("motion", "moving", "approaching")
        if is_motion:
            motion_anchors.append(st)
        color = (0, 80, 255) if is_motion else (
            (0, 200, 255) if st.state in ("calibrating", "warming") else (60, 220, 60)
        )
        name = st.anchor_id
        if anchors_meta:
            name = anchors_meta.get(st.anchor_id, name)
        if len(name) > 25:
            name = name[:24] + ".."
        line = f"{st.state.upper()[:8]:<8} {name}"
        cv2.putText(frame, line, (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
        cv2.circle(frame, (x0 + panel_w - 16, y - 4), 5, color, -1)

    # Honest motion alert: small badges over the side panel header, NOT a fake
    # silhouette pretending to track position. With 1 ESP32 we have NO
    # direction info, so we only report "motion detected at anchor X".
    if motion_anchors:
        _draw_motion_badges(frame, motion_anchors, x0, y0, panel_w, anchors_meta)
    return bool(motion_anchors)


def _draw_motion_badges(frame, motion_anchors, panel_x0, panel_y0, panel_w,
                        anchors_meta=None) -> None:
    """Stack a small ⚠ badge above the anchors panel for each firing anchor.

    Honest indicator. No fake animation that misleads the user. Just:
    'anchor X reports motion, last seen Ys ago'.
    """
    h, w = frame.shape[:2]
    pulse = 0.6 + 0.4 * abs(math.sin(time.time() * 3.0))
    bright = (60, 140, 255)
    base = (0, 60, 200)
    now = time.time()

    badge_h = 30
    spacing = 6
    total_h = len(motion_anchors) * (badge_h + spacing)
    y = panel_y0 - total_h - 8

    for a in motion_anchors[:5]:
        name = (anchors_meta or {}).get(a.anchor_id, a.anchor_id)
        if len(name) > 30:
            name = name[:29] + ".."
        secs_ago = max(0.0, now - (a.last_event or now))
        line = f"!  {name}    ({secs_ago:.0f}s)"
        (tw, th_), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bg_w = min(panel_w, max(tw + 28, 160))
        bx = panel_x0 + (panel_w - bg_w)  # right-align with panel
        col = bright if pulse > 0.7 else base
        cv2.rectangle(frame, (bx, y), (bx + bg_w, y + badge_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (bx, y), (bx + bg_w, y + badge_h), col, 2)
        cv2.putText(frame, line, (bx + 12, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        y += badge_h + spacing


def _draw_wifi_banner(frame, st, n_visible: int) -> None:
    """Wi-Fi wall-radar status: corner panel + full-width alert when motion."""
    h, w = frame.shape[:2]
    panel_w = 300
    panel_h = 96
    x0 = w - panel_w
    cv2.rectangle(frame, (x0, 0), (w, panel_h), (25, 25, 35), -1)
    cv2.rectangle(frame, (x0, 0), (w, panel_h), (90, 90, 110), 1)

    colors = {
        "init":        (160, 160, 160),
        "calibrating": (  0, 200, 255),
        "warming":     (  0, 200, 255),
        "quiet":       ( 60, 220,  60),
        "motion":      (  0,  80, 255),
        "no_link":     (130, 130, 130),
    }
    labels = {
        "init":        "WI-FI INIT",
        "calibrating": "WI-FI CALIBRANDO",
        "warming":     "WI-FI ACUMULANDO",
        "quiet":       "WI-FI: SIN MOVIMIENTO",
        "motion":      "WI-FI: MOVIMIENTO!",
        "no_link":     "WI-FI DESCONECTADO",
    }
    color = colors.get(st.state, (180, 180, 180))
    src_tag = f"[{getattr(st, 'source', '?')}]"
    cv2.putText(frame, f"{src_tag} {labels.get(st.state, 'WI-FI ?')}",
                (x0 + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

    if st.raw is not None:
        unit = "ms" if getattr(st, "source", "rtt") == "rtt" else "%"
        cv2.putText(frame, f"{st.raw:5.1f}{unit}   anom={st.anomaly_rate*100:4.1f}%",
                    (x0 + 8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    med = st.baseline_median if st.baseline_median is not None else 0.0
    mad = st.baseline_mad if st.baseline_mad is not None else 0.0
    cv2.putText(frame,
                f"med={med:4.1f}+/-{mad:4.1f}ms  loss={st.loss_rate*100:3.0f}%",
                (x0 + 8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 200), 1)

    if st.state != "motion":
        return

    # MOTION: full-width prominent alert. Pulsates red border so it's impossible to miss.
    pulse = int(time.time() * 4) % 2  # 4 Hz blink
    border = (0, 0, 255) if pulse else (0, 80, 255)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border, 8)

    # Strip across the top with the alert text.
    strip_h = 56
    cv2.rectangle(frame, (0, 0), (x0 - 4, strip_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (x0 - 4, strip_h), border, 2)
    if n_visible == 0:
        msg = "ALGUIEN TRAS EL MURO"
        sub = "Wi-Fi detecta movimiento sin nadie visible"
    else:
        msg = "MOVIMIENTO TRAS EL MURO"
        sub = f"+ {n_visible} visible(s) en camara"
    cv2.putText(frame, msg, (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, border, 3, cv2.LINE_AA)
    cv2.putText(frame, sub, (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1, cv2.LINE_AA)


def _draw_multi_ap_panel(frame, states, top: int = 6, focused_idx=None,
                         highlight_bssids=None):
    """Bottom-left panel listing top-N visible APs with per-AP motion state.

    If `highlight_bssids` is a non-empty set, rows whose BSSID is in that set
    are drawn at full brightness; others are dimmed to indicate they're not
    part of the active camera direction.
    """
    if not states:
        return None
    h, w = frame.shape[:2]
    ordered = sorted(states.values(), key=lambda s: -s.signal_pct)[:top]
    if not ordered:
        return None

    panel_w = 360
    panel_h = 30 + len(ordered) * 22 + 12
    x0 = 0
    y0 = h - panel_h
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, h), (25, 25, 35), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, h), (90, 90, 110), 1)

    focus_hint = ""
    if focused_idx is not None and 0 <= focused_idx < len(ordered):
        focus_hint = f"  FOCUS: {focused_idx + 1}"
    cv2.putText(frame, "Multi-AP (1-6 focus, 0=off)" + focus_hint,
                (x0 + 8, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 230), 1)

    state_colors = {
        "motion":      (  0,  80, 255),
        "quiet":       ( 60, 220,  60),
        "warming":     (  0, 200, 255),
        "calibrating": (  0, 200, 255),
        "init":        (160, 160, 160),
    }
    use_highlight = bool(highlight_bssids)
    for i, st in enumerate(ordered):
        y = y0 + 36 + i * 22
        color = state_colors.get(st.state, (160, 160, 160))
        in_dir = use_highlight and st.bssid in highlight_bssids
        if use_highlight and not in_dir:
            color = tuple(int(c * 0.35) for c in color)
        label = getattr(st, "label", "") or ""
        if label:
            name_disp = label[:22]
        else:
            name_disp = (st.ssid[:20] + "..") if len(st.ssid) > 22 else st.ssid
        prefix = f"{i+1}."
        if focused_idx == i:
            cv2.rectangle(frame, (x0 + 2, y - 14), (x0 + panel_w - 2, y + 6),
                          (60, 60, 80), -1)
        if in_dir:
            cv2.rectangle(frame, (x0 + 2, y - 14), (x0 + panel_w - 2, y + 6),
                          (40, 80, 100), -1)
        line = f"{prefix:<3} {st.signal_pct:3d}%  {name_disp:<22} anom={st.anomaly_rate*100:3.0f}%"
        cv2.putText(frame, line, (x0 + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.circle(frame, (x0 + panel_w - 14, y - 4), 5, color, -1)

    return ordered


def _draw_ble_panel(frame, states, top: int = 6, highlight_macs=None):
    """Top-right (below the wifi banner) panel listing top BLE devices.

    Distinguishes 'NEAR' (approaching) from 'MOVE' (moving) from 'OK' (quiet).
    Labeled devices (data/ble_devices.json) get highlighted with **name**.
    """
    if not states:
        return None
    h, w = frame.shape[:2]
    ordered = sorted(states.values(), key=lambda s: -s.rssi)[:top]
    if not ordered:
        return None

    panel_w = 340
    panel_h = 30 + len(ordered) * 22 + 8
    x0 = w - panel_w
    y0 = 110  # below the existing wifi banner
    cv2.rectangle(frame, (x0, y0), (w, y0 + panel_h), (25, 25, 40), -1)
    cv2.rectangle(frame, (x0, y0), (w, y0 + panel_h), (90, 90, 130), 1)
    cv2.putText(frame, "BLE devices (proximidad/movimiento)",
                (x0 + 8, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 240), 1)

    state_colors = {
        "approaching": (  0,   0, 255),
        "moving":      (  0,  80, 255),
        "quiet":       ( 60, 220,  60),
        "warming":     (  0, 200, 255),
        "calibrating": (  0, 200, 255),
        "init":        (160, 160, 160),
    }
    tag_short = {
        "approaching": "NEAR",
        "moving":      "MOVE",
        "quiet":       "OK  ",
        "warming":     "WARM",
        "calibrating": "CAL ",
        "init":        "INIT",
    }
    use_highlight = bool(highlight_macs)
    for i, s in enumerate(ordered):
        y = y0 + 36 + i * 22
        color = state_colors.get(s.state, (160, 160, 160))
        in_dir = use_highlight and s.mac in highlight_macs
        if use_highlight and not in_dir:
            color = tuple(int(c * 0.35) for c in color)
        tag = tag_short.get(s.state, "????")
        if s.label:
            name_disp = f"** {s.label[:18]} **"
        elif s.name:
            name_disp = s.name[:22]
        else:
            name_disp = s.mac[-8:]
        delta_str = f"{s.delta:+4.0f}dB" if s.median is not None else "  --  "
        if in_dir:
            cv2.rectangle(frame, (x0 + 2, y - 14), (x0 + panel_w - 2, y + 6),
                          (40, 80, 100), -1)
        line = f"[{tag}] {s.rssi:4d}dB {delta_str}  {name_disp}"
        cv2.putText(frame, line, (x0 + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.circle(frame, (x0 + panel_w - 14, y - 4), 5, color, -1)

    return ordered


def _capture_active_anchors(multi_ap_states, ble_states, focused_idx, multi_ap_ordered):
    """Snapshot the anchors that are currently 'active' (in motion / NEAR / labeled BLE moving).

    AP selection:
      - If a focused AP exists, use that one (intentional).
      - Else: any AP currently in 'motion'.
    BLE selection:
      - Any device in 'approaching' or 'moving' state AND with a user label
        (we don't tag randomized anonymous MACs).
    """
    ap_bssids = []
    if focused_idx is not None and multi_ap_ordered and 0 <= focused_idx < len(multi_ap_ordered):
        ap_bssids.append(multi_ap_ordered[focused_idx].bssid)
    elif multi_ap_states:
        for st in multi_ap_states.values():
            if getattr(st, "state", None) == "motion":
                ap_bssids.append(st.bssid)

    ble_macs = []
    if ble_states:
        for st in ble_states.values():
            if getattr(st, "state", None) in ("approaching", "moving") and getattr(st, "label", ""):
                ble_macs.append(st.mac)
    return ap_bssids, ble_macs


def _interactive_tag(captured_aps, captured_bles, ap_states, ble_states):
    """Print a summary of what was captured and ask for a name. Returns the new name or None."""
    from backend.camera_directions import load_directions, save_directions

    print("\n=== TAG DIRECCION ===")
    if captured_aps:
        print("  APs activos:")
        for b in captured_aps:
            st = ap_states.get(b)
            ssid = getattr(st, "ssid", "") if st else ""
            label = getattr(st, "label", "") if st else ""
            print(f"    AP  {b}  {label or ssid}")
    else:
        print("  APs activos: (ninguno)")
    if captured_bles:
        print("  BLE etiquetados activos:")
        for m in captured_bles:
            st = ble_states.get(m)
            label = getattr(st, "label", "") if st else ""
            print(f"    BLE {m}  ({label})")
    else:
        print("  BLE etiquetados activos: (ninguno)")

    if not captured_aps and not captured_bles:
        print("[tag] No hay anchors activos para vincular. Generar motion primero ")
        print("      (que alguien cruce, o presiona 1-6 para enfocar un AP).\n")
        return None

    try:
        name = input("Nombre para esta direccion (Enter = cancelar): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("[tag] Cancelado.")
        return None
    if not name:
        print("[tag] Cancelado.")
        return None

    dirs = load_directions()
    dirs[name] = {"ap_bssids": captured_aps, "ble_macs": captured_bles}
    save_directions(dirs)
    print(f"[tag] '{name}' guardada. Tecla 'n' para ciclar direcciones.\n")
    return name


def _draw_recording_banner(frame, remaining_s, n_aps, n_bles):
    """Big centered banner shown while a tag recording is in progress."""
    h, w = frame.shape[:2]
    pulse = int(time.time() * 3) % 2
    border = (0, 200, 255) if pulse else (0, 140, 220)
    banner_w = min(640, w - 40)
    x0 = max(20, (w - banner_w) // 2)
    y0 = 180
    cv2.rectangle(frame, (x0, y0), (x0 + banner_w, y0 + 84), (0, 0, 0), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + banner_w, y0 + 84), border, 3)
    cv2.putText(frame, f"GRABANDO TAG: {remaining_s:5.1f}s",
                (x0 + 20, y0 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, border, 3, cv2.LINE_AA)
    cv2.putText(frame,
                f"Capturados hasta ahora: {n_aps} APs + {n_bles} BLE",
                (x0 + 20, y0 + 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 250), 1, cv2.LINE_AA)
    cv2.putText(frame,
                "Camina, vuelve y presiona  t  de nuevo para finalizar (o c=cancelar)",
                (x0 + 20, y0 + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 240), 1, cv2.LINE_AA)


def _draw_direction_banner(frame, current_dir, dir_info, ap_states, ble_states):
    """Top-center banner showing the active direction + ALERT if its anchors fire."""
    h, w = frame.shape[:2]

    alert = False
    if dir_info:
        for bssid in dir_info.get("ap_bssids", []):
            st = ap_states.get(bssid)
            if st and getattr(st, "state", None) == "motion":
                alert = True
                break
        if not alert:
            for mac in dir_info.get("ble_macs", []):
                st = ble_states.get(mac)
                if st and getattr(st, "state", None) in ("approaching", "moving"):
                    alert = True
                    break

    banner_w = 500
    x0 = max(0, (w - banner_w) // 2)
    y0 = 100

    if current_dir is None:
        cv2.rectangle(frame, (x0, y0), (x0 + banner_w, y0 + 36), (40, 40, 50), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + banner_w, y0 + 36), (110, 110, 130), 1)
        cv2.putText(frame, "Sin direccion activa. t=tag,  n=ciclar,  c=limpiar",
                    (x0 + 12, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 220), 1)
        return alert

    banner_h = 64
    if alert:
        pulse = int(time.time() * 4) % 2
        bg = (0, 0, 255) if pulse else (0, 80, 255)
        msg = f"!! MOVIMIENTO: {current_dir} !!"
    else:
        bg = (40, 100, 40)
        msg = f"Direccion: {current_dir}  (OK)"

    cv2.rectangle(frame, (x0, y0), (x0 + banner_w, y0 + banner_h), bg, -1)
    cv2.rectangle(frame, (x0, y0), (x0 + banner_w, y0 + banner_h), (255, 255, 255), 1)
    cv2.putText(frame, msg, (x0 + 14, y0 + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    ap_count = len(dir_info.get("ap_bssids", []))
    ble_count = len(dir_info.get("ble_macs", []))
    cv2.putText(frame, f"{ap_count} APs + {ble_count} BLE   (t=tag, n=cicla, c=off)",
                (x0 + 14, y0 + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 240), 1, cv2.LINE_AA)
    return alert


def cmd_radar(args) -> None:
    from backend.face_identifier import FaceIdentifier
    from backend.floor_mapper import FloorMapper

    model = _load_model(args.model)
    tracker = PersonTracker(min_hits=args.min_hits, max_lost=args.max_lost)

    face_id: Optional[FaceIdentifier] = None
    if not args.no_face:
        face_id = FaceIdentifier()
        loaded = face_id.load()
        if loaded == 0:
            print("[radar] Sin embeddings registrados. Corre primero:")
            print("        python -m backend.camera_detector enroll")
            print("        (Se mostraran etiquetas '#id' generericas mientras tanto.)\n")

    mapper = FloorMapper.load()
    if mapper.homography is None:
        print(f"[radar] Sin calibracion de piso. Usando mapeo lineal "
              f"({mapper.width_m:.1f}m x {mapper.length_m:.1f}m).")
        print("        Para mejor precision corre:  python -m backend.camera_detector calibrate\n")

    wifi_thread = None
    if args.wifi_radar:
        from backend.wifi_radar import WallRadarThread, make_reader
        try:
            reader, resolved, info = make_reader(args.wifi_source)
        except RuntimeError as exc:
            print(f"[radar] Wi-Fi wall-radar deshabilitado: {exc}")
        else:
            wifi_thread = WallRadarThread(
                reader=reader, source=resolved,
                window=30, baseline_secs=10.0,
                k_sigma=args.wifi_k, anomaly_threshold=args.wifi_threshold,
                interval=0.15,
            )
            wifi_thread.start()
            print(f"[radar] Wi-Fi wall-radar ON  source={resolved}  ({info})")
            print(f"        k={args.wifi_k}sigma  thr={args.wifi_threshold*100:.0f}%  "
                  f"calibrando 10s con sala vacia.\n")

    multi_ap_thread = None
    if args.multi_ap:
        from backend.multi_ap_radar import MultiAPRadarThread
        multi_ap_thread = MultiAPRadarThread(
            scan_interval=args.multi_ap_interval,
            window=10, baseline_samples=6,
            k_sigma=2.0, anomaly_threshold=0.30,
        )
        multi_ap_thread.start()
        print(f"[radar] Multi-AP scanner ON  (scan cada {args.multi_ap_interval:.0f}s)")
        print(f"        Top {args.multi_ap_top} APs en panel inferior izquierdo.")
        print(f"        Teclas 1-{args.multi_ap_top} = focus en ese AP (esa direccion)")
        print(f"        Tecla 0 = quitar focus\n")

    ble_thread = None
    if args.ble_radar:
        from backend.ble_radar import BLERadarThread
        ble_thread = BLERadarThread(
            window=20, baseline_samples=10,
            k_sigma=3.0, anomaly_threshold=0.30,
        )
        ble_thread.start()
        print(f"[radar] BLE radar ON  (passive scan).")
        print(f"        Detecta celulares cercanos por RSSI. Panel derecho.")
        print(f"        NEAR=acercandose, MOVE=movimiento, OK=quieto.\n")

    focused_idx: Optional[int] = None

    # Camera direction tagging (link "where I'm pointing" -> anchors)
    from backend.camera_directions import load_directions
    directions = load_directions()
    direction_names: List[str] = sorted(directions.keys())
    current_direction: Optional[str] = None
    direction_idx: int = -1

    # Recording tag mode: smart accumulation during a walk.
    #   tag_focused_bssid: if set, only this AP is considered (else top-N by motion)
    #   tag_ap_motion_count: per-AP count of frames seen in 'motion' state
    #   tag_ble_max_delta:   per-labeled-BLE max abs(RSSI - baseline) seen
    tag_recording: bool = False
    tag_record_until: float = 0.0
    tag_focused_bssid: Optional[str] = None
    tag_ap_motion_count: Dict[str, int] = {}
    tag_ble_max_delta: Dict[str, float] = {}
    tag_ble_max_stale: Dict[str, float] = {}     # longest gap since last advertisement
    tag_ble_was_fresh: set = set()                # MACs that were active at some point
    tag_ble_labels: Dict[str, str] = {}
    tag_ap_labels: Dict[str, str] = {}
    if direction_names:
        print(f"[radar] {len(direction_names)} direccion(es) cargada(s): "
              f"{', '.join(direction_names)}")
        print(f"        Tecla 'n' para ciclar, 'c' para desactivar, 't' para tagear nueva.\n")
    else:
        print(f"[radar] Sin direcciones guardadas. Apunta camara a pared, generá motion, "
              f"presioná 't' para crear la primera.\n")

    # Optional dashboard broadcast — publica "who is visible" cada N segundos
    bc_publisher = None
    last_presence_broadcast = 0.0
    if args.broadcast:
        from backend.event_broadcast import init_broadcaster, publish as _pub
        init_broadcaster(args.broadcast)
        bc_publisher = _pub
        print(f"[radar] presence broadcast -> {args.broadcast}/api/radar/event "
              f"cada {args.broadcast_every:.1f}s")

    # Optional dashboard listen — la camara se entera de los radars en vivo
    listener = None
    anchors_meta: Dict[str, str] = {}
    listen_url = args.listen or args.broadcast  # auto-listen al mismo server si broadcasteamos
    if listen_url:
        from backend.dashboard_listener import DashboardListener
        listener = DashboardListener(listen_url)
        listener.start()
        # Pre-cargar nombres amigables desde anchors.json (solo lectura)
        try:
            import json as _json
            from pathlib import Path as _P
            ap = _P(__file__).resolve().parent.parent / "data" / "anchors.json"
            if ap.exists():
                cfg = _json.loads(ap.read_text(encoding="utf-8"))
                for a in cfg.get("anchors", []):
                    anchors_meta[a["id"]] = a.get("name", a["id"])
        except Exception:
            pass

    cap = _open_camera(args.camera, width=args.cam_width, height=args.cam_height)
    if cap is None:
        if wifi_thread is not None:
            wifi_thread.stop()
        return
    print(f"[radar] 'q' para salir. min_hits={args.min_hits} conf>={args.conf} "
          f"imgsz={args.imgsz} cam={args.cam_width}x{args.cam_height}")

    fps_t0 = time.time()
    fps_frames = 0
    fps_disp = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fh, fw = frame.shape[:2]

            raw = _detect(model, frame, args.conf, imgsz=args.imgsz)
            tracker.update(raw)
            confirmed = tracker.confirmed()

            # Identify each confirmed track (sticky best-score)
            if face_id is not None and face_id.known:
                for t in confirmed:
                    _identify_track(face_id, frame, t, fw, fh,
                                    retry_every=args.face_retry_every)

            # Draw boxes + labels on the camera view, collect points for the map
            map_positions: Dict[str, Dict] = {}
            for t in confirmed:
                x1, y1, x2, y2 = t["bbox"]
                label = t.get("label") or f"#{t['id']}"
                color = LABEL_COLORS.get(t.get("label"), DEFAULT_COLOR)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                tag = label
                if t.get("label_score", 0) > 0:
                    tag = f"{label} {t['label_score']:.2f}"
                cv2.putText(frame, tag, (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                cv2.circle(frame, ((x1 + x2) // 2, y2), 4, color, -1)  # foot point

                # World coordinates from the foot (bottom-center) of the box
                fx_px = (x1 + x2) / 2.0
                fy_px = float(y2)
                wx, wy = mapper.pixel_to_world(fx_px, fy_px, fw, fh)
                map_positions[label] = {"xy": (wx, wy), "color": color}

            # Header
            n_conf = len(confirmed)
            header_color = (0, 255, 255) if n_conf else (120, 120, 120)
            cv2.putText(frame, f"Personas: {n_conf}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, header_color, 2)

            # Broadcast presence snapshot to the dashboard
            if bc_publisher is not None:
                now = time.time()
                if now - last_presence_broadcast >= args.broadcast_every:
                    people = []
                    for t in confirmed:
                        label = t.get("label") or f"#{t['id']}"
                        x1, y1, x2, y2 = t["bbox"]
                        fx_px = (x1 + x2) / 2.0
                        fy_px = float(y2)
                        wx, wy = mapper.pixel_to_world(fx_px, fy_px, fw, fh)
                        people.append({
                            "label": label,
                            "track_id": int(t.get("id", -1)),
                            "x_m": round(wx, 2),
                            "y_m": round(wy, 2),
                            "score": round(t.get("label_score", 0.0), 2),
                            "identified": bool(t.get("label")),
                        })
                    bc_publisher(
                        anchor_id=None,
                        source="camera",
                        kind="presence_snapshot",
                        state="quiet" if n_conf == 0 else "active",
                        value=n_conf,
                        extra={
                            "people": people,
                            "count": n_conf,
                            "fps": round(fps_disp, 1),
                        },
                    )
                    last_presence_broadcast = now

            # FPS
            fps_frames += 1
            if time.time() - fps_t0 >= 1.0:
                fps_disp = fps_frames / (time.time() - fps_t0)
                fps_t0 = time.time()
                fps_frames = 0
            cv2.putText(frame, f"{fps_disp:.1f} fps  {args.model}",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if wifi_thread is not None:
                _draw_wifi_banner(frame, wifi_thread.get_state(), n_visible=n_conf)

            # Live anchor panel + through-wall alert (suscripcion WebSocket)
            if listener is not None:
                _draw_anchor_listen_panel(
                    frame,
                    listener.get_states(),
                    n_visible=n_conf,
                    connected=listener.connected,
                    anchors_meta=anchors_meta,
                )

            # Snapshot states once per frame (avoid repeated lock contention)
            ap_states_now = multi_ap_thread.get_states() if multi_ap_thread else {}
            ble_states_now = ble_thread.get_states() if ble_thread else {}

            dir_info = directions.get(current_direction) if current_direction else None
            highlight_bssids = set(dir_info["ap_bssids"]) if dir_info else None
            highlight_macs = set(dir_info["ble_macs"]) if dir_info else None

            multi_ap_ordered = None
            if multi_ap_thread is not None:
                multi_ap_ordered = _draw_multi_ap_panel(
                    frame, ap_states_now,
                    top=args.multi_ap_top, focused_idx=focused_idx,
                    highlight_bssids=highlight_bssids,
                )

            if ble_thread is not None:
                _draw_ble_panel(frame, ble_states_now, top=args.ble_top,
                                highlight_macs=highlight_macs)

            _draw_direction_banner(frame, current_direction, dir_info,
                                   ap_states_now, ble_states_now)

            # Tag-recording: smart accumulation
            pending_tag_prompt = False
            if tag_recording:
                for st in ap_states_now.values():
                    if tag_focused_bssid is not None and st.bssid != tag_focused_bssid:
                        continue
                    tag_ap_labels[st.bssid] = (
                        getattr(st, "label", "") or getattr(st, "ssid", "") or st.bssid
                    )
                    if getattr(st, "state", None) == "motion":
                        tag_ap_motion_count[st.bssid] = tag_ap_motion_count.get(st.bssid, 0) + 1
                now_ts = time.time()
                for st in ble_states_now.values():
                    if not getattr(st, "label", ""):
                        continue
                    tag_ble_labels[st.mac] = st.label
                    if st.median is not None:
                        delta = abs(st.rssi - st.median)
                        if delta > tag_ble_max_delta.get(st.mac, 0.0):
                            tag_ble_max_delta[st.mac] = delta
                    # Silence tracking: only flag stale if the device was *fresh*
                    # at some point during this recording (avoids false positives
                    # for devices that were already far before we started).
                    stale = max(0.0, now_ts - getattr(st, "last_seen", 0.0))
                    if stale < 2.0:
                        tag_ble_was_fresh.add(st.mac)
                    if st.mac in tag_ble_was_fresh:
                        if stale > tag_ble_max_stale.get(st.mac, 0.0):
                            tag_ble_max_stale[st.mac] = stale
                remaining = tag_record_until - time.time()
                # Show progress: count of unique APs with motion + BLE labels seen
                ap_count = len([c for c in tag_ap_motion_count.values() if c >= 1])
                ble_count = len(tag_ble_max_delta)
                if remaining <= 0:
                    tag_recording = False
                    pending_tag_prompt = True
                else:
                    _draw_recording_banner(frame, remaining, ap_count, ble_count)

            cv2.imshow("Radar - Camara", frame)
            cv2.imshow("Radar - Mapa", mapper.render_map(map_positions, size_px=args.map_size))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            # Multi-AP focus selection via keyboard
            if multi_ap_thread is not None:
                if key == ord("0"):
                    focused_idx = None
                elif ord("1") <= key <= ord("9"):
                    idx = key - ord("1")
                    if idx < args.multi_ap_top:
                        focused_idx = idx
            # Direction tagging  (RECORDING mode: press t once to start, t again to finish)
            if key == ord("t"):
                if tag_recording:
                    print("[tag] Finalizando recording...")
                    tag_record_until = time.time() - 1
                else:
                    tag_recording = True
                    tag_record_until = time.time() + args.tag_duration
                    tag_ap_motion_count = {}
                    tag_ble_max_delta = {}
                    tag_ble_max_stale = {}
                    tag_ble_was_fresh = set()
                    tag_ble_labels = {}
                    tag_ap_labels = {}
                    tag_focused_bssid = None
                    if (focused_idx is not None and multi_ap_ordered
                            and 0 <= focused_idx < len(multi_ap_ordered)):
                        tag_focused_bssid = multi_ap_ordered[focused_idx].bssid
                        ap_name = (
                            getattr(multi_ap_ordered[focused_idx], "label", "")
                            or getattr(multi_ap_ordered[focused_idx], "ssid", "")
                            or tag_focused_bssid
                        )
                        print(f"\n[tag] GRABANDO {args.tag_duration:.0f}s  FOCUS: {ap_name}")
                        print(f"      Solo este AP cuenta para Wi-Fi. Smartwatch/BLE labelados se siguen midiendo.")
                    else:
                        print(f"\n[tag] GRABANDO {args.tag_duration:.0f}s  (sin focus AP)")
                        print(f"      Va a capturar los top-{args.tag_max_aps} APs por tiempo en motion.")
                    print(f"      Anda al lugar, volve, presiona 't' otra vez para nombrar.\n")
            elif key == ord("n"):
                if direction_names:
                    direction_idx = (direction_idx + 1) % len(direction_names)
                    current_direction = direction_names[direction_idx]
                    print(f"[dir] Activa: {current_direction}")
                else:
                    print("[dir] No hay direcciones guardadas. Presioná 't' para crear una.")
            elif key == ord("c"):
                if tag_recording:
                    tag_recording = False
                    tag_ap_motion_count = {}
                    tag_ble_max_delta = {}
                    tag_ble_max_stale = {}
                    tag_ble_was_fresh = set()
                    tag_ble_labels = {}
                    tag_ap_labels = {}
                    tag_focused_bssid = None
                    print("[tag] Recording cancelado.")
                elif current_direction is not None:
                    print(f"[dir] Desactivada: {current_direction}")
                    current_direction = None
                    direction_idx = -1

            # If a recording just finished this frame, compute top-N and prompt.
            if pending_tag_prompt:
                if tag_focused_bssid is not None:
                    # Only the focused AP, if it had at least one motion frame
                    if tag_ap_motion_count.get(tag_focused_bssid, 0) >= 1:
                        captured_aps = [tag_focused_bssid]
                    else:
                        captured_aps = []
                else:
                    # Top-N by motion frames, must have >= 2 frames to count
                    ranked = sorted(tag_ap_motion_count.items(), key=lambda x: -x[1])
                    captured_aps = [b for b, c in ranked[:args.tag_max_aps] if c >= 2]
                # Capture BLE if: max delta is significant OR device went silent
                # after being active during the recording (= "device went away").
                captured_bles = []
                all_ble_macs = set(tag_ble_labels.keys()) | set(tag_ble_max_delta.keys()) | set(tag_ble_max_stale.keys())
                for mac in all_ble_macs:
                    max_d = tag_ble_max_delta.get(mac, 0.0)
                    max_s = tag_ble_max_stale.get(mac, 0.0)
                    if max_d >= args.tag_ble_threshold or max_s >= args.tag_ble_silence:
                        captured_bles.append(mac)

                # Print summary in terminal
                print("\n=== TAG: muestreo terminado ===")
                if tag_ap_motion_count:
                    print("  APs vistos en motion (count = #frames):")
                    for bssid, count in sorted(tag_ap_motion_count.items(), key=lambda x: -x[1]):
                        name = tag_ap_labels.get(bssid, bssid)
                        chosen = " *" if bssid in captured_aps else ""
                        print(f"    {count:4d}  {bssid}  {name}{chosen}")
                if tag_ble_labels:
                    print("  BLE labelados (max delta, silencio mas largo):")
                    for mac in sorted(tag_ble_labels,
                                      key=lambda m: -max(tag_ble_max_delta.get(m, 0),
                                                          tag_ble_max_stale.get(m, 0))):
                        name = tag_ble_labels.get(mac, "")
                        d = tag_ble_max_delta.get(mac, 0.0)
                        s = tag_ble_max_stale.get(mac, 0.0)
                        chosen = " *" if mac in captured_bles else ""
                        reason = ""
                        if mac in captured_bles:
                            if d >= args.tag_ble_threshold and s >= args.tag_ble_silence:
                                reason = "  (delta+silence)"
                            elif d >= args.tag_ble_threshold:
                                reason = "  (delta)"
                            else:
                                reason = "  (silence)"
                        print(f"    delta={d:4.1f}dB  silence={s:4.1f}s  {mac}  {name}{chosen}{reason}")
                print("  (* = elegido para la tag)\n")

                # Reset accumulators
                tag_ap_motion_count = {}
                tag_ble_max_delta = {}
                tag_ble_max_stale = {}
                tag_ble_was_fresh = set()
                tag_ble_labels = {}
                tag_ap_labels = {}
                tag_focused_bssid = None

                new_name = _interactive_tag(captured_aps, captured_bles,
                                            ap_states_now, ble_states_now)
                if new_name:
                    directions = load_directions()
                    direction_names = sorted(directions.keys())
                    if new_name in direction_names:
                        direction_idx = direction_names.index(new_name)
                        current_direction = new_name
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if wifi_thread is not None:
            wifi_thread.stop()
        if multi_ap_thread is not None:
            multi_ap_thread.stop()
        if ble_thread is not None:
            ble_thread.stop()
        if listener is not None:
            listener.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--camera", type=int, default=0)
        sp.add_argument("--conf", type=float, default=0.55,
                        help="YOLO confidence threshold (default 0.55, raise to reduce FPs)")
        sp.add_argument("--model", default=DEFAULT_MODEL,
                        help="yolov8n.pt (fast) or yolov8s.pt (more accurate, slower)")
        sp.add_argument("--min-hits", type=int, default=3,
                        help="Frames a track must persist before counting as confirmed")
        sp.add_argument("--max-lost", type=int, default=5,
                        help="Frames a track can vanish before being dropped")

    pl = sub.add_parser("live", help="Live window with confirmed/candidate boxes")
    add_common(pl)
    pl.set_defaults(func=cmd_live)

    pc = sub.add_parser("count", help="Headless confirmed-person count to stdout")
    add_common(pc)
    pc.add_argument("--interval", type=float, default=2.0)
    pc.set_defaults(func=cmd_count)

    pt = sub.add_parser("test", help="10-frame smoke test")
    add_common(pt)
    pt.set_defaults(func=cmd_test)

    pe = sub.add_parser("enroll", help="Enroll faces from data/faces/<Name>/")
    pe.set_defaults(func=cmd_enroll)

    pca = sub.add_parser("calibrate", help="Click 4 floor points to set the homography")
    pca.add_argument("--camera", type=int, default=0)
    pca.set_defaults(func=cmd_calibrate)

    pr = sub.add_parser("radar", help="YOLO + face ID + 2D cartesian map")
    add_common(pr)
    pr.add_argument("--no-face", action="store_true",
                    help="Skip face identification (faster; shows '#id' generic labels)")
    pr.add_argument("--map-size", type=int, default=520, help="Pixel size of the map window")
    pr.add_argument("--wifi-radar", action="store_true",
                    help="Habilita el detector de movimiento Wi-Fi tras la pared")
    pr.add_argument("--wifi-source", choices=["auto", "rtt", "signal"], default="auto",
                    help="Fuente del radar Wi-Fi: rtt (ping, en vivo) o signal (netsh, cacheado).")
    pr.add_argument("--wifi-k", type=float, default=4.0,
                    help="Anomalia = sample > median + k*MAD. Mayor = menos sensible. Default 4.")
    pr.add_argument("--wifi-threshold", type=float, default=0.30,
                    help="Fraccion de la ventana con anomalias para gatillar motion. Default 0.30")
    pr.add_argument("--multi-ap", action="store_true",
                    help="Habilita scan multi-AP. Cada router visible es un rayo direccional. "
                         "Teclas 1-6 = focus en un AP (esa pared). Tecla 0 = quitar focus.")
    pr.add_argument("--multi-ap-interval", type=float, default=4.0,
                    help="Segundos entre scans multi-AP (Windows rate-limit ~3-4s). Default 4.")
    pr.add_argument("--multi-ap-top", type=int, default=6,
                    help="Cuantos APs (los mas fuertes) mostrar en el panel. Default 6.")
    pr.add_argument("--ble-radar", action="store_true",
                    help="Habilita scan BLE pasivo. Detecta celulares cercanos por RSSI. "
                         "Requiere: pip install bleak.")
    pr.add_argument("--ble-top", type=int, default=6,
                    help="Cuantos dispositivos BLE mostrar en el panel. Default 6.")
    pr.add_argument("--tag-duration", type=float, default=60.0,
                    help="Segundos que dura el modo grabar tag (caminar y volver). Default 60.")
    pr.add_argument("--tag-max-aps", type=int, default=3,
                    help="Cuantos APs (top-N por tiempo en motion) capturar en una tag. Default 3.")
    pr.add_argument("--tag-ble-threshold", type=float, default=5.0,
                    help="dB de variacion BLE para que un device labeled cuente en la tag. Default 5.")
    pr.add_argument("--tag-ble-silence", type=float, default=5.0,
                    help="Segundos de silencio BLE (sin advertisements) para capturar como 'se fue'. Default 5.")
    # --- Performance tuning ------------------------------------------------
    pr.add_argument("--cam-width", type=int, default=640,
                    help="Ancho de la cámara (default 640). 1280 si querés HD pero MUCHO más lento.")
    pr.add_argument("--cam-height", type=int, default=480,
                    help="Alto de la cámara (default 480).")
    pr.add_argument("--imgsz", type=int, default=416,
                    help="Resolucion interna del modelo YOLO (default 416). 320 es mas rapido, 640 mas preciso.")
    pr.add_argument("--face-retry-every", type=int, default=8,
                    help="Cada N frames se reintenta el face recognition por track. "
                         "Subi este numero si la cámara va lenta.")
    pr.add_argument("--broadcast", default=None,
                    help="URL del dashboard FastAPI (ej http://127.0.0.1:8000). "
                         "Publica snapshots de quien esta visible + posicion al 3D.")
    pr.add_argument("--broadcast-every", type=float, default=1.0,
                    help="Segundos entre snapshots al dashboard. Default 1.0")
    pr.add_argument("--listen", default=None,
                    help="URL del dashboard FastAPI. Suscribe al WebSocket para "
                         "ver estados de TODOS los radars (CSI/BLE/WiFi) sobre la "
                         "imagen + alerta ALGUIEN TRAS EL MURO. Si pones --broadcast, "
                         "se auto-activa con la misma URL.")
    pr.set_defaults(func=cmd_radar)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
