"""Face detection + recognition using OpenCV YuNet + SFace (ONNX).

Why YuNet/SFace instead of face_recognition (dlib):
  - Already bundled in opencv-python (>=4.5.4). Zero extra deps on Windows.
  - Faster (SFace ~10ms/face on CPU) and more accurate than HOG+dlib.
  - Models are downloaded once on first use from the official OpenCV Zoo.

Enrollment:
  - Put 3-5 clear photos per identity in:  data/faces/<Name>/
  - Run:  python -m backend.camera_detector enroll
  - Embeddings cached at:  data/faces/embeddings.npz

Recognition is fused with YOLO person tracking in camera_detector.cmd_radar.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = _REPO / "data" / "models"
FACES_DIR = _REPO / "data" / "faces"
EMBEDDINGS_FILE = FACES_DIR / "embeddings.npz"

YUNET_FILE = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_FILE = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

# OpenCV-recommended cosine match threshold for SFace
COSINE_THRESHOLD = 0.363


def _ensure_model(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[face] Descargando {path.name} ...")
    urllib.request.urlretrieve(url, str(path))
    print(f"[face] Guardado -> {path}")


class FaceIdentifier:
    def __init__(
        self,
        detector_path: Path = YUNET_FILE,
        recognizer_path: Path = SFACE_FILE,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        _ensure_model(detector_path, YUNET_URL)
        _ensure_model(recognizer_path, SFACE_URL)
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320),
            score_threshold, nms_threshold, top_k,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        self.known: Dict[str, List[np.ndarray]] = {}

    # ------------- enrollment -------------

    def enroll_from_folders(self, faces_root: Path = FACES_DIR) -> Dict[str, int]:
        faces_root = Path(faces_root)
        counts: Dict[str, int] = {}
        if not faces_root.exists():
            faces_root.mkdir(parents=True, exist_ok=True)
            print(f"[face] Crea subcarpetas con fotos en: {faces_root}")
            print("       Ej: data/faces/Tata/, data/faces/Abuela/, data/faces/Yo/")
            return counts

        person_dirs = [p for p in faces_root.iterdir() if p.is_dir()]
        if not person_dirs:
            print(f"[face] Sin subcarpetas en {faces_root}. Agrega data/faces/<Nombre>/foto.jpg")
            return counts

        for person_dir in sorted(person_dirs):
            name = person_dir.name
            embs: List[np.ndarray] = []
            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  [skip] no se puede leer {img_path.name}")
                    continue
                face = self._best_face(img)
                if face is None:
                    print(f"  [skip] sin rostro detectable: {img_path.name}")
                    continue
                aligned = self.recognizer.alignCrop(img, face)
                feat = self.recognizer.feature(aligned)
                embs.append(feat.copy())
                print(f"  [ok]  {name}/{img_path.name}")
            if embs:
                self.known[name] = embs
                counts[name] = len(embs)
            else:
                print(f"  [warn] {name}: 0 rostros utilizables.")
        return counts

    def save(self, path: Path = EMBEDDINGS_FILE) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        flat = {}
        for name, embs in self.known.items():
            for i, e in enumerate(embs):
                flat[f"{name}||{i}"] = e
        np.savez(str(path), **flat)
        print(f"[face] Embeddings guardados -> {path}")

    def load(self, path: Path = EMBEDDINGS_FILE) -> int:
        path = Path(path)
        if not path.exists():
            return 0
        data = np.load(str(path))
        self.known = {}
        for key in data.files:
            name, _ = key.split("||", 1)
            self.known.setdefault(name, []).append(data[key])
        total = sum(len(v) for v in self.known.values())
        print(f"[face] Cargados {total} embeddings de {len(self.known)} identidades.")
        return total

    # ------------- detection / identification -------------

    def _detect_in(self, img: np.ndarray):
        h, w = img.shape[:2]
        if h < 30 or w < 30:
            return None
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        return faces  # Nx15 or None

    def _best_face(self, img: np.ndarray):
        faces = self._detect_in(img)
        if faces is None or len(faces) == 0:
            return None
        idx = int(np.argmax(faces[:, -1]))  # highest score
        return faces[idx]

    def identify_in_crop(
        self, crop: np.ndarray, threshold: float = COSINE_THRESHOLD
    ) -> Tuple[Optional[str], float, Optional[Tuple[int, int, int, int]]]:
        """Return (name | None, score, face_bbox_within_crop | None)."""
        if not self.known or crop is None or crop.size == 0:
            return None, 0.0, None
        face = self._best_face(crop)
        if face is None:
            return None, 0.0, None
        aligned = self.recognizer.alignCrop(crop, face)
        feat = self.recognizer.feature(aligned)

        best_name: Optional[str] = None
        best_score = -1.0
        for name, embs in self.known.items():
            for e in embs:
                s = float(self.recognizer.match(feat, e, cv2.FaceRecognizerSF_FR_COSINE))
                if s > best_score:
                    best_score = s
                    best_name = name

        bbox = _face_bbox(face)
        if best_score < threshold:
            return None, best_score, bbox
        return best_name, best_score, bbox

    @classmethod
    def load_default(cls) -> "FaceIdentifier":
        fid = cls()
        fid.load()
        return fid


def _face_bbox(face_row: np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = face_row[:4]
    return int(x), int(y), int(x + w), int(y + h)
