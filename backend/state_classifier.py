"""Multi-AP occupancy state classifier.

Uses delta-from-baseline features (RSSI vs empty-room baseline) so the
classifier learns "how the body perturbs each AP" rather than memorizing
absolute RSSI patterns that drift with the environment.
"""

import argparse
import json
import pickle
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

try:
    from backend.wifi_scanner import WiFiScanner
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend.wifi_scanner import WiFiScanner


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_FILE = DATA_DIR / "state_samples.jsonl"
MODEL_FILE = DATA_DIR / "state_model.pkl"
MISSING_DBM = -100.0
DEFAULT_BASELINE_LABEL = "vacio"


def _new_session_id() -> str:
    return f"s{int(time.time())}_{uuid.uuid4().hex[:6]}"


class StateClassifier:
    def __init__(self):
        self.bssids: List[str] = []
        self.model: Optional[RandomForestClassifier] = None
        self.baseline: Dict[str, float] = {}  # bssid -> mean RSSI when empty

    def _vectorize(self, rssi: Dict[str, float]) -> np.ndarray:
        # Delta from baseline per BSSID. Negative = signal weaker than empty (body absorbing).
        return np.array([
            rssi.get(b, MISSING_DBM) - self.baseline.get(b, MISSING_DBM)
            for b in self.bssids
        ], dtype=float)

    def fit(
        self,
        samples: List[dict],
        bssids: Optional[List[str]] = None,
        baseline_label: str = DEFAULT_BASELINE_LABEL,
    ) -> dict:
        baseline_samples = [s for s in samples if s["label"] == baseline_label]
        if not baseline_samples:
            raise ValueError(
                f"No samples with label '{baseline_label}' found — needed as empty-room baseline. "
                f"Record some with: record --label {baseline_label}"
            )

        if bssids is None:
            bssids = self._auto_select_bssids(samples)
        self.bssids = list(bssids)

        # Baseline = per-BSSID mean RSSI when the label says room is empty
        self.baseline = {}
        for b in self.bssids:
            vals = [s["rssi"][b] for s in baseline_samples if b in s["rssi"]]
            self.baseline[b] = float(np.mean(vals)) if vals else MISSING_DBM

        X = np.array([self._vectorize(s["rssi"]) for s in samples])
        y = np.array([s["label"] for s in samples])
        groups = np.array([s.get("session_id", "legacy") for s in samples])

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)

        info: dict = {
            "n_samples": len(samples),
            "n_bssids": len(self.bssids),
            "baseline_label": baseline_label,
            "n_baseline_samples": len(baseline_samples),
            "classes": sorted({str(c) for c in y}),
            "samples_per_class": {str(k): int(v) for k, v in Counter(y).items()},
            "samples_per_session": {str(k): int(v) for k, v in Counter(groups).items()},
        }

        unique_groups = set(groups)
        if len(unique_groups) >= 2:
            try:
                scores = cross_val_score(self.model, X, y, cv=LeaveOneGroupOut(), groups=groups)
                info["cv_method"] = "LeaveOneGroupOut (honest)"
                info["cv_accuracy"] = round(float(scores.mean()), 3)
                info["cv_std"] = round(float(scores.std()), 3)
                info["cv_n_splits"] = int(len(scores))
            except Exception as exc:
                info["cv_error"] = str(exc)
        else:
            info["cv_warning"] = "Only 1 session — true accuracy unknown until you record another session"
            min_per_class = min(Counter(y).values())
            if len(info["classes"]) >= 2 and min_per_class >= 2:
                cv = min(5, min_per_class)
                try:
                    scores = cross_val_score(self.model, X, y, cv=cv)
                    info["cv_accuracy_inflated"] = round(float(scores.mean()), 3)
                except Exception as exc:
                    info["cv_error"] = str(exc)

        return info

    def predict(self, rssi: Dict[str, float]) -> dict:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        vec = self._vectorize(rssi).reshape(1, -1)
        label = str(self.model.predict(vec)[0])
        proba = {
            str(c): round(float(p), 3)
            for c, p in zip(self.model.classes_, self.model.predict_proba(vec)[0])
        }
        return {"label": label, "proba": proba}

    @staticmethod
    def _auto_select_bssids(samples: List[dict], min_ratio: float = 0.8) -> List[str]:
        counts: Counter = Counter()
        for s in samples:
            counts.update(s["rssi"].keys())
        threshold = len(samples) * min_ratio
        return sorted(b for b, c in counts.items() if c >= threshold)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "bssids": self.bssids,
                "model": self.model,
                "baseline": self.baseline,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "StateClassifier":
        with open(path, "rb") as f:
            data = pickle.load(f)
        c = cls()
        c.bssids = data["bssids"]
        c.model = data["model"]
        c.baseline = data.get("baseline", {})
        return c


def load_samples(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_sample(path: Path, sample: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def cmd_record(args) -> None:
    scanner = WiFiScanner()
    session_id = _new_session_id()
    print(f"Recording label='{args.label}' for {args.seconds}s (each scan ~5s).")
    print(f"Session: {session_id}")
    print("Move into position now. Ctrl+C to stop early.\n")
    start = time.time()
    n = 0
    try:
        while time.time() - start < args.seconds:
            rssi = scanner.get_rssi_dict(force_refresh=True)
            if rssi:
                append_sample(SAMPLES_FILE, {
                    "ts": time.time(),
                    "session_id": session_id,
                    "label": args.label,
                    "rssi": rssi,
                })
                n += 1
                print(f"  [{n:4d}] heard {len(rssi)} APs")
            else:
                err = scanner.last_error
                print(f"  (no APs — {err['code'] if err else 'empty scan'})")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    print(f"\n{n} samples appended to {SAMPLES_FILE}")


def cmd_stats(args) -> None:
    samples = load_samples(SAMPLES_FILE)
    print(f"Total samples: {len(samples)}\n")
    if not samples:
        return
    by_label = Counter(s["label"] for s in samples)
    print("Samples per label:")
    for label, count in sorted(by_label.items()):
        print(f"  {label:30s} {count}")
    by_session = Counter(s.get("session_id", "legacy") for s in samples)
    print(f"\nSamples per session ({len(by_session)} sessions):")
    for sid, count in by_session.most_common():
        print(f"  {sid:30s} {count}")
    bssid_counts: Counter = Counter()
    for s in samples:
        bssid_counts.update(s["rssi"].keys())
    print(f"\nTop BSSIDs by appearance (out of {len(samples)} samples):")
    for bssid, count in bssid_counts.most_common(10):
        pct = 100 * count / len(samples)
        print(f"  {bssid}  {count:4d}  ({pct:5.1f}%)")


def cmd_train(args) -> None:
    samples = load_samples(SAMPLES_FILE)
    if len(samples) < 5:
        print(f"Need at least 5 samples, have {len(samples)}.")
        return
    bssids = [b.strip().lower() for b in args.bssids.split(",")] if args.bssids else None
    clf = StateClassifier()
    try:
        info = clf.fit(samples, bssids=bssids, baseline_label=args.baseline_label)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return
    print("Training result:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"\nBaseline (top 10 BSSIDs of {len(clf.bssids)}):")
    for b in clf.bssids[:10]:
        print(f"  {b}  baseline={clf.baseline.get(b, 0):+6.1f} dBm")
    if len(clf.bssids) > 10:
        print(f"  ... and {len(clf.bssids) - 10} more")
    clf.save(MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")


def _beep(pattern: str) -> None:
    try:
        import winsound
    except ImportError:
        return
    patterns = {
        "walk":         [(1500, 200), (0, 100), (1500, 200), (0, 100), (1500, 200)],
        "record_start": [(700, 800)],
        "record_end":   [(1200, 250), (0, 150), (1200, 250)],
        "done":         [(900, 200), (1100, 200), (1400, 400)],
    }
    for freq, dur in patterns.get(pattern, []):
        if freq == 0:
            time.sleep(dur / 1000)
        else:
            winsound.Beep(freq, dur)


def cmd_session(args) -> None:
    """Guided multi-label recording session with audio cues + auto-train."""
    scanner = WiFiScanner()
    rec = args.seconds
    session_id = _new_session_id()

    phases = [
        ("cocina", "ANDA A LA COCINA",                              45),
        ("pieza",  "ANDA A LA PIEZA",                               45),
        ("vacio",  "SAL DEL DEPTO (o lo mas lejos posible del PC)", 60),
    ]

    print("=" * 60)
    print("SESION GUIADA DE GRABACION")
    print(f"Session ID: {session_id}")
    print("=" * 60)
    for i, (label, instr, walk) in enumerate(phases, 1):
        print(f"  {i}. {walk}s para caminar -> {rec}s grabando '{label}' ({instr})")
    print("=" * 60)
    print()
    print("CLAVES DE AUDIO:")
    print("  3 beeps agudos -> empieza a caminar al siguiente lugar")
    print("  1 beep grave largo -> empieza grabacion, QUEDATE QUIETO")
    print("  2 beeps medios -> grabacion terminada")
    print("  3 beeps ascendentes -> sesion completa, vuelve al PC")
    print()
    print("Arrancando en 10s. Quedate frente al PC hasta el primer beep.\n")
    time.sleep(10)

    total = 0
    for label, instr, walk in phases:
        print(f"\n>>> {instr} -- tienes {walk}s")
        _beep("walk")
        time.sleep(walk)

        print(f"--- GRABANDO '{label}' por {rec}s -- NO TE MUEVAS ---")
        _beep("record_start")

        start = time.time()
        n = 0
        while time.time() - start < rec:
            rssi = scanner.get_rssi_dict(force_refresh=True)
            if rssi:
                append_sample(SAMPLES_FILE, {
                    "ts": time.time(),
                    "session_id": session_id,
                    "label": label,
                    "rssi": rssi,
                })
                n += 1
        _beep("record_end")
        print(f"    -> {n} muestras de '{label}'")
        total += n

    _beep("done")
    print(f"\n[OK] Sesion completa: {total} muestras nuevas.\n")

    print("=== Entrenando ===")
    samples = load_samples(SAMPLES_FILE)
    if len(samples) < 5:
        print(f"Pocas muestras ({len(samples)}). Algo fallo en la sesion.")
        return
    clf = StateClassifier()
    try:
        info = clf.fit(samples)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return
    for k, v in info.items():
        print(f"  {k}: {v}")
    clf.save(MODEL_FILE)
    print(f"\nModelo guardado en {MODEL_FILE}")
    print("Cuando vuelvas al PC, corre: python -m backend.state_classifier live")


def cmd_live(args) -> None:
    if not MODEL_FILE.exists():
        print(f"No model at {MODEL_FILE}. Run `train` first.")
        return
    clf = StateClassifier.load(MODEL_FILE)
    if not clf.baseline:
        print("[WARN] Loaded model has no baseline — train with the new code to use delta features.")
    scanner = WiFiScanner()

    confirm = max(1, args.confirm)
    recent: List[str] = []
    stable_state: Optional[str] = None
    print(f"Live prediction (hysteresis={confirm}). Ctrl+C to stop.\n")

    try:
        while True:
            rssi = scanner.get_rssi_dict(force_refresh=True)
            if rssi:
                result = clf.predict(rssi)
                raw = result["label"]
                top_p = result["proba"][raw]
                heard = sum(1 for b in clf.bssids if b in rssi)

                recent.append(raw)
                if len(recent) > confirm:
                    recent.pop(0)
                if len(recent) == confirm and len(set(recent)) == 1:
                    stable_state = recent[0]

                mark = "**" if raw == stable_state else "  "
                print(f"  {mark} raw={raw:12s} p={top_p:.2f}  stable={stable_state}  ({heard}/{len(clf.bssids)} APs)")
            else:
                print("  (scan empty)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record", help="Record labeled RSSI samples")
    pr.add_argument("--label", required=True)
    pr.add_argument("--seconds", type=int, default=60)
    pr.add_argument("--interval", type=float, default=1.0)
    pr.set_defaults(func=cmd_record)

    ps = sub.add_parser("stats", help="Show counts per label and AP coverage")
    ps.set_defaults(func=cmd_stats)

    pt = sub.add_parser("train", help="Train classifier from collected samples")
    pt.add_argument("--bssids", help="Comma-separated BSSIDs (default: auto-select stable)")
    pt.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL,
                    help=f"Label whose samples define the empty-room baseline (default: {DEFAULT_BASELINE_LABEL})")
    pt.set_defaults(func=cmd_train)

    pl = sub.add_parser("live", help="Continuous live prediction with hysteresis")
    pl.add_argument("--interval", type=float, default=1.0)
    pl.add_argument("--confirm", type=int, default=3,
                    help="Consecutive identical predictions needed to switch stable state")
    pl.set_defaults(func=cmd_live)

    pn = sub.add_parser("session", help="Guided multi-label recording + auto-train")
    pn.add_argument("--seconds", type=int, default=75)
    pn.set_defaults(func=cmd_session)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
