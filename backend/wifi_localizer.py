"""
Wi-Fi Fingerprint-based Indoor Localization Engine.
Uses KNN regression + Kalman filter for position estimation.
"""

import json
import pickle
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class WiFiLocalizer:
    """Wi-Fi fingerprinting indoor localization system."""

    def __init__(self):
        # Fingerprint database: list of dicts
        # Each: {id, position: [x, y], rssi: {bssid: dbm}, timestamp}
        self.fingerprint_db: List[Dict[str, Any]] = []

        # Ordered list of all known BSSIDs across all fingerprints
        self.ap_list: List[str] = []

        # ML model
        self.model: Optional[KNeighborsRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained: bool = False
        self.accuracy: Optional[float] = None

        # Kalman filter state
        # State: [x, y, vx, vy]
        self._kalman_state: Optional[np.ndarray] = None
        self._kalman_cov: Optional[np.ndarray] = None
        self._kalman_initialized: bool = False
        self._last_predict_time: float = 0.0

        # Kalman parameters
        self._process_noise = 0.1
        self._measurement_noise = 1.0

    # ------------------------------------------------------------------
    # Fingerprint management
    # ------------------------------------------------------------------

    def add_fingerprint(self, position: List[float], rssi_dict: Dict[str, float]) -> str:
        """Add a fingerprint measurement at a given position."""
        fp_id = str(uuid.uuid4())[:8]
        fp = {
            "id": fp_id,
            "position": list(position),
            "rssi": dict(rssi_dict),
            "timestamp": time.time(),
        }
        self.fingerprint_db.append(fp)
        self._update_ap_list()
        # Model is stale after new data
        self.is_trained = False
        logger.info("Added fingerprint %s at position %s with %d APs", fp_id, position, len(rssi_dict))
        return fp_id

    def remove_fingerprint(self, fp_id: str) -> bool:
        """Remove a fingerprint by its id."""
        before = len(self.fingerprint_db)
        self.fingerprint_db = [fp for fp in self.fingerprint_db if fp["id"] != fp_id]
        removed = len(self.fingerprint_db) < before
        if removed:
            self._update_ap_list()
            self.is_trained = False
            logger.info("Removed fingerprint %s", fp_id)
        return removed

    def _update_ap_list(self):
        """Rebuild the ordered AP list from all fingerprints."""
        ap_set = set()
        for fp in self.fingerprint_db:
            ap_set.update(fp["rssi"].keys())
        self.ap_list = sorted(ap_set)

    def _fingerprint_to_vector(self, rssi_dict: Dict[str, float]) -> np.ndarray:
        """Convert an RSSI dict to a fixed-length feature vector.
        Missing APs get -100 dBm (very weak / not heard).
        """
        vec = np.full(len(self.ap_list), -100.0)
        for i, bssid in enumerate(self.ap_list):
            if bssid in rssi_dict:
                vec[i] = rssi_dict[bssid]
        return vec

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def train_model(self) -> Dict[str, Any]:
        """Train the KNN regression model on current fingerprints."""
        n = len(self.fingerprint_db)
        if n < 2:
            msg = f"Need at least 2 fingerprints to train, have {n}."
            logger.warning(msg)
            return {"success": False, "message": msg, "n_fingerprints": n}

        self._update_ap_list()

        if len(self.ap_list) == 0:
            msg = "No AP data in fingerprints."
            return {"success": False, "message": msg}

        # Build feature matrix and target matrix
        X = np.array([self._fingerprint_to_vector(fp["rssi"]) for fp in self.fingerprint_db])
        Y = np.array([fp["position"] for fp in self.fingerprint_db])

        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Choose k
        k = min(3, n - 1)
        k = max(1, k)

        self.model = KNeighborsRegressor(
            n_neighbors=k,
            weights="distance",
            metric="euclidean",
        )
        self.model.fit(X_scaled, Y)
        self.is_trained = True

        # Cross-validation (if enough samples)
        accuracy_info: Dict[str, Any] = {
            "success": True,
            "n_fingerprints": n,
            "n_aps": len(self.ap_list),
            "k_neighbors": k,
        }

        if n >= 4:
            cv_folds = min(5, n)
            try:
                scores = cross_val_score(
                    KNeighborsRegressor(n_neighbors=k, weights="distance"),
                    X_scaled, Y, cv=cv_folds, scoring="r2",
                )
                self.accuracy = float(np.mean(scores))
                accuracy_info["cv_r2_mean"] = round(self.accuracy, 4)
                accuracy_info["cv_r2_std"] = round(float(np.std(scores)), 4)
            except Exception as exc:
                logger.warning("Cross-validation failed: %s", exc)
                self.accuracy = None
                accuracy_info["cv_r2_mean"] = None
        else:
            self.accuracy = None
            accuracy_info["cv_r2_mean"] = None
            accuracy_info["message"] = "Not enough samples for cross-validation."

        # Reset Kalman on retrain
        self._kalman_initialized = False
        self._kalman_state = None
        self._kalman_cov = None

        logger.info("Model trained: %s", accuracy_info)
        return accuracy_info

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_position(self, rssi_dict: Dict[str, float]) -> Dict[str, float]:
        """Predict position from an RSSI measurement."""
        if not self.is_trained or self.model is None or self.scaler is None:
            logger.warning("Model not trained – returning origin.")
            return {"x": 0.0, "y": 0.0, "confidence": 0.0}

        vec = self._fingerprint_to_vector(rssi_dict).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)
        raw_pred = self.model.predict(vec_scaled)[0]  # [x, y]

        # Apply Kalman filter
        filtered = self._kalman_filter(raw_pred)

        confidence = self.get_confidence()

        return {
            "x": round(float(filtered[0]), 3),
            "y": round(float(filtered[1]), 3),
            "confidence": round(confidence, 3),
        }

    # ------------------------------------------------------------------
    # Kalman Filter (2D with velocity model)
    # ------------------------------------------------------------------

    def _kalman_filter(self, measurement: np.ndarray) -> np.ndarray:
        """2D Kalman filter with constant-velocity model."""
        now = time.time()

        if not self._kalman_initialized:
            # Initialize state to first measurement
            self._kalman_state = np.array([measurement[0], measurement[1], 0.0, 0.0])
            self._kalman_cov = np.eye(4) * 1.0
            self._kalman_initialized = True
            self._last_predict_time = now
            return measurement

        # Time delta
        dt = now - self._last_predict_time
        dt = max(dt, 0.01)  # Avoid zero
        dt = min(dt, 5.0)  # Cap large gaps
        self._last_predict_time = now

        # State transition matrix F
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        # Process noise Q
        q = self._process_noise
        Q = np.array([
            [q * dt**3 / 3, 0, q * dt**2 / 2, 0],
            [0, q * dt**3 / 3, 0, q * dt**2 / 2],
            [q * dt**2 / 2, 0, q * dt, 0],
            [0, q * dt**2 / 2, 0, q * dt],
        ])

        # Measurement matrix H (we observe x, y only)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # Measurement noise R
        R = np.eye(2) * self._measurement_noise

        # Predict
        x_pred = F @ self._kalman_state
        P_pred = F @ self._kalman_cov @ F.T + Q

        # Update
        z = np.array([measurement[0], measurement[1]])
        y_innov = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        self._kalman_state = x_pred + K @ y_innov
        self._kalman_cov = (np.eye(4) - K @ H) @ P_pred

        return self._kalman_state[:2]

    def get_confidence(self) -> float:
        """Compute confidence from Kalman covariance (0-1 scale)."""
        if self._kalman_cov is None:
            return 0.0
        # Trace of position covariance (first 2 diagonal elements)
        pos_trace = self._kalman_cov[0, 0] + self._kalman_cov[1, 1]
        # Map to 0-1: low trace = high confidence
        # trace ~ 0 → confidence ~ 1; trace ~ 10 → confidence ~ 0
        confidence = max(0.0, min(1.0, 1.0 - pos_trace / 10.0))
        return confidence

    def reset_kalman(self):
        """Reset the Kalman filter state."""
        self._kalman_initialized = False
        self._kalman_state = None
        self._kalman_cov = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_fingerprints(self, path: str):
        """Save fingerprint database to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "fingerprints": self.fingerprint_db,
            "ap_list": self.ap_list,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d fingerprints to %s", len(self.fingerprint_db), path)

    def load_fingerprints(self, path: str) -> bool:
        """Load fingerprint database from JSON file."""
        p = Path(path)
        if not p.exists():
            logger.warning("Fingerprint file not found: %s", path)
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.fingerprint_db = data.get("fingerprints", [])
            self._update_ap_list()
            self.is_trained = False
            logger.info("Loaded %d fingerprints from %s", len(self.fingerprint_db), path)
            return True
        except Exception as exc:
            logger.error("Error loading fingerprints: %s", exc)
            return False

    def save_model(self, path: str):
        """Save trained model + scaler to pickle."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "ap_list": self.ap_list,
            "is_trained": self.is_trained,
            "accuracy": self.accuracy,
        }
        with open(p, "wb") as f:
            pickle.dump(data, f)
        logger.info("Saved model to %s", path)

    def load_model(self, path: str) -> bool:
        """Load trained model + scaler from pickle."""
        p = Path(path)
        if not p.exists():
            logger.warning("Model file not found: %s", path)
            return False
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.ap_list = data["ap_list"]
            self.is_trained = data.get("is_trained", True)
            self.accuracy = data.get("accuracy")
            logger.info("Loaded model from %s", path)
            return True
        except Exception as exc:
            logger.error("Error loading model: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics."""
        return {
            "n_fingerprints": len(self.fingerprint_db),
            "n_aps": len(self.ap_list),
            "is_trained": self.is_trained,
            "accuracy": self.accuracy,
            "kalman_initialized": self._kalman_initialized,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    loc = WiFiLocalizer()

    # Demo: add some fingerprints
    loc.add_fingerprint([1.0, 1.0], {"aa:bb:cc:dd:ee:01": -45, "aa:bb:cc:dd:ee:02": -60})
    loc.add_fingerprint([5.0, 1.0], {"aa:bb:cc:dd:ee:01": -70, "aa:bb:cc:dd:ee:02": -40})
    loc.add_fingerprint([3.0, 5.0], {"aa:bb:cc:dd:ee:01": -55, "aa:bb:cc:dd:ee:02": -55})

    result = loc.train_model()
    print("Train:", result)

    pred = loc.predict_position({"aa:bb:cc:dd:ee:01": -50, "aa:bb:cc:dd:ee:02": -50})
    print("Prediction:", pred)

    print("Stats:", loc.get_stats())
