"""
3D House Scanner using OpenCV Structure from Motion.
Provides camera calibration, feature extraction, and basic SfM reconstruction.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.warning("OpenCV not available – 3D scanning features disabled.")


class HouseScanner3D:
    """Structure from Motion based 3D house scanner."""

    def __init__(self):
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.is_calibrated: bool = False

        # Recorded frames for reconstruction
        self.frames: List[np.ndarray] = []
        self.keypoints_list: List[Any] = []
        self.descriptors_list: List[np.ndarray] = []

        # Reconstruction results
        self.point_cloud: Optional[np.ndarray] = None
        self.camera_poses: List[np.ndarray] = []
        self.generated_floor_plan: Optional[Dict[str, Any]] = None

        # Feature detector
        self._feature_detector = None
        self._matcher = None
        if HAS_OPENCV:
            self._init_feature_detector()

    def _init_feature_detector(self):
        """Initialize ORB feature detector and brute-force matcher."""
        try:
            self._feature_detector = cv2.ORB_create(nfeatures=2000)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        except Exception as exc:
            logger.error("Failed to initialize feature detector: %s", exc)

    # ------------------------------------------------------------------
    # Camera calibration
    # ------------------------------------------------------------------

    def calibrate_camera(
        self,
        images: List[np.ndarray],
        board_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025,
    ) -> Dict[str, Any]:
        """
        Calibrate camera using chessboard images.

        Args:
            images: List of calibration images (grayscale or BGR).
            board_size: Inner corners of chessboard (cols, rows).
            square_size: Size of each square in meters.

        Returns:
            Dict with calibration results.
        """
        if not HAS_OPENCV:
            return {"success": False, "message": "OpenCV not available."}

        if len(images) < 3:
            return {"success": False, "message": "Need at least 3 calibration images."}

        # Prepare object points
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        objp *= square_size

        obj_points = []  # 3D points in real-world space
        img_points = []  # 2D points in image plane
        img_size = None

        for img in images:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            if img_size is None:
                img_size = (gray.shape[1], gray.shape[0])

            found, corners = cv2.findChessboardCorners(gray, board_size, None)
            if found:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points.append(objp)
                img_points.append(corners_refined)

        if len(obj_points) < 3:
            return {
                "success": False,
                "message": f"Only found chessboard in {len(obj_points)} images (need >= 3).",
            }

        # Calibrate
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None
        )

        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.is_calibrated = True

        return {
            "success": True,
            "rms_error": round(ret, 4),
            "focal_length": [round(mtx[0, 0], 2), round(mtx[1, 1], 2)],
            "principal_point": [round(mtx[0, 2], 2), round(mtx[1, 2], 2)],
            "images_used": len(obj_points),
        }

    def set_default_calibration(self, width: int = 1280, height: int = 720):
        """Set a reasonable default camera calibration for a typical webcam."""
        focal = max(width, height) * 0.8
        self.camera_matrix = np.array([
            [focal, 0, width / 2],
            [0, focal, height / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)
        self.is_calibrated = True
        logger.info("Set default camera calibration for %dx%d", width, height)

    # ------------------------------------------------------------------
    # Frame recording
    # ------------------------------------------------------------------

    def record_walkthrough(self, frame: np.ndarray) -> int:
        """
        Add a frame from a walkthrough recording.
        Returns total frame count.
        """
        if not HAS_OPENCV:
            return 0

        self.frames.append(frame.copy())
        return len(self.frames)

    def clear_frames(self):
        """Clear all recorded frames."""
        self.frames.clear()
        self.keypoints_list.clear()
        self.descriptors_list.clear()
        self.point_cloud = None
        self.camera_poses.clear()
        logger.info("Cleared all frames and reconstruction data.")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self) -> Dict[str, Any]:
        """Extract ORB features from all recorded frames."""
        if not HAS_OPENCV or self._feature_detector is None:
            return {"success": False, "message": "OpenCV/ORB not available."}

        if len(self.frames) < 2:
            return {"success": False, "message": "Need at least 2 frames."}

        self.keypoints_list.clear()
        self.descriptors_list.clear()

        total_keypoints = 0
        for i, frame in enumerate(self.frames):
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            kp, desc = self._feature_detector.detectAndCompute(gray, None)
            self.keypoints_list.append(kp)
            self.descriptors_list.append(desc if desc is not None else np.array([]))
            total_keypoints += len(kp)

        avg_kp = total_keypoints / len(self.frames) if self.frames else 0
        logger.info("Extracted features: %d total keypoints across %d frames", total_keypoints, len(self.frames))

        return {
            "success": True,
            "n_frames": len(self.frames),
            "total_keypoints": total_keypoints,
            "avg_keypoints_per_frame": round(avg_kp, 1),
        }

    # ------------------------------------------------------------------
    # SfM Reconstruction
    # ------------------------------------------------------------------

    def reconstruct_sfm(self) -> Dict[str, Any]:
        """
        Perform pairwise Structure from Motion reconstruction.
        Uses sequential frame pairs for relative pose estimation.
        """
        if not HAS_OPENCV:
            return {"success": False, "message": "OpenCV not available."}

        if len(self.descriptors_list) < 2:
            return {"success": False, "message": "Need features from at least 2 frames. Run extract_features first."}

        if not self.is_calibrated:
            # Use default calibration
            h, w = self.frames[0].shape[:2]
            self.set_default_calibration(w, h)

        all_points_3d = []
        self.camera_poses = [np.eye(4)]  # First camera at origin

        for i in range(len(self.descriptors_list) - 1):
            desc1 = self.descriptors_list[i]
            desc2 = self.descriptors_list[i + 1]
            kp1 = self.keypoints_list[i]
            kp2 = self.keypoints_list[i + 1]

            if len(desc1) == 0 or len(desc2) == 0:
                continue

            # Match features
            try:
                matches = self._matcher.knnMatch(desc1, desc2, k=2)
            except Exception:
                continue

            # Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 8:
                logger.warning("Frame pair %d-%d: only %d good matches (need >= 8)", i, i + 1, len(good_matches))
                continue

            # Get matched point coordinates
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

            # Find essential matrix
            E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is None:
                continue

            # Recover pose
            _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)

            # Build camera pose matrix
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t.flatten()

            # Accumulate pose relative to previous
            prev_pose = self.camera_poses[-1]
            abs_pose = prev_pose @ pose
            self.camera_poses.append(abs_pose)

            # Triangulate points
            P1 = self.camera_matrix @ prev_pose[:3, :]
            P2 = self.camera_matrix @ abs_pose[:3, :]

            inlier_pts1 = pts1[pose_mask.ravel() > 0]
            inlier_pts2 = pts2[pose_mask.ravel() > 0]

            if len(inlier_pts1) >= 2:
                points_4d = cv2.triangulatePoints(P1, P2, inlier_pts1.T, inlier_pts2.T)
                points_3d = (points_4d[:3] / points_4d[3]).T

                # Filter out points at infinity or behind camera
                valid_mask = (
                    (np.abs(points_3d[:, 0]) < 100) &
                    (np.abs(points_3d[:, 1]) < 100) &
                    (np.abs(points_3d[:, 2]) < 100) &
                    (points_3d[:, 2] > 0)
                )
                all_points_3d.append(points_3d[valid_mask])

        if all_points_3d:
            self.point_cloud = np.vstack(all_points_3d)
        else:
            self.point_cloud = np.array([]).reshape(0, 3)

        n_points = len(self.point_cloud) if self.point_cloud is not None else 0
        logger.info("SfM reconstruction: %d 3D points, %d camera poses", n_points, len(self.camera_poses))

        return {
            "success": True,
            "n_points": n_points,
            "n_cameras": len(self.camera_poses),
            "n_frame_pairs": len(self.descriptors_list) - 1,
        }

    # ------------------------------------------------------------------
    # Floor plan generation
    # ------------------------------------------------------------------

    def generate_floor_plan(self, scale: float = 1.0) -> Dict[str, Any]:
        """
        Generate a simplified floor plan from the 3D point cloud.
        Projects points onto the XZ plane and finds bounding structures.
        """
        if self.point_cloud is None or len(self.point_cloud) == 0:
            return {"success": False, "message": "No 3D points available. Run reconstruct_sfm first."}

        points = self.point_cloud * scale

        # Project to XZ plane (top-down view)
        xz_points = points[:, [0, 2]]

        # Compute bounding box
        x_min, z_min = xz_points.min(axis=0)
        x_max, z_max = xz_points.max(axis=0)
        width = x_max - x_min
        length = z_max - z_min

        # Simple room detection: divide space into grid cells and
        # identify occupied regions as potential rooms
        grid_res = 0.5  # meters
        n_x = max(1, int(width / grid_res) + 1)
        n_z = max(1, int(length / grid_res) + 1)

        grid = np.zeros((n_z, n_x), dtype=int)
        for pt in xz_points:
            ix = min(int((pt[0] - x_min) / grid_res), n_x - 1)
            iz = min(int((pt[1] - z_min) / grid_res), n_z - 1)
            grid[iz, ix] += 1

        # Threshold: cells with enough points are "structure"
        threshold = max(1, np.percentile(grid[grid > 0], 25)) if np.any(grid > 0) else 1

        # Generate a single-room floor plan from bounding box
        room = {
            "id": "scanned_room",
            "name": "Scanned Room",
            "color": "#8BC34A",
            "walls": [
                [round(x_min, 2), round(z_min, 2), round(x_max, 2), round(z_min, 2)],
                [round(x_max, 2), round(z_min, 2), round(x_max, 2), round(z_max, 2)],
                [round(x_max, 2), round(z_max, 2), round(x_min, 2), round(z_max, 2)],
                [round(x_min, 2), round(z_max, 2), round(x_min, 2), round(z_min, 2)],
            ],
            "center": {"x": round((x_min + x_max) / 2, 2), "z": round((z_min + z_max) / 2, 2)},
            "width": round(width, 2),
            "depth": round(length, 2),
        }

        self.generated_floor_plan = {
            "rooms": [room],
            "dimensions": {"width": round(width, 2), "length": round(length, 2)},
            "wallHeight": 2.8,
            "router": {"x": round((x_min + x_max) / 2, 2), "z": round((z_min + z_max) / 2, 2)},
            "source": "sfm_scan",
            "n_points": len(self.point_cloud),
        }

        return {
            "success": True,
            "floor_plan": self.generated_floor_plan,
            "bounds": {
                "x_min": round(x_min, 2), "x_max": round(x_max, 2),
                "z_min": round(z_min, 2), "z_max": round(z_max, 2),
            },
            "grid_size": [n_x, n_z],
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get scanner status summary."""
        return {
            "opencv_available": HAS_OPENCV,
            "is_calibrated": self.is_calibrated,
            "n_frames": len(self.frames),
            "n_keypoints": sum(len(kp) for kp in self.keypoints_list),
            "n_3d_points": len(self.point_cloud) if self.point_cloud is not None else 0,
            "n_camera_poses": len(self.camera_poses),
            "has_floor_plan": self.generated_floor_plan is not None,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    scanner = HouseScanner3D()
    print("Scanner status:", scanner.get_status())
