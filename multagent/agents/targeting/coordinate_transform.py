"""
Coordinate transform: camera pixels -> laser DAC coordinates.
Extracted from lasercar.py transform_coordinates() + load_calibration_data().

Uses KDTree-based weighted interpolation from calibration data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

LASER_MAX = 0xFFF  # 4095


class CoordinateTransform:
    """
    Transforms camera pixel coordinates to laser DAC coordinates.
    Supports multiple motors, each with independent calibration.
    """

    def __init__(
        self,
        num_motors: int = 2,
        frame_width: int = 1920,
        frame_height: int = 1080,
        weighted_k: int = 5,
    ):
        self.num_motors = num_motors
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.weighted_k = weighted_k

        # Per-motor calibration data
        self._calibration_data: Dict[int, list] = {}
        self._kdtrees: Dict[int, object] = {}
        self._regions: Dict[int, Optional[np.ndarray]] = {}

    def load_calibration(self, motor_index: int, filepath: str) -> bool:
        """Load calibration data from JSON file for a motor."""
        try:
            path = Path(filepath)
            if not path.exists():
                logger.warning(f"Calibration file not found: {filepath}")
                return False

            with open(path) as f:
                data = json.load(f)

            points = data.get("points", data) if isinstance(data, dict) else data
            if not points:
                logger.warning(f"No calibration points in {filepath}")
                return False

            self._calibration_data[motor_index] = points
            self._build_kdtree(motor_index)

            # Load region corners if present
            if isinstance(data, dict) and "region" in data:
                self._regions[motor_index] = np.array(data["region"], dtype=np.float32)

            logger.info(
                f"Motor {motor_index}: loaded {len(points)} calibration points from {filepath}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load calibration for motor {motor_index}: {e}")
            return False

    def _build_kdtree(self, motor_index: int) -> None:
        """Build KDTree from calibration points for fast nearest-neighbor lookup."""
        from scipy.spatial import KDTree

        points = self._calibration_data[motor_index]
        camera_coords = np.array([[p["camera_x"], p["camera_y"]] for p in points])
        self._kdtrees[motor_index] = KDTree(camera_coords)

    def transform(
        self, motor_index: int, camera_x: float, camera_y: float
    ) -> Tuple[int, int]:
        """
        Transform camera pixel coordinates to laser DAC coordinates.
        Uses weighted k-nearest-neighbor interpolation from calibration data.
        Falls back to linear mapping if no calibration available.
        """
        if motor_index not in self._kdtrees:
            return self._linear_fallback(camera_x, camera_y)

        kdtree = self._kdtrees[motor_index]
        cal_points = self._calibration_data[motor_index]

        k = min(self.weighted_k, len(cal_points))
        dists, indices = kdtree.query([camera_x, camera_y], k=k)

        # Handle single-point case
        if k == 1:
            dists = [dists]
            indices = [indices]

        # Filter valid indices
        valid = [(d, i) for d, i in zip(dists, indices) if i < len(cal_points)]
        if not valid:
            return self._linear_fallback(camera_x, camera_y)

        # Inverse distance squared weighting
        dists_arr = np.array([d for d, _ in valid])
        weights = 1.0 / (np.maximum(dists_arr, 1e-9) ** 2)
        weights /= np.sum(weights)

        laser_x = sum(
            cal_points[i]["laser_x"] * w for (_, i), w in zip(valid, weights)
        )
        laser_y = sum(
            cal_points[i]["laser_y"] * w for (_, i), w in zip(valid, weights)
        )

        return self._clamp(int(laser_x), int(laser_y))

    def _linear_fallback(self, camera_x: float, camera_y: float) -> Tuple[int, int]:
        """Simple linear mapping when no calibration data is available."""
        laser_x = int((camera_x / max(1, self.frame_width)) * LASER_MAX)
        laser_y = int((camera_y / max(1, self.frame_height)) * LASER_MAX)
        return self._clamp(laser_x, laser_y)

    def is_point_in_region(self, motor_index: int, x: float, y: float) -> bool:
        """Check if a camera-space point is within the laser targeting region."""
        if motor_index not in self._regions:
            return True  # No region defined = allow all

        import cv2

        region = self._regions[motor_index]
        result = cv2.pointPolygonTest(region, (float(x), float(y)), False)
        return result >= 0

    @staticmethod
    def _clamp(x: int, y: int) -> Tuple[int, int]:
        return (
            max(0, min(LASER_MAX, x)),
            max(0, min(LASER_MAX, y)),
        )

    @property
    def is_calibrated(self) -> bool:
        return len(self._kdtrees) > 0

    def get_status(self) -> dict:
        return {
            "num_motors": self.num_motors,
            "calibrated_motors": list(self._kdtrees.keys()),
            "points_per_motor": {
                m: len(pts) for m, pts in self._calibration_data.items()
            },
        }
