"""
Weed trajectory prediction with YOLO processing delay compensation.
Extracted from lasercar.py WeedTrajectoryPredictor class.
"""

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np


class WeedTrajectoryPredictor:
    """Enhanced trajectory prediction system with YOLO processing delay compensation."""

    def __init__(self, max_history_length=20):
        self.max_history_length = max_history_length
        self.weed_trajectories: Dict[int, Dict[str, Any]] = {}
        self.min_movement_threshold = 2.0

        # YOLO processing delay compensation
        self.yolo_processing_delay = 1.0
        self.max_yolo_delay = 3.0
        self.min_yolo_delay = 0.2

    def set_yolo_delay(self, delay_seconds: float) -> None:
        """Set YOLO processing delay for prediction compensation."""
        self.yolo_processing_delay = max(
            self.min_yolo_delay, min(delay_seconds, self.max_yolo_delay)
        )

    def update_weed_position(self, weed_id: int, pixel_x: float, pixel_y: float, timestamp: float) -> None:
        """Update weed position and calculate trajectory with delay compensation."""
        compensated_timestamp = timestamp + self.yolo_processing_delay

        if weed_id not in self.weed_trajectories:
            self.weed_trajectories[weed_id] = {
                "positions": deque(maxlen=self.max_history_length),
                "timestamps": deque(maxlen=self.max_history_length),
                "velocities": deque(maxlen=self.max_history_length),
                "last_update": compensated_timestamp,
                "trajectory_confidence": 0.0,
                "movement_detected": False,
                "average_velocity": np.array([0.0, 0.0]),
                "velocity_consistency": 0.0,
                "observation_duration": 0.0,
                "processing_delay": self.yolo_processing_delay,
            }

        trajectory = self.weed_trajectories[weed_id]
        current_pos = np.array([pixel_x, pixel_y])

        trajectory["positions"].append(current_pos)
        trajectory["timestamps"].append(compensated_timestamp)
        trajectory["last_update"] = compensated_timestamp
        trajectory["processing_delay"] = self.yolo_processing_delay

        if len(trajectory["timestamps"]) > 1:
            trajectory["observation_duration"] = (
                compensated_timestamp - trajectory["timestamps"][0]
            )

        if len(trajectory["positions"]) >= 2:
            self._calculate_motion_parameters(trajectory)

    def _calculate_motion_parameters(self, trajectory: dict) -> None:
        """Calculate velocity and detect meaningful movement."""
        positions = list(trajectory["positions"])
        timestamps = list(trajectory["timestamps"])

        if len(positions) < 2:
            return

        pos_current = positions[-1]
        pos_previous = positions[-2]
        dt = timestamps[-1] - timestamps[-2]

        if dt > 0:
            displacement = pos_current - pos_previous
            velocity = displacement / dt
            movement_distance = np.linalg.norm(displacement)

            if movement_distance > self.min_movement_threshold:
                trajectory["velocities"].append(velocity)
                trajectory["movement_detected"] = True
                self._update_velocity_statistics(trajectory)
            else:
                trajectory["velocities"].append(np.array([0.0, 0.0]))

    def _update_velocity_statistics(self, trajectory: dict) -> None:
        """Update velocity statistics for better prediction."""
        velocities = list(trajectory["velocities"])
        if len(velocities) < 2:
            return

        non_zero = [v for v in velocities if np.linalg.norm(v) > 0.5]

        if len(non_zero) >= 2:
            trajectory["average_velocity"] = np.mean(non_zero, axis=0)

            if len(non_zero) >= 3:
                directions = []
                for vel in non_zero:
                    vel_norm = np.linalg.norm(vel)
                    if vel_norm > 0:
                        directions.append(vel / vel_norm)

                if len(directions) >= 2:
                    angle_diffs = []
                    for i in range(1, len(directions)):
                        dot = np.clip(np.dot(directions[i - 1], directions[i]), -1, 1)
                        angle_diffs.append(np.arccos(dot))

                    if angle_diffs:
                        avg_angle_diff = np.mean(angle_diffs)
                        trajectory["velocity_consistency"] = max(
                            0.0, 1.0 - (avg_angle_diff / np.pi)
                        )

        trajectory["trajectory_confidence"] = self._calculate_trajectory_confidence(
            trajectory
        )

    def _calculate_trajectory_confidence(self, trajectory: dict) -> float:
        """Calculate confidence level of trajectory prediction."""
        if len(trajectory["positions"]) < 2:
            return 0.1

        positions = list(trajectory["positions"])
        timestamps = list(trajectory["timestamps"])

        # Fast trajectory method
        if len(positions) >= 2:
            observation_time = trajectory.get("observation_duration", 0)

            if observation_time >= 0.8:
                start_pos = positions[0]
                end_pos = positions[-1]
                total_displacement = np.linalg.norm(end_pos - start_pos)

                if total_displacement > 10.0:
                    avg_velocity = (end_pos - start_pos) / observation_time
                    consistency_score = self._check_trajectory_consistency(
                        positions, timestamps, start_pos, avg_velocity
                    )

                    time_factor = min(1.0, observation_time / 1.0)
                    movement_factor = min(1.0, total_displacement / 50.0)
                    consistency_factor = consistency_score
                    delay_factor = max(
                        0.7,
                        1.0 - (trajectory.get("processing_delay", 1.0) / 5.0),
                    )

                    fast_confidence = (
                        time_factor
                        * movement_factor
                        * consistency_factor
                        * delay_factor
                    )

                    if fast_confidence > 0.4:
                        trajectory["fast_trajectory_ready"] = True
                        trajectory["start_position"] = start_pos
                        trajectory["end_position"] = end_pos
                        trajectory["trajectory_velocity"] = avg_velocity
                        return min(0.9, fast_confidence)

        # Fallback method
        observation_time = trajectory.get("observation_duration", 0)
        time_confidence = min(1.0, observation_time / 1.0)
        position_confidence = min(1.0, len(trajectory["positions"]) / 8.0)
        movement_confidence = (
            1.0 if trajectory.get("movement_detected", False) else 0.1
        )
        consistency_confidence = trajectory.get("velocity_consistency", 0.0)
        delay_factor = max(
            0.7, 1.0 - (trajectory.get("processing_delay", 1.0) / 5.0)
        )

        overall = (
            time_confidence
            * position_confidence
            * movement_confidence
            * (0.5 + 0.5 * consistency_confidence)
            * delay_factor
        )
        return max(0.1, min(1.0, overall))

    def _check_trajectory_consistency(
        self, positions, timestamps, start_pos, avg_velocity
    ) -> float:
        """Check if intermediate positions are consistent with trajectory."""
        if len(positions) < 3:
            return 1.0

        scores = []
        start_time = timestamps[0]

        for i in range(1, len(positions) - 1):
            time_elapsed = timestamps[i] - start_time
            predicted_pos = start_pos + avg_velocity * time_elapsed
            deviation = np.linalg.norm(positions[i] - predicted_pos)
            max_allowed = 30.0
            scores.append(max(0.0, 1.0 - (deviation / max_allowed)))

        return np.mean(scores) if scores else 1.0

    def predict_complete_trajectory(
        self,
        weed_id: int,
        prediction_duration_seconds: float,
        speed_scaling_factor: float = 1.0,
        time_step: float = 0.1,
    ) -> Optional[Dict[str, Any]]:
        """Predict complete future trajectory path."""
        if weed_id not in self.weed_trajectories:
            return None

        trajectory = self.weed_trajectories[weed_id]
        positions = list(trajectory["positions"])

        if len(positions) < 2:
            return None

        current_position = positions[-1]

        if not trajectory.get("movement_detected", False):
            return {
                "trajectory_points": [current_position],
                "timestamps": [0.0],
                "confidence": 0.1,
                "is_moving": False,
                "delay_compensated": True,
            }

        # Use fast trajectory if available
        if trajectory.get("fast_trajectory_ready", False):
            velocity = trajectory.get(
                "trajectory_velocity", np.array([0.0, 0.0])
            )
        else:
            velocity = trajectory.get("average_velocity", np.array([0.0, 0.0]))

        velocity = velocity * speed_scaling_factor

        if np.linalg.norm(velocity) < 1.0:
            return None

        # Generate trajectory with delay compensation
        delay_compensation = trajectory.get(
            "processing_delay", self.yolo_processing_delay
        )
        start_position = current_position + velocity * delay_compensation

        num_points = int(prediction_duration_seconds / time_step)
        trajectory_points = []
        timestamps = []

        for i in range(num_points + 1):
            t = i * time_step
            trajectory_points.append(start_position + velocity * t)
            timestamps.append(t)

        return {
            "trajectory_points": trajectory_points,
            "timestamps": timestamps,
            "confidence": trajectory["trajectory_confidence"],
            "is_moving": True,
            "velocity": velocity,
            "speed": float(np.linalg.norm(velocity)),
            "delay_compensated": True,
            "compensation_applied": delay_compensation,
            "method": "fast"
            if trajectory.get("fast_trajectory_ready", False)
            else "average",
        }

    def get_movement_info(self, weed_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed movement information."""
        if weed_id not in self.weed_trajectories:
            return None

        trajectory = self.weed_trajectories[weed_id]
        avg_vel = trajectory.get("average_velocity", np.array([0.0, 0.0]))
        speed = float(np.linalg.norm(avg_vel))
        direction = avg_vel / speed if speed > 0 else np.array([0.0, 0.0])

        return {
            "has_movement": trajectory.get("movement_detected", False),
            "speed": speed,
            "direction": direction,
            "consistency": trajectory.get("velocity_consistency", 0.0),
            "confidence": trajectory.get("trajectory_confidence", 0.0),
            "observation_duration": trajectory.get("observation_duration", 0.0),
            "processing_delay": trajectory.get("processing_delay", 0.0),
        }

    def cleanup_old_trajectories(self, current_time: float, max_age_seconds: float = 15.0) -> List[int]:
        """Remove old trajectory data. Returns list of removed weed IDs."""
        expired = [
            wid
            for wid, traj in self.weed_trajectories.items()
            if current_time - traj["last_update"] > max_age_seconds
        ]
        for wid in expired:
            del self.weed_trajectories[wid]
        return expired

    def remove_weed(self, weed_id: int) -> None:
        """Remove trajectory data for a specific weed."""
        self.weed_trajectories.pop(weed_id, None)
