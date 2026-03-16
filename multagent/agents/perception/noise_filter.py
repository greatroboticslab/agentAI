"""
Advanced noise filter for shaky hand movements and rough terrain.
Extracted from lasercar.py AdvancedNoiseFilter class.

3-stage pipeline:
  1. Outlier detection and rejection
  2. Moving average smoothing
  3. Velocity-based filtering
"""

from collections import deque

import numpy as np


class AdvancedNoiseFilter:
    """Advanced noise filter for shaky hand movements and rough terrain like grass."""

    def __init__(self, filter_strength=0.3, smoothing_window=5):
        self.filter_strength = filter_strength
        self.smoothing_window = smoothing_window
        self.position_history = {}
        self.raw_history = {}
        self.velocity_history = {}
        self.max_history = 20

        self.movement_threshold = 3.0
        self.velocity_smoothing = 0.7
        self.direction_consistency_weight = 0.8
        self.outlier_threshold = 50.0

    def filter_position(self, weed_id, raw_x, raw_y):
        """Apply advanced noise filtering with multiple stages."""
        if weed_id not in self.position_history:
            self._initialize_weed_tracking(weed_id, raw_x, raw_y)
            return np.array([raw_x, raw_y])

        raw_pos = np.array([raw_x, raw_y])

        # Stage 1: Outlier detection and rejection
        filtered_pos = self._detect_and_handle_outliers(weed_id, raw_pos)

        # Stage 2: Moving average smoothing
        smoothed_pos = self._apply_moving_average(weed_id, filtered_pos)

        # Stage 3: Velocity-based filtering
        final_pos = self._apply_velocity_filtering(weed_id, smoothed_pos)

        # Store results
        self.position_history[weed_id].append(final_pos)
        self.raw_history[weed_id].append(raw_pos)

        return final_pos

    def _initialize_weed_tracking(self, weed_id, raw_x, raw_y):
        """Initialize tracking for a new weed."""
        initial_pos = np.array([raw_x, raw_y])
        self.position_history[weed_id] = deque([initial_pos], maxlen=self.max_history)
        self.raw_history[weed_id] = deque([initial_pos], maxlen=self.max_history)
        self.velocity_history[weed_id] = deque(maxlen=self.max_history)

    def _detect_and_handle_outliers(self, weed_id, raw_pos):
        """Detect and handle outlier positions (sudden jumps)."""
        if len(self.position_history[weed_id]) < 2:
            return raw_pos

        last_filtered = self.position_history[weed_id][-1]
        distance = np.linalg.norm(raw_pos - last_filtered)

        if distance > self.outlier_threshold:
            if len(self.velocity_history[weed_id]) > 0:
                recent_velocity = np.mean(
                    list(self.velocity_history[weed_id])[-3:], axis=0
                )
                predicted_pos = last_filtered + recent_velocity * 0.1
                return predicted_pos
            else:
                return last_filtered

        return raw_pos

    def _apply_moving_average(self, weed_id, position):
        """Apply moving average smoothing to reduce jitter."""
        window_size = min(self.smoothing_window, len(self.raw_history[weed_id]))
        if window_size <= 1:
            return position

        recent_positions = list(self.raw_history[weed_id])[-window_size:]
        recent_positions.append(position)

        weights = np.exp(np.linspace(-1, 0, len(recent_positions)))
        weights /= np.sum(weights)

        smoothed = np.average(recent_positions, axis=0, weights=weights)
        return smoothed

    def _apply_velocity_filtering(self, weed_id, position):
        """Apply velocity-based filtering for consistent movement."""
        if len(self.position_history[weed_id]) < 2:
            return position

        last_pos = self.position_history[weed_id][-1]
        current_velocity = position - last_pos
        movement_magnitude = np.linalg.norm(current_velocity)

        if movement_magnitude < self.movement_threshold:
            filtered_pos = (
                (1 - self.filter_strength * 1.5) * position
                + self.filter_strength * 1.5 * last_pos
            )
        else:
            if len(self.velocity_history[weed_id]) > 0:
                recent_velocities = list(self.velocity_history[weed_id])[-3:]
                if len(recent_velocities) > 0:
                    avg_velocity = np.mean(recent_velocities, axis=0)
                    velocity_consistency = self._calculate_velocity_consistency(
                        current_velocity, avg_velocity
                    )
                    dynamic_filter_strength = self.filter_strength * (
                        1 - velocity_consistency * self.direction_consistency_weight
                    )
                    filtered_pos = (
                        (1 - dynamic_filter_strength) * position
                        + dynamic_filter_strength * last_pos
                    )
                else:
                    filtered_pos = (
                        (1 - self.filter_strength) * position
                        + self.filter_strength * last_pos
                    )
            else:
                filtered_pos = (
                    (1 - self.filter_strength) * position
                    + self.filter_strength * last_pos
                )

        # Store velocity for future reference
        if movement_magnitude > self.movement_threshold:
            smoothed_velocity = current_velocity
            if len(self.velocity_history[weed_id]) > 0:
                last_velocity = self.velocity_history[weed_id][-1]
                smoothed_velocity = (
                    (1 - self.velocity_smoothing) * current_velocity
                    + self.velocity_smoothing * last_velocity
                )
            self.velocity_history[weed_id].append(smoothed_velocity)

        return filtered_pos

    def _calculate_velocity_consistency(self, current_velocity, average_velocity):
        """Calculate how consistent current velocity is with recent trend."""
        if np.linalg.norm(average_velocity) < 0.1:
            return 0.0

        current_norm = np.linalg.norm(current_velocity)
        avg_norm = np.linalg.norm(average_velocity)

        if current_norm < 0.1 or avg_norm < 0.1:
            return 0.0

        current_dir = current_velocity / current_norm
        avg_dir = average_velocity / avg_norm

        dot_product = np.clip(np.dot(current_dir, avg_dir), -1, 1)
        angle_diff = np.arccos(dot_product)
        consistency = 1.0 - (angle_diff / np.pi)

        return max(0.0, consistency)

    # --- Parameter setters ---

    def set_filter_strength(self, strength):
        self.filter_strength = max(0.0, min(1.0, strength))

    def set_smoothing_window(self, window_size):
        self.smoothing_window = max(1, min(10, window_size))

    def set_movement_threshold(self, threshold):
        self.movement_threshold = max(1.0, threshold)

    def set_outlier_threshold(self, threshold):
        self.outlier_threshold = max(10.0, threshold)

    def get_filter_stats(self, weed_id):
        """Get filtering statistics for debugging."""
        if weed_id not in self.position_history:
            return None
        return {
            "positions_tracked": len(self.position_history[weed_id]),
            "velocities_tracked": len(self.velocity_history.get(weed_id, [])),
            "filter_strength": self.filter_strength,
            "smoothing_window": self.smoothing_window,
        }

    def remove_weed(self, weed_id):
        """Remove tracking data for a weed."""
        self.position_history.pop(weed_id, None)
        self.raw_history.pop(weed_id, None)
        self.velocity_history.pop(weed_id, None)
