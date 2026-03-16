"""
Weed tracking + ID assignment.
Extracted from lasercar.py _update_weed_tracking() and _cleanup_old_weeds().

Uses spatial matching (scipy cdist) for cross-frame weed association.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackedWeed:
    """A tracked weed with persistent ID across frames."""
    weed_id: int
    pixel_x: int
    pixel_y: int
    confidence: float
    box: tuple  # (x1, y1, x2, y2)
    first_seen: float
    last_seen: float
    visible_this_frame: bool = True
    targeted: bool = False
    area_fraction: float = 0.0
    aspect_ratio: float = 1.0
    filtered_x: float = 0.0
    filtered_y: float = 0.0


class WeedTracker:
    """
    Weed tracking with persistent ID assignment.
    Associates detections across frames using spatial proximity.
    """

    def __init__(self, match_threshold: float = 50.0, max_age: float = 10.0):
        self.tracked_weeds: List[TrackedWeed] = []
        self._weed_counter: int = 0
        self.match_threshold = match_threshold
        self.max_age = max_age

    def update(self, detections: list, current_time: float) -> List[TrackedWeed]:
        """
        Update tracking with new detections.
        Returns list of all currently tracked weeds.

        Args:
            detections: List of Detection objects from YoloDetector.
            current_time: Current timestamp.
        """
        # Reset visibility flag
        for weed in self.tracked_weeds:
            weed.visible_this_frame = False

        used_detections = set()

        # Match existing weeds to new detections
        if self.tracked_weeds and detections:
            existing_positions = np.array(
                [[w.pixel_x, w.pixel_y] for w in self.tracked_weeds]
            )
            new_positions = np.array(
                [[d.pixel_x, d.pixel_y] for d in detections]
            )

            from scipy.spatial.distance import cdist
            distances = cdist(existing_positions, new_positions)

            # Greedy nearest-neighbor matching
            for i, weed in enumerate(self.tracked_weeds):
                best_idx = -1
                min_dist = self.match_threshold

                for j in range(len(detections)):
                    if j not in used_detections and distances[i, j] < min_dist:
                        min_dist = distances[i, j]
                        best_idx = j

                if best_idx != -1:
                    det = detections[best_idx]
                    weed.pixel_x = det.pixel_x
                    weed.pixel_y = det.pixel_y
                    weed.confidence = det.confidence
                    weed.box = det.box
                    weed.last_seen = current_time
                    weed.visible_this_frame = True
                    weed.area_fraction = det.area_fraction
                    weed.aspect_ratio = det.aspect_ratio
                    used_detections.add(best_idx)

        # Create new weeds for unmatched detections
        new_weeds = []
        for i, det in enumerate(detections):
            if i not in used_detections:
                self._weed_counter += 1
                new_weed = TrackedWeed(
                    weed_id=self._weed_counter,
                    pixel_x=det.pixel_x,
                    pixel_y=det.pixel_y,
                    confidence=det.confidence,
                    box=det.box,
                    first_seen=current_time,
                    last_seen=current_time,
                    visible_this_frame=True,
                    area_fraction=det.area_fraction,
                    aspect_ratio=det.aspect_ratio,
                )
                self.tracked_weeds.append(new_weed)
                new_weeds.append(new_weed)

        return self.tracked_weeds

    def cleanup(self, current_time: float) -> List[int]:
        """
        Remove weeds not seen for max_age seconds.
        Returns list of removed weed IDs.
        """
        removed_ids = [
            w.weed_id
            for w in self.tracked_weeds
            if current_time - w.last_seen >= self.max_age
        ]
        self.tracked_weeds = [
            w for w in self.tracked_weeds
            if current_time - w.last_seen < self.max_age
        ]
        return removed_ids

    def get_visible_weeds(self) -> List[TrackedWeed]:
        """Get weeds visible in the current frame."""
        return [w for w in self.tracked_weeds if w.visible_this_frame]

    def get_weed(self, weed_id: int) -> Optional[TrackedWeed]:
        """Get a specific tracked weed by ID."""
        for w in self.tracked_weeds:
            if w.weed_id == weed_id:
                return w
        return None

    @property
    def count(self) -> int:
        return len(self.tracked_weeds)

    @property
    def visible_count(self) -> int:
        return sum(1 for w in self.tracked_weeds if w.visible_this_frame)
