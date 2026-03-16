"""
Target selection: choose which weed to fire at.
Extracted from lasercar.py _select_new_target() logic.
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TargetCandidate:
    """A weed candidate for targeting."""

    def __init__(
        self,
        weed_id: int,
        pixel_x: float,
        pixel_y: float,
        confidence: float,
        is_moving: bool = False,
        speed: float = 0.0,
        consistency: float = 0.0,
        first_seen: float = 0.0,
    ):
        self.weed_id = weed_id
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.confidence = confidence
        self.is_moving = is_moving
        self.speed = speed
        self.consistency = consistency
        self.first_seen = first_seen
        self.priority: float = 0.0


class TargetSelector:
    """Selects the highest-priority weed target from detection list."""

    def __init__(self, observation_time: float = 1.0):
        self.observation_time = observation_time
        self._targeted_ids: set = set()  # Already targeted weed IDs

    def select(
        self,
        candidates: List[TargetCandidate],
        current_target_id: Optional[int] = None,
        current_time: Optional[float] = None,
    ) -> Optional[TargetCandidate]:
        """
        Select the best target from candidates.

        Priority scoring:
        - Moving weeds: speed * consistency * confidence * time_factor
        - Static weeds: observation_time based
        - Already targeted weeds are skipped.
        """
        if not candidates:
            return None

        if current_time is None:
            current_time = time.time()

        # Filter out already-targeted weeds
        available = [
            c for c in candidates
            if c.weed_id not in self._targeted_ids
            and c.weed_id != current_target_id
        ]

        if not available:
            return None

        # Score each candidate
        best = None
        best_score = -1.0

        for c in available:
            observation_duration = current_time - c.first_seen

            if observation_duration < self.observation_time:
                continue  # Not observed long enough

            if c.is_moving:
                time_factor = min(1.0, observation_duration / 3.0)
                score = c.speed * c.consistency * c.confidence * time_factor
            else:
                score = c.confidence * min(1.0, observation_duration / self.observation_time)

            c.priority = score

            if score > best_score:
                best_score = score
                best = c

        return best

    def mark_targeted(self, weed_id: int) -> None:
        """Mark a weed as already targeted (skip in future selection)."""
        self._targeted_ids.add(weed_id)

    def clear_targeted(self, weed_id: int) -> None:
        """Remove a weed from the targeted set."""
        self._targeted_ids.discard(weed_id)

    def reset(self) -> None:
        """Clear all targeted weed records."""
        self._targeted_ids.clear()

    @property
    def targeted_count(self) -> int:
        return len(self._targeted_ids)
