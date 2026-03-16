"""
Self-optimizer: tracks performance metrics and suggests parameter tuning.
"""

import time
from collections import deque
from typing import Any, Dict, List, Optional


class PerformanceMetrics:
    """Tracks system performance metrics over time."""

    def __init__(self, window_size: int = 100):
        self._firing_results: deque = deque(maxlen=window_size)
        self._latencies: deque = deque(maxlen=window_size)
        self._detection_counts: deque = deque(maxlen=window_size)

    def record_firing(self, weed_id: int, duration: float, mode: str) -> None:
        self._firing_results.append({
            "timestamp": time.time(),
            "weed_id": weed_id,
            "duration": duration,
            "mode": mode,
        })

    def record_latency(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)

    def record_detections(self, count: int) -> None:
        self._detection_counts.append(count)

    @property
    def total_firings(self) -> int:
        return len(self._firing_results)

    @property
    def avg_firing_duration(self) -> float:
        if not self._firing_results:
            return 0
        return sum(r["duration"] for r in self._firing_results) / len(self._firing_results)

    @property
    def avg_latency(self) -> float:
        if not self._latencies:
            return 0
        return sum(self._latencies) / len(self._latencies)

    @property
    def avg_detections(self) -> float:
        if not self._detection_counts:
            return 0
        return sum(self._detection_counts) / len(self._detection_counts)

    def get_summary(self) -> dict:
        return {
            "total_firings": self.total_firings,
            "avg_firing_duration": round(self.avg_firing_duration, 2),
            "avg_yolo_latency_ms": round(self.avg_latency, 1),
            "avg_detections_per_frame": round(self.avg_detections, 1),
        }


class SelfOptimizer:
    """
    Analyzes performance trends and suggests parameter adjustments.
    Brain uses LLM to make final decisions based on these suggestions.
    """

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self._last_optimization_time: float = 0.0
        self._min_optimization_interval: float = 10.0  # seconds

    def update_from_events(self, events: list) -> None:
        """Extract performance data from buffered events."""
        for event in events:
            et = event.event_type
            data = event.data

            if et == "weed_detection":
                self.metrics.record_detections(len(data.get("weeds", [])))
                self.metrics.record_latency(data.get("yolo_latency_ms", 0))
            elif et == "firing_complete":
                self.metrics.record_firing(
                    data.get("weed_id", 0),
                    data.get("duration", 0),
                    data.get("mode", "static"),
                )

    def get_suggestions(self) -> List[dict]:
        """Generate parameter tuning suggestions based on metrics."""
        now = time.time()
        if now - self._last_optimization_time < self._min_optimization_interval:
            return []

        self._last_optimization_time = now
        suggestions = []

        # High latency -> lower YOLO confidence to speed up
        if self.metrics.avg_latency > 100:
            suggestions.append({
                "target": "Perception",
                "param": "yolo_confidence",
                "direction": "decrease",
                "reason": f"YOLO latency high ({self.metrics.avg_latency:.0f}ms)",
            })

        # Too many detections per frame -> increase confidence threshold
        if self.metrics.avg_detections > 20:
            suggestions.append({
                "target": "Perception",
                "param": "yolo_confidence",
                "direction": "increase",
                "reason": f"Too many detections ({self.metrics.avg_detections:.0f}/frame)",
            })

        return suggestions

    def get_metrics(self) -> dict:
        return self.metrics.get_summary()
