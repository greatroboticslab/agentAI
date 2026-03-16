"""
Perception-specific event types.
"""

from core.events import Event


class WeedDetectionEvent(Event):
    """Published each frame after detection + tracking + filtering."""
    # data: {"weeds": list, "frame_shape": tuple, "yolo_latency_ms": float, "frame_timestamp": float}
    def __init__(self, source: str, weeds: list, frame_shape: tuple,
                 yolo_latency_ms: float, frame_timestamp: float, **kwargs):
        super().__init__(
            event_type="weed_detection",
            source=source,
            data={
                "weeds": weeds,
                "frame_shape": frame_shape,
                "yolo_latency_ms": yolo_latency_ms,
                "frame_timestamp": frame_timestamp,
            },
            **kwargs,
        )


class NewWeedEvent(Event):
    """New weed first appears."""
    def __init__(self, source: str, weed_id: int, position: tuple, confidence: float, **kwargs):
        super().__init__(
            event_type="new_weed",
            source=source,
            data={"weed_id": weed_id, "position": position, "confidence": confidence},
            **kwargs,
        )


class WeedLostEvent(Event):
    """Weed disappeared (cleaned up)."""
    def __init__(self, source: str, weed_id: int, last_position: tuple = (0, 0), **kwargs):
        super().__init__(
            event_type="weed_lost",
            source=source,
            data={"weed_id": weed_id, "last_position": last_position},
            **kwargs,
        )
