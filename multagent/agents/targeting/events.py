"""
Targeting-specific event types.
"""

from core.events import Event


class TargetSelectedEvent(Event):
    """New target selected for firing."""
    def __init__(self, source: str, weed_id: int, position: tuple,
                 priority: float, mode: str, **kwargs):
        super().__init__(
            event_type="target_selected",
            source=source,
            data={
                "weed_id": weed_id,
                "position": position,
                "priority": priority,
                "mode": mode,
            },
            **kwargs,
        )


class FiringStartedEvent(Event):
    """Firing sequence started."""
    def __init__(self, source: str, weed_id: int, position: tuple,
                 mode: str, **kwargs):
        super().__init__(
            event_type="firing_started",
            source=source,
            data={
                "weed_id": weed_id,
                "position": position,
                "mode": mode,
            },
            **kwargs,
        )


class FiringCompleteEvent(Event):
    """Firing sequence completed."""
    def __init__(self, source: str, weed_id: int, duration: float,
                 mode: str, **kwargs):
        super().__init__(
            event_type="firing_complete",
            source=source,
            data={
                "weed_id": weed_id,
                "duration": duration,
                "mode": mode,
            },
            **kwargs,
        )


class LaserStatusEvent(Event):
    """Laser status change."""
    def __init__(self, source: str, enabled: bool, power: int = 0, **kwargs):
        super().__init__(
            event_type="laser_status",
            source=source,
            data={"enabled": enabled, "power": power},
            **kwargs,
        )
