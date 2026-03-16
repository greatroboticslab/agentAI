"""
Navigation-specific event types.
"""

from core.events import Event


class VehicleStoppedEvent(Event):
    """Vehicle has stopped."""
    def __init__(self, source: str, reason: str = "", weed_id: int = 0, **kwargs):
        super().__init__(
            event_type="vehicle_stopped",
            source=source,
            data={"reason": reason, "weed_id": weed_id},
            **kwargs,
        )


class VehicleMovingEvent(Event):
    """Vehicle started moving."""
    def __init__(self, source: str, speed: int = 0, direction: str = "forward", **kwargs):
        super().__init__(
            event_type="vehicle_moving",
            source=source,
            data={"speed": speed, "direction": direction},
            **kwargs,
        )


class ModeChangeEvent(Event):
    """Operation mode changed."""
    def __init__(self, source: str, old_mode: str, new_mode: str, **kwargs):
        super().__init__(
            event_type="mode_change",
            source=source,
            data={"old_mode": old_mode, "new_mode": new_mode},
            **kwargs,
        )


class RemoteControlEvent(Event):
    """Remote control input received."""
    def __init__(self, source: str, channels: list, switches: dict, **kwargs):
        super().__init__(
            event_type="remote_control",
            source=source,
            data={"channels": channels, "switches": switches},
            **kwargs,
        )
