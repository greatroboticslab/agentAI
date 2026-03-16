"""
Vehicle command generation.
Sends movement commands to ESP32 via EdgeBridge.
"""

import logging
from typing import Any, Callable, Coroutine, Optional

from core.events import HardwareCommand

logger = logging.getLogger(__name__)


class VehicleCommands:
    """Generate and send vehicle control commands."""

    def __init__(self, send_command: Optional[Callable[..., Coroutine]] = None):
        self._send_command = send_command
        self.is_moving: bool = False
        self.current_speed: int = 0

    async def forward(self, speed: int = 50) -> None:
        """Move vehicle forward at given speed (0-100)."""
        speed = max(0, min(100, speed))
        await self._send({"type": "vehicle", "action": "FORWARD", "speed": speed})
        self.is_moving = True
        self.current_speed = speed
        logger.debug(f"Vehicle forward: speed={speed}")

    async def stop(self) -> None:
        """Stop vehicle."""
        await self._send({"type": "vehicle", "action": "STOP", "speed": 0})
        self.is_moving = False
        self.current_speed = 0
        logger.debug("Vehicle stopped")

    async def set_speed(self, speed: int) -> None:
        """Update speed without changing direction."""
        speed = max(0, min(100, speed))
        await self._send({"type": "vehicle", "action": "SPEED", "speed": speed})
        self.current_speed = speed

    async def _send(self, cmd_data: dict) -> None:
        if self._send_command:
            await self._send_command(cmd_data)
