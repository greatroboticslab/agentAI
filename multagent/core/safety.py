"""
Cloud-side safety policies.
Validates commands before sending to edge devices.
"""

import logging
from typing import Any, Dict, Set

from core.events import HardwareCommand

logger = logging.getLogger(__name__)


class SafetyPolicy:
    """Cloud-side safety validation for hardware commands."""

    def __init__(self):
        self._enabled = True
        self._blocked_commands: Set[str] = set()
        self._max_laser_power: float = 1.0
        self._max_vehicle_speed: float = 1.0

    def validate_command(self, command: HardwareCommand) -> bool:
        """
        Validate a hardware command against safety policies.
        Returns True if command is safe to send, False otherwise.
        """
        if not self._enabled:
            return True

        if command.command_type in self._blocked_commands:
            logger.warning(f"Blocked command: {command.command_type}")
            return False

        # Check laser power limits
        if command.command_type == "LASER_POWER":
            power = command.params.get("value", 0)
            if power > self._max_laser_power:
                logger.warning(f"Laser power {power} exceeds max {self._max_laser_power}")
                return False

        # Check vehicle speed limits
        if command.command_type == "VEHICLE_SPEED":
            speed = command.params.get("value", 0)
            if speed > self._max_vehicle_speed:
                logger.warning(f"Vehicle speed {speed} exceeds max {self._max_vehicle_speed}")
                return False

        return True

    def set_max_laser_power(self, power: float) -> None:
        self._max_laser_power = max(0.0, min(1.0, power))

    def set_max_vehicle_speed(self, speed: float) -> None:
        self._max_vehicle_speed = max(0.0, min(1.0, speed))

    def block_command(self, command_type: str) -> None:
        self._blocked_commands.add(command_type)

    def unblock_command(self, command_type: str) -> None:
        self._blocked_commands.discard(command_type)

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False
        logger.warning("Safety policies DISABLED")
