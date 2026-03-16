"""
Mode manager: FlySky remote control mode switching.
Extracted from lasercar.py FlySkyRemoteControl._handle_switch_changes().

Modes:
  SwA: Auto-patrol (forward + detect + stop + fire + resume)
  SwB: Auto-static (vehicle stopped, auto-detect + fire)
  SwC: Emergency stop (all systems off)
  SwD: Manual control (user drives vehicle)
"""

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class OperationMode(Enum):
    IDLE = "IDLE"
    SWA = "SWA"   # Auto-patrol
    SWB = "SWB"   # Auto-static
    SWC = "SWC"   # Emergency stop
    SWD = "SWD"   # Manual


class ModeManager:
    """State machine for operation mode switching."""

    # FlySky switch value thresholds
    SWITCH_LOW = 1200
    SWITCH_HIGH = 1800

    def __init__(self):
        self.current_mode: OperationMode = OperationMode.IDLE
        self._prev_switches: dict = {}

    def process_remote_data(self, data: dict) -> Optional[OperationMode]:
        """
        Process remote control data and return new mode if changed.
        Returns None if mode hasn't changed.
        """
        switches = data.get("switches", {})
        if not switches:
            return None

        new_mode = self._determine_mode(switches)

        if new_mode != self.current_mode:
            old_mode = self.current_mode
            self.current_mode = new_mode
            logger.info(f"Mode change: {old_mode.value} -> {new_mode.value}")
            return new_mode

        self._prev_switches = switches
        return None

    def _determine_mode(self, switches: dict) -> OperationMode:
        """Determine operation mode from switch positions."""
        # SwC is emergency — always check first
        swc = switches.get("SwC", 0)
        if isinstance(swc, (int, float)) and swc > self.SWITCH_HIGH:
            return OperationMode.SWC

        swa = switches.get("SwA", 0)
        swb = switches.get("SwB", 0)
        swd = switches.get("SwD", 0)

        if isinstance(swd, (int, float)) and swd > self.SWITCH_HIGH:
            return OperationMode.SWD
        if isinstance(swa, (int, float)) and swa > self.SWITCH_HIGH:
            return OperationMode.SWA
        if isinstance(swb, (int, float)) and swb > self.SWITCH_HIGH:
            return OperationMode.SWB

        return OperationMode.IDLE

    @property
    def is_auto_mode(self) -> bool:
        return self.current_mode in (OperationMode.SWA, OperationMode.SWB)

    @property
    def is_emergency(self) -> bool:
        return self.current_mode == OperationMode.SWC

    @property
    def is_manual(self) -> bool:
        return self.current_mode == OperationMode.SWD
