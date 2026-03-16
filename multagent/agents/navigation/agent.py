"""
NavigationAgent: Vehicle movement, mode management, and remote control.
Coordinates vehicle stop/go with firing events.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Set, Tuple

from pydantic import Field

from core.embodied_role import EmbodiedRole
from core.events import Event, HardwareCommand

from agents.navigation.events import (
    ModeChangeEvent,
    VehicleMovingEvent,
    VehicleStoppedEvent,
)
from agents.navigation.mode_manager import ModeManager, OperationMode
from agents.navigation.vehicle_commands import VehicleCommands

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "forward_speed": 50,
    "stationary_timeout": 5.0,
    "post_strike_advance": 0.2,
    "stabilization_time": 3.0,
    "swa_struck_zone_radius": 150,
}


class NavigationAgent(EmbodiedRole):
    """
    Navigation Agent — the system's legs.
    Manages vehicle movement, operation modes, and firing coordination.
    """

    name: str = "Navigation"
    profile: str = "Vehicle movement, mode management, and remote control"
    goal: str = "Coordinate vehicle movement with weed targeting"
    capabilities: list = ["navigation", "mode_management", "vehicle_control"]
    subscribed_events: list = [
        "firing_started", "firing_complete", "remote_control",
        "param_update", "mode_command",
    ]
    params: Dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_PARAMS))

    # Sub-components
    _mode_manager: Optional[ModeManager] = None
    _vehicle_cmds: Optional[VehicleCommands] = None

    # State
    _is_stopped_for_firing: bool = False
    _struck_weeds: Dict[int, Tuple[float, float]] = {}  # weed_id -> (x, y)
    _last_stop_time: float = 0.0

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._struck_weeds = {}

    def initialize(self) -> None:
        """Initialize sub-components."""
        self._mode_manager = ModeManager()
        self._vehicle_cmds = VehicleCommands(
            send_command=self._send_hw_command,
        )
        logger.info("NavigationAgent initialized")

    async def _send_hw_command(self, cmd_data: dict) -> None:
        """Send hardware command via EdgeBridge."""
        cmd = HardwareCommand(
            command_type=cmd_data["type"],
            target_device="esp32",
            parameters=cmd_data,
        )
        await self.send_hardware_command(cmd)

    async def run_loop(self) -> None:
        if self._mode_manager is None:
            self.initialize()
        await super().run_loop()

    async def on_event(self, event: Event) -> None:
        """Handle incoming events."""
        if event.event_type == "firing_started":
            await self._on_firing_started(event)
        elif event.event_type == "firing_complete":
            await self._on_firing_complete(event)
        elif event.event_type == "remote_control":
            await self._on_remote_control(event)
        elif event.event_type == "mode_command":
            await self._on_mode_command(event)
        elif event.event_type == "param_update":
            if event.data.get("target_agent") == self.name:
                self.update_params(event.data["params"])

    async def _on_firing_started(self, event: Event) -> None:
        """Firing started -> stop vehicle if in auto-patrol mode."""
        if self._mode_manager.current_mode != OperationMode.SWA:
            return
        if not self._vehicle_cmds.is_moving:
            return

        await self._vehicle_cmds.stop()
        self._is_stopped_for_firing = True
        self._last_stop_time = time.time()

        if self._event_bus:
            await self._event_bus.publish(VehicleStoppedEvent(
                source=self.name,
                reason="firing",
                weed_id=event.data.get("weed_id", 0),
            ))

    async def _on_firing_complete(self, event: Event) -> None:
        """Firing complete -> record struck weed, resume forward."""
        weed_id = event.data.get("weed_id", 0)
        position = event.data.get("position", (0, 0))

        # Record struck weed for SwA quadruple protection
        if weed_id:
            self._struck_weeds[weed_id] = position

        if self._mode_manager.current_mode != OperationMode.SWA:
            return
        if not self._is_stopped_for_firing:
            return

        # Brief advance to clear weed area
        advance_time = self.params.get("post_strike_advance", 0.2)
        speed = self.params.get("forward_speed", 50)

        await self._vehicle_cmds.forward(speed)
        self._is_stopped_for_firing = False

        if self._event_bus:
            await self._event_bus.publish(VehicleMovingEvent(
                source=self.name,
                speed=speed,
                direction="forward",
            ))

    async def _on_remote_control(self, event: Event) -> None:
        """Handle remote control mode switching."""
        new_mode = self._mode_manager.process_remote_data(event.data)
        if new_mode is None:
            return

        # Mode changed
        old_mode_name = event.data.get("old_mode", "IDLE")

        if new_mode == OperationMode.SWC:
            # Emergency stop
            await self._emergency_stop()
        elif new_mode == OperationMode.SWA:
            # Start auto-patrol
            speed = self.params.get("forward_speed", 50)
            await self._vehicle_cmds.forward(speed)
        elif new_mode == OperationMode.SWB:
            # Auto-static: stop vehicle
            await self._vehicle_cmds.stop()
        elif new_mode == OperationMode.IDLE:
            await self._vehicle_cmds.stop()

        if self._event_bus:
            await self._event_bus.publish(ModeChangeEvent(
                source=self.name,
                old_mode=self._mode_manager._prev_switches.get("mode", "IDLE"),
                new_mode=new_mode.value,
            ))

    async def _on_mode_command(self, event: Event) -> None:
        """Handle mode command from Brain agent."""
        mode_str = event.data.get("mode", "").upper()
        try:
            mode = OperationMode(mode_str)
            self._mode_manager.current_mode = mode
            logger.info(f"Mode set by Brain: {mode.value}")
        except ValueError:
            logger.warning(f"Unknown mode command: {mode_str}")

    async def _emergency_stop(self) -> None:
        """Emergency stop: halt vehicle and disable all auto systems."""
        await self._vehicle_cmds.stop()
        self._is_stopped_for_firing = False
        logger.warning("EMERGENCY STOP activated")

    def is_weed_already_struck(self, weed_id: int, pixel_x: float = 0, pixel_y: float = 0) -> bool:
        """
        SwA quadruple protection: check if weed was already struck.
        Checks both weed ID and spatial proximity.
        """
        # Check 1: Direct ID match
        if weed_id in self._struck_weeds:
            return True

        # Check 2: Spatial zone exclusion
        zone_radius = self.params.get("swa_struck_zone_radius", 150)
        for _, (sx, sy) in self._struck_weeds.items():
            dist = ((pixel_x - sx) ** 2 + (pixel_y - sy) ** 2) ** 0.5
            if dist < zone_radius:
                return True

        return False

    def on_params_updated(self, changed_params: dict) -> None:
        pass

    def get_status(self) -> dict:
        status = super().get_status()
        status.update({
            "current_mode": self._mode_manager.current_mode.value if self._mode_manager else "IDLE",
            "is_moving": self._vehicle_cmds.is_moving if self._vehicle_cmds else False,
            "current_speed": self._vehicle_cmds.current_speed if self._vehicle_cmds else 0,
            "is_stopped_for_firing": self._is_stopped_for_firing,
            "struck_weeds": len(self._struck_weeds),
        })
        return status
