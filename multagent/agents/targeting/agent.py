"""
TargetingAgent: Laser targeting, coordinate transformation, and firing control.
Subscribes to WeedDetectionEvent, selects targets, computes firing params, sends laser commands.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from pydantic import Field

from core.embodied_role import EmbodiedRole
from core.events import Event, HardwareCommand

from agents.targeting.coordinate_transform import CoordinateTransform
from agents.targeting.events import (
    FiringCompleteEvent,
    FiringStartedEvent,
    LaserStatusEvent,
    TargetSelectedEvent,
)
from agents.targeting.firing_controller import FiringController, FiringPhase
from agents.targeting.laser_patterns import LaserPatterns
from agents.targeting.target_selector import TargetCandidate, TargetSelector

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "observation_time": 1.0,
    "prediction_duration": 8.0,
    "prediction_delay": 1.5,
    "speed_scaling_factor": 0.85,
    "static_firing_duration": 15.0,
    "aiming_duration": 1.0,
    "min_confidence_for_execution": 0.25,
    "pattern_enabled": True,
    "pattern_type": "zigzag",
    "pattern_size": 80,
    "pattern_density": 0.7,
    "pattern_update_rate": 40.0,
    "num_motors": 2,
    "frame_width": 1920,
    "frame_height": 1080,
}


class TargetingAgent(EmbodiedRole):
    """
    Targeting Agent — the system's hands.
    Pipeline: WeedDetection → Target Selection → Coordinate Transform → Firing Control → Laser Commands
    """

    name: str = "Targeting"
    profile: str = "Laser targeting, coordinate transformation, and firing control"
    goal: str = "Precisely target and fire laser at detected weeds"
    capabilities: list = ["targeting", "firing", "laser_control"]
    subscribed_events: list = ["weed_detection", "param_update", "firing_complete"]
    params: Dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_PARAMS))

    # Sub-components
    _coord_transform: Optional[CoordinateTransform] = None
    _target_selector: Optional[TargetSelector] = None
    _firing_controller: Optional[FiringController] = None
    _laser_patterns: Optional[LaserPatterns] = None

    # State
    _current_target: Optional[TargetCandidate] = None
    _latest_weeds: list = []
    _firing_count: int = 0
    _targeting_enabled: bool = True

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._firing_count = 0
        self._latest_weeds = []

    def initialize(
        self,
        calibration_files: Optional[Dict[int, str]] = None,
    ) -> None:
        """Initialize sub-components. Call after construction."""
        num_motors = self.params.get("num_motors", 2)
        frame_w = self.params.get("frame_width", 1920)
        frame_h = self.params.get("frame_height", 1080)

        self._coord_transform = CoordinateTransform(
            num_motors=num_motors,
            frame_width=frame_w,
            frame_height=frame_h,
        )

        # Load calibration data if provided
        if calibration_files:
            for motor_idx, filepath in calibration_files.items():
                self._coord_transform.load_calibration(motor_idx, filepath)

        self._laser_patterns = LaserPatterns()
        self._target_selector = TargetSelector(
            observation_time=self.params.get("observation_time", 1.0),
        )
        self._firing_controller = FiringController(
            coord_transform=self._coord_transform,
            laser_patterns=self._laser_patterns,
            send_command=self._send_hw_command,
        )
        self._apply_params_to_controller()

        logger.info("TargetingAgent initialized")

    def _apply_params_to_controller(self) -> None:
        """Sync params to firing controller."""
        if self._firing_controller:
            fc = self._firing_controller
            fc.aiming_duration = self.params.get("aiming_duration", 1.0)
            fc.static_firing_duration = self.params.get("static_firing_duration", 15.0)
            fc.pattern_enabled = self.params.get("pattern_enabled", True)
            fc.pattern_type = self.params.get("pattern_type", "zigzag")
            fc.pattern_size = self.params.get("pattern_size", 80)
            fc.pattern_density = self.params.get("pattern_density", 0.7)
            fc.update_rate = self.params.get("pattern_update_rate", 40.0)

    async def _send_hw_command(self, cmd_data: dict) -> None:
        """Send hardware command via EdgeBridge."""
        cmd = HardwareCommand(
            command_type=cmd_data["type"],
            target_device=cmd_data.get("device", "dac"),
            parameters=cmd_data,
        )
        await self.send_hardware_command(cmd)

    async def run_loop(self) -> None:
        """Main loop: initialize and start processing."""
        if self._firing_controller is None:
            self.initialize()
        await super().run_loop()

    async def on_event(self, event: Event) -> None:
        """Handle incoming events."""
        if event.event_type == "weed_detection":
            await self._on_weed_detection(event)
        elif event.event_type == "param_update":
            if event.data.get("target_agent") == self.name:
                self.update_params(event.data["params"])

    async def _on_weed_detection(self, event: Event) -> None:
        """Process weed detection event: select target and fire."""
        if not self._targeting_enabled or self._firing_controller is None:
            return

        weeds = event.data.get("weeds", [])
        self._latest_weeds = weeds

        # If already firing, skip target selection
        if self._firing_controller.is_firing:
            return

        # Check if previous firing completed
        if self._firing_controller.phase == FiringPhase.COMPLETED:
            await self._on_firing_done()

        # Build candidates from weed data
        candidates = []
        for w in weeds:
            if not w.get("visible", True):
                continue

            # Get movement info from Perception agent (via registry)
            movement = self._get_movement_info(w["weed_id"])

            candidates.append(TargetCandidate(
                weed_id=w["weed_id"],
                pixel_x=w.get("filtered_x", w["pixel_x"]),
                pixel_y=w.get("filtered_y", w["pixel_y"]),
                confidence=w["confidence"],
                is_moving=movement.get("has_movement", False) if movement else False,
                speed=movement.get("speed", 0) if movement else 0,
                consistency=movement.get("direction_consistency", 0) if movement else 0,
                first_seen=w.get("first_seen", time.time()),
            ))

        # Select target
        target = self._target_selector.select(
            candidates,
            current_target_id=self._firing_controller.current_weed_id,
        )

        if target is None:
            return

        self._current_target = target
        await self._start_firing(target)

    async def _start_firing(self, target: TargetCandidate) -> None:
        """Begin firing at selected target."""
        # Get trajectory prediction from Perception
        trajectory = self._get_trajectory_prediction(target.weed_id)

        mode = "static"
        if trajectory and trajectory.get("is_moving") and \
                trajectory.get("confidence", 0) >= self.params.get("min_confidence_for_execution", 0.25):
            # Trajectory firing for moving weeds
            mode = "trajectory"
            points = trajectory["trajectory_points"]
            timestamps = trajectory.get("timestamps", [])

            if points and timestamps:
                await self._firing_controller.start_trajectory_firing(
                    target.weed_id, points, timestamps
                )
            else:
                await self._firing_controller.start_static_firing(
                    target.weed_id, target.pixel_x, target.pixel_y
                )
                mode = "static"
        else:
            # Static firing for stationary weeds
            await self._firing_controller.start_static_firing(
                target.weed_id, target.pixel_x, target.pixel_y
            )

        self._firing_count += 1

        # Publish events
        if self._event_bus:
            await self._event_bus.publish(TargetSelectedEvent(
                source=self.name,
                weed_id=target.weed_id,
                position=(target.pixel_x, target.pixel_y),
                priority=target.priority,
                mode=mode,
            ))
            await self._event_bus.publish(FiringStartedEvent(
                source=self.name,
                weed_id=target.weed_id,
                position=(target.pixel_x, target.pixel_y),
                mode=mode,
            ))

    async def _on_firing_done(self) -> None:
        """Handle firing completion."""
        weed_id = self._firing_controller.current_weed_id
        status = self._firing_controller.get_status()

        if weed_id is not None:
            self._target_selector.mark_targeted(weed_id)

        await self._firing_controller.stop()
        self._current_target = None

        if self._event_bus and weed_id is not None:
            await self._event_bus.publish(FiringCompleteEvent(
                source=self.name,
                weed_id=weed_id,
                duration=status.get("elapsed", 0),
                mode=status.get("mode", "static"),
            ))

    def _get_movement_info(self, weed_id: int) -> Optional[dict]:
        """Query Perception agent for weed movement info."""
        if not self._agent_registry:
            return None
        perception = self._agent_registry.get_agent("Perception")
        if perception and hasattr(perception, "get_movement_info"):
            return perception.get_movement_info(weed_id)
        return None

    def _get_trajectory_prediction(self, weed_id: int) -> Optional[dict]:
        """Query Perception agent for trajectory prediction."""
        if not self._agent_registry:
            return None
        perception = self._agent_registry.get_agent("Perception")
        if perception and hasattr(perception, "get_trajectory_prediction"):
            return perception.get_trajectory_prediction(
                weed_id,
                duration=self.params.get("prediction_duration", 8.0),
                speed_scale=self.params.get("speed_scaling_factor", 0.85),
            )
        return None

    def on_params_updated(self, changed_params: dict) -> None:
        """Apply parameter changes."""
        self._apply_params_to_controller()
        if self._target_selector and "observation_time" in changed_params:
            self._target_selector.observation_time = changed_params["observation_time"]

    def get_status(self) -> dict:
        status = super().get_status()
        status.update({
            "targeting_enabled": self._targeting_enabled,
            "is_firing": self._firing_controller.is_firing if self._firing_controller else False,
            "firing_count": self._firing_count,
            "current_target": self._current_target.weed_id if self._current_target else None,
            "calibrated": self._coord_transform.is_calibrated if self._coord_transform else False,
        })
        if self._firing_controller:
            status["firing_status"] = self._firing_controller.get_status()
        return status
