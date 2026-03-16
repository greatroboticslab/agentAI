"""
Firing controller: manages laser firing sequences.
Merged from lasercar.py StaticTargetingSystem + DualMotorAutonomousTrajectoryFollower.

In the new architecture, firing commands are sent via EdgeBridge (not direct hardware).
The controller computes DAC coordinates and emits HardwareCommand events.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from agents.targeting.coordinate_transform import CoordinateTransform
from agents.targeting.laser_patterns import LaserPatterns

logger = logging.getLogger(__name__)


class FiringPhase(Enum):
    IDLE = "idle"
    AIMING = "aiming"       # Laser OFF, moving to target
    FIRING = "firing"        # Laser ON, holding/following target
    COMPLETED = "completed"


class FiringMode(Enum):
    STATIC = "static"        # Fixed position firing
    TRAJECTORY = "trajectory"  # Follow predicted path


class FiringController:
    """
    Controls laser firing sequences.
    Computes DAC positions and sends commands via a callback.
    """

    def __init__(
        self,
        coord_transform: CoordinateTransform,
        laser_patterns: LaserPatterns,
        send_command: Optional[Callable[..., Coroutine]] = None,
    ):
        self.coord_transform = coord_transform
        self.laser_patterns = laser_patterns
        self._send_command = send_command  # async callback for hardware commands

        # Firing parameters
        self.aiming_duration: float = 1.0
        self.static_firing_duration: float = 15.0
        self.pattern_enabled: bool = True
        self.pattern_type: str = "zigzag"
        self.pattern_size: int = 80
        self.pattern_density: float = 0.7
        self.update_rate: float = 40.0  # Hz

        # State
        self._phase: FiringPhase = FiringPhase.IDLE
        self._mode: FiringMode = FiringMode.STATIC
        self._current_weed_id: Optional[int] = None
        self._firing_task: Optional[asyncio.Task] = None
        self._cancel_event: asyncio.Event = asyncio.Event()
        self._start_time: float = 0.0

    @property
    def phase(self) -> FiringPhase:
        return self._phase

    @property
    def is_firing(self) -> bool:
        return self._phase in (FiringPhase.AIMING, FiringPhase.FIRING)

    @property
    def current_weed_id(self) -> Optional[int]:
        return self._current_weed_id

    async def start_static_firing(
        self, weed_id: int, pixel_x: float, pixel_y: float
    ) -> None:
        """Start static firing at a fixed position."""
        await self.stop()

        self._current_weed_id = weed_id
        self._mode = FiringMode.STATIC
        self._cancel_event.clear()
        self._start_time = time.time()

        self._firing_task = asyncio.create_task(
            self._static_firing_loop(pixel_x, pixel_y)
        )

    async def start_trajectory_firing(
        self,
        weed_id: int,
        trajectory_points: List[Tuple[float, float]],
        timestamps: List[float],
    ) -> None:
        """Start trajectory-following firing along predicted path."""
        await self.stop()

        self._current_weed_id = weed_id
        self._mode = FiringMode.TRAJECTORY
        self._cancel_event.clear()
        self._start_time = time.time()

        self._firing_task = asyncio.create_task(
            self._trajectory_firing_loop(trajectory_points, timestamps)
        )

    async def stop(self) -> None:
        """Stop current firing sequence."""
        if self._firing_task and not self._firing_task.done():
            self._cancel_event.set()
            try:
                await asyncio.wait_for(self._firing_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._firing_task.cancel()

        # Ensure laser is OFF
        await self._laser_off()
        self._phase = FiringPhase.IDLE
        self._current_weed_id = None

    async def _static_firing_loop(self, pixel_x: float, pixel_y: float) -> None:
        """Execute static firing: aim phase → fire phase."""
        update_interval = 1.0 / self.update_rate

        try:
            # Phase 1: AIMING (laser OFF, move to position)
            self._phase = FiringPhase.AIMING
            await self._laser_off()
            aim_start = time.time()

            while time.time() - aim_start < self.aiming_duration:
                if self._cancel_event.is_set():
                    return
                await self._send_dac_position(pixel_x, pixel_y)
                await asyncio.sleep(update_interval)

            # Phase 2: FIRING (laser ON, hold position)
            self._phase = FiringPhase.FIRING
            await self._laser_on()
            fire_start = time.time()

            while time.time() - fire_start < self.static_firing_duration:
                if self._cancel_event.is_set():
                    return
                await self._send_dac_position(pixel_x, pixel_y)
                await asyncio.sleep(update_interval)

            # Complete
            self._phase = FiringPhase.COMPLETED

        except asyncio.CancelledError:
            pass
        finally:
            await self._laser_off()

    async def _trajectory_firing_loop(
        self,
        trajectory_points: List[Tuple[float, float]],
        timestamps: List[float],
    ) -> None:
        """Execute trajectory-following firing."""
        if not trajectory_points or not timestamps:
            return

        update_interval = 1.0 / self.update_rate
        total_duration = timestamps[-1] if timestamps else 0

        try:
            # Brief aiming phase
            self._phase = FiringPhase.AIMING
            await self._laser_off()
            first_x, first_y = trajectory_points[0]
            aim_start = time.time()

            while time.time() - aim_start < self.aiming_duration:
                if self._cancel_event.is_set():
                    return
                await self._send_dac_position(first_x, first_y)
                await asyncio.sleep(update_interval)

            # Firing phase: follow trajectory
            self._phase = FiringPhase.FIRING
            await self._laser_on()
            exec_start = time.time()

            while True:
                if self._cancel_event.is_set():
                    return

                elapsed = time.time() - exec_start
                if elapsed >= total_duration:
                    break

                # Find current target point via binary search on timestamps
                target_idx = self._find_trajectory_index(timestamps, elapsed)
                px, py = trajectory_points[target_idx]

                await self._send_dac_position(px, py)
                await asyncio.sleep(update_interval)

            self._phase = FiringPhase.COMPLETED

        except asyncio.CancelledError:
            pass
        finally:
            await self._laser_off()

    def _find_trajectory_index(self, timestamps: List[float], elapsed: float) -> int:
        """Binary search for the trajectory point at the given time."""
        lo, hi = 0, len(timestamps) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if timestamps[mid] <= elapsed:
                lo = mid
            else:
                hi = mid - 1
        return lo

    async def _send_dac_position(self, pixel_x: float, pixel_y: float) -> None:
        """Transform pixel coords to DAC and send to all motors."""
        if not self._send_command:
            return

        for motor_idx in range(self.coord_transform.num_motors):
            laser_x, laser_y = self.coord_transform.transform(
                motor_idx, pixel_x, pixel_y
            )

            if self.pattern_enabled:
                points = self.laser_patterns.generate(
                    laser_x, laser_y,
                    self.pattern_type, self.pattern_size, self.pattern_density,
                )
                await self._send_command({
                    "type": "dac_pattern",
                    "motor": motor_idx,
                    "points": points,
                })
            else:
                await self._send_command({
                    "type": "dac_position",
                    "motor": motor_idx,
                    "x": laser_x,
                    "y": laser_y,
                })

    async def _laser_on(self) -> None:
        if self._send_command:
            await self._send_command({"type": "laser_control", "action": "ON"})

    async def _laser_off(self) -> None:
        if self._send_command:
            await self._send_command({"type": "laser_control", "action": "OFF"})

    def get_status(self) -> dict:
        elapsed = time.time() - self._start_time if self._start_time else 0
        total = self.static_firing_duration if self._mode == FiringMode.STATIC else 0
        return {
            "phase": self._phase.value,
            "mode": self._mode.value,
            "weed_id": self._current_weed_id,
            "elapsed": round(elapsed, 2),
            "total_duration": total,
            "progress": min(1.0, elapsed / total) if total > 0 else 0,
        }
