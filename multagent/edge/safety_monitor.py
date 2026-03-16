"""
Local safety monitor: runs independently of cloud connection.
Emergency stop on heartbeat timeout, temperature limits, etc.
"""

import asyncio
import logging
import time
from typing import Optional

from edge.hardware_driver import ESP32Controller

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """
    Independent safety monitor running on edge device.
    Triggers emergency stop if cloud connection is lost or hardware limits exceeded.
    """

    HEARTBEAT_TIMEOUT_MS = 500
    MAX_CPU_TEMP = 85
    MAX_GPU_TEMP = 90
    CHECK_INTERVAL = 0.05  # 50ms

    def __init__(self, esp32: ESP32Controller):
        self.esp32 = esp32
        self._last_heartbeat: float = time.time()
        self._running = False
        self._emergency_active = False

    def update_heartbeat(self) -> None:
        """Called when heartbeat ACK received from cloud."""
        self._last_heartbeat = time.time()
        if self._emergency_active:
            logger.info("Heartbeat restored, clearing emergency state")
            self._emergency_active = False

    async def run(self) -> None:
        """Main safety monitoring loop."""
        self._running = True
        logger.info("SafetyMonitor started")

        while self._running:
            # Check heartbeat timeout
            elapsed_ms = (time.time() - self._last_heartbeat) * 1000
            if elapsed_ms > self.HEARTBEAT_TIMEOUT_MS:
                if not self._emergency_active:
                    self._emergency_stop("Cloud heartbeat timeout")

            # Check temperature (platform-dependent, simplified)
            temp = self._read_cpu_temp()
            if temp is not None and temp > self.MAX_CPU_TEMP:
                self._emergency_stop(f"CPU temperature too high: {temp}C")

            await asyncio.sleep(self.CHECK_INTERVAL)

    def _emergency_stop(self, reason: str) -> None:
        """Shut laser + stop motors immediately."""
        logger.warning(f"[SAFETY] EMERGENCY STOP: {reason}")
        self._emergency_active = True
        try:
            self.esp32.laser_off()
            self.esp32.vehicle_stop()
        except Exception as e:
            logger.error(f"Emergency stop hardware error: {e}")

    def _read_cpu_temp(self) -> Optional[float]:
        """Read CPU temperature. Returns None if not available."""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return float(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError):
            return None

    def stop(self) -> None:
        self._running = False

    @property
    def is_emergency(self) -> bool:
        return self._emergency_active
