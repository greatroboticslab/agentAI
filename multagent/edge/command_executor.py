"""
Command executor: receives cloud JSON commands and calls hardware drivers.
"""

import json
import logging
from typing import Optional

from edge.hardware_driver import ESP32Controller, HeliosDAC

logger = logging.getLogger(__name__)


class CommandExecutor:
    """Executes hardware commands received from cloud."""

    def __init__(self, dac: HeliosDAC, esp32: ESP32Controller):
        self.dac = dac
        self.esp32 = esp32

    def execute(self, cmd_json: str) -> dict:
        """
        Execute a command received from cloud.

        Command format: {"type": "...", ...}
        Supported types:
          - laser_control: {"action": "ON"/"OFF", "power": int}
          - dac_position: {"motor": int, "x": int, "y": int}
          - dac_pattern: {"motor": int, "points": [(x,y)...]}
          - vehicle: {"action": "FORWARD"/"STOP", "speed": int}
          - camera: {"resolution": [w,h], "fps": int}
        """
        try:
            cmd = json.loads(cmd_json) if isinstance(cmd_json, str) else cmd_json
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON"}

        cmd_type = cmd.get("type", "")

        try:
            if cmd_type == "laser_control":
                return self._handle_laser(cmd)
            elif cmd_type == "dac_position":
                return self._handle_dac_position(cmd)
            elif cmd_type == "dac_pattern":
                return self._handle_dac_pattern(cmd)
            elif cmd_type == "vehicle":
                return self._handle_vehicle(cmd)
            else:
                return {"status": "error", "message": f"Unknown command type: {cmd_type}"}
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return {"status": "error", "message": str(e)}

    def _handle_laser(self, cmd: dict) -> dict:
        action = cmd.get("action", "").upper()
        if action == "ON":
            self.esp32.laser_on()
        elif action == "OFF":
            self.esp32.laser_off()

        power = cmd.get("power")
        if power is not None:
            self.esp32.set_laser_power(int(power))

        return {"status": "ok", "laser_enabled": self.esp32.laser_enabled}

    def _handle_dac_position(self, cmd: dict) -> dict:
        motor = cmd.get("motor", 0)
        x = cmd.get("x", 0)
        y = cmd.get("y", 0)
        frame = self.dac.create_single_point_frame(x, y)
        success = self.dac.send_frame(motor, frame)
        return {"status": "ok" if success else "error"}

    def _handle_dac_pattern(self, cmd: dict) -> dict:
        motor = cmd.get("motor", 0)
        points = cmd.get("points", [])
        # Convert to list of tuples
        pts = [(p[0], p[1]) if isinstance(p, (list, tuple)) else (p.get("x", 0), p.get("y", 0))
               for p in points]
        frame = self.dac.create_pattern_frame(pts)
        success = self.dac.send_frame(motor, frame)
        return {"status": "ok" if success else "error"}

    def _handle_vehicle(self, cmd: dict) -> dict:
        action = cmd.get("action", "").upper()
        speed = cmd.get("speed", 50)

        if action == "FORWARD":
            self.esp32.vehicle_forward(speed)
        elif action == "STOP":
            self.esp32.vehicle_stop()
        elif action == "SPEED":
            self.esp32.vehicle_forward(speed)

        return {"status": "ok", "action": action}
