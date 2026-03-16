"""Tests for Edge Client components."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from edge.hardware_driver import HeliosDAC, HeliosPoint, LASER_MAX, POINTS_PER_FRAME


# ==================== HeliosDAC Tests ====================

class TestHeliosDAC:
    def test_create_single_point_frame(self):
        dac = HeliosDAC()
        frame = dac.create_single_point_frame(2048, 2048)
        # All 1000 points should be the same
        assert frame[0].x == 2048
        assert frame[0].y == 2048
        assert frame[999].x == 2048

    def test_clamp_values(self):
        dac = HeliosDAC()
        frame = dac.create_single_point_frame(-100, 5000)
        assert frame[0].x == 0
        assert frame[0].y == LASER_MAX

    def test_create_pattern_frame(self):
        dac = HeliosDAC()
        points = [(1000, 1000), (2000, 2000), (3000, 3000)]
        frame = dac.create_pattern_frame(points)
        # Should have filled all 1000 points
        assert frame[0].x == 1000
        # Last point should be from the pattern
        assert frame[999].x in (1000, 2000, 3000)

    def test_empty_pattern_frame(self):
        dac = HeliosDAC()
        frame = dac.create_pattern_frame([])
        # Should not crash
        assert frame[0].x == 0

    def test_num_devices_before_init(self):
        dac = HeliosDAC()
        assert dac.num_devices == 0


# ==================== HeliosPoint Tests ====================

class TestHeliosPoint:
    def test_structure(self):
        p = HeliosPoint(x=2048, y=1024, r=255, g=128, b=64, i=130)
        assert p.x == 2048
        assert p.y == 1024
        assert p.r == 255
        assert p.i == 130


# ==================== CommandExecutor Tests ====================

class TestCommandExecutor:
    def test_laser_control(self):
        from unittest.mock import MagicMock
        from edge.command_executor import CommandExecutor
        from edge.hardware_driver import ESP32Controller

        dac = HeliosDAC()
        esp32 = MagicMock(spec=ESP32Controller)
        esp32.laser_enabled = False

        executor = CommandExecutor(dac=dac, esp32=esp32)

        result = executor.execute({"type": "laser_control", "action": "ON"})
        assert result["status"] == "ok"
        esp32.laser_on.assert_called_once()

    def test_vehicle_command(self):
        from unittest.mock import MagicMock
        from edge.command_executor import CommandExecutor
        from edge.hardware_driver import ESP32Controller

        dac = HeliosDAC()
        esp32 = MagicMock(spec=ESP32Controller)

        executor = CommandExecutor(dac=dac, esp32=esp32)

        result = executor.execute({"type": "vehicle", "action": "FORWARD", "speed": 60})
        assert result["status"] == "ok"
        esp32.vehicle_forward.assert_called_once_with(60)

    def test_unknown_command(self):
        from unittest.mock import MagicMock
        from edge.command_executor import CommandExecutor
        from edge.hardware_driver import ESP32Controller

        dac = HeliosDAC()
        esp32 = MagicMock(spec=ESP32Controller)

        executor = CommandExecutor(dac=dac, esp32=esp32)
        result = executor.execute({"type": "unknown_cmd"})
        assert result["status"] == "error"

    def test_invalid_json(self):
        from unittest.mock import MagicMock
        from edge.command_executor import CommandExecutor
        from edge.hardware_driver import ESP32Controller

        dac = HeliosDAC()
        esp32 = MagicMock(spec=ESP32Controller)

        executor = CommandExecutor(dac=dac, esp32=esp32)
        result = executor.execute("not valid json{{{")
        assert result["status"] == "error"
