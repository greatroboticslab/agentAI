"""Tests for Targeting Agent sub-components."""

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.targeting.coordinate_transform import CoordinateTransform, LASER_MAX
from agents.targeting.laser_patterns import LaserPatterns
from agents.targeting.target_selector import TargetCandidate, TargetSelector
from agents.targeting.firing_controller import FiringController, FiringPhase, FiringMode


# ==================== CoordinateTransform Tests ====================

class TestCoordinateTransform:
    def setup_method(self):
        self.ct = CoordinateTransform(num_motors=2, frame_width=1920, frame_height=1080)

    def test_linear_fallback(self):
        """Without calibration, should use linear mapping."""
        x, y = self.ct.transform(0, 960, 540)  # Center of frame
        assert 0 <= x <= LASER_MAX
        assert 0 <= y <= LASER_MAX
        # Center should map to ~center of DAC range
        assert abs(x - LASER_MAX // 2) < 100
        assert abs(y - LASER_MAX // 2) < 100

    def test_linear_fallback_corners(self):
        x0, y0 = self.ct.transform(0, 0, 0)
        assert x0 == 0 and y0 == 0

        x1, y1 = self.ct.transform(0, 1920, 1080)
        assert x1 == LASER_MAX and y1 == LASER_MAX

    def test_clamp(self):
        x, y = CoordinateTransform._clamp(-10, 5000)
        assert x == 0
        assert y == LASER_MAX

    def test_not_calibrated(self):
        assert self.ct.is_calibrated is False

    def test_load_calibration_missing_file(self):
        result = self.ct.load_calibration(0, "/nonexistent/path.json")
        assert result is False

    def test_load_calibration_from_data(self, tmp_path):
        """Load calibration from a JSON file."""
        import json
        cal_data = {
            "points": [
                {"camera_x": 0, "camera_y": 0, "laser_x": 0, "laser_y": 0},
                {"camera_x": 960, "camera_y": 540, "laser_x": 2048, "laser_y": 2048},
                {"camera_x": 1920, "camera_y": 1080, "laser_x": 4095, "laser_y": 4095},
            ]
        }
        cal_file = tmp_path / "cal_motor_0.json"
        cal_file.write_text(json.dumps(cal_data))

        result = self.ct.load_calibration(0, str(cal_file))
        assert result is True
        assert self.ct.is_calibrated is True

        # Transform center should be close to calibration center
        x, y = self.ct.transform(0, 960, 540)
        assert abs(x - 2048) < 200
        assert abs(y - 2048) < 200

    def test_status(self):
        status = self.ct.get_status()
        assert status["num_motors"] == 2
        assert status["calibrated_motors"] == []


# ==================== LaserPatterns Tests ====================

class TestLaserPatterns:
    def setup_method(self):
        self.lp = LaserPatterns()

    def test_zigzag_generates_points(self):
        points = self.lp.generate(2048, 2048, "zigzag", 80, 0.7)
        assert len(points) > 0
        # Points should be near center
        for x, y in points:
            assert abs(x - 2048) <= 80
            assert abs(y - 2048) <= 80

    def test_circle_generates_points(self):
        points = self.lp.generate(2048, 2048, "circle", 80, 0.7)
        assert len(points) >= 8

    def test_cross_generates_points(self):
        points = self.lp.generate(2048, 2048, "cross", 80, 0.7)
        assert len(points) > 0

    def test_unknown_pattern_returns_center(self):
        points = self.lp.generate(100, 200, "unknown_pattern")
        assert points == [(100, 200)]

    def test_zigzag_density(self):
        low = self.lp.generate_zigzag(2048, 2048, 80, density=0.3)
        high = self.lp.generate_zigzag(2048, 2048, 80, density=1.0)
        assert len(high) > len(low)


# ==================== TargetSelector Tests ====================

class TestTargetSelector:
    def setup_method(self):
        self.ts = TargetSelector(observation_time=0.5)

    def _make_candidate(self, weed_id=1, x=100, y=200, conf=0.8,
                         moving=False, speed=0, first_seen=0):
        return TargetCandidate(
            weed_id=weed_id, pixel_x=x, pixel_y=y, confidence=conf,
            is_moving=moving, speed=speed, consistency=0.8,
            first_seen=first_seen,
        )

    def test_no_candidates(self):
        assert self.ts.select([], current_time=1.0) is None

    def test_select_single(self):
        c = self._make_candidate(first_seen=0.0)
        result = self.ts.select([c], current_time=1.0)
        assert result is not None
        assert result.weed_id == 1

    def test_observation_time_filter(self):
        """Weed not observed long enough should be skipped."""
        c = self._make_candidate(first_seen=0.9)
        result = self.ts.select([c], current_time=1.0)
        assert result is None

    def test_skip_targeted(self):
        self.ts.mark_targeted(1)
        c = self._make_candidate(weed_id=1, first_seen=0.0)
        result = self.ts.select([c], current_time=1.0)
        assert result is None

    def test_clear_targeted(self):
        self.ts.mark_targeted(1)
        self.ts.clear_targeted(1)
        c = self._make_candidate(weed_id=1, first_seen=0.0)
        result = self.ts.select([c], current_time=1.0)
        assert result is not None

    def test_priority_scoring(self):
        """Higher confidence should rank higher."""
        c1 = self._make_candidate(weed_id=1, conf=0.5, first_seen=0.0)
        c2 = self._make_candidate(weed_id=2, conf=0.9, first_seen=0.0)
        result = self.ts.select([c1, c2], current_time=1.0)
        assert result.weed_id == 2


# ==================== FiringController Tests ====================

class TestFiringController:
    def setup_method(self):
        self.commands_sent = []
        self.ct = CoordinateTransform(num_motors=1, frame_width=640, frame_height=480)
        self.lp = LaserPatterns()

        async def mock_send(cmd):
            self.commands_sent.append(cmd)

        self.fc = FiringController(
            coord_transform=self.ct,
            laser_patterns=self.lp,
            send_command=mock_send,
        )
        self.fc.aiming_duration = 0.1
        self.fc.static_firing_duration = 0.2
        self.fc.update_rate = 20.0

    def test_initial_state(self):
        assert self.fc.phase == FiringPhase.IDLE
        assert self.fc.is_firing is False

    @pytest.mark.asyncio
    async def test_static_firing_sequence(self):
        """Should go through AIMING -> FIRING -> COMPLETED."""
        await self.fc.start_static_firing(1, 320, 240)
        # Wait for completion
        await asyncio.sleep(0.5)
        assert self.fc.phase == FiringPhase.COMPLETED
        # Should have sent commands (laser off for aim, laser on for fire, laser off at end)
        laser_cmds = [c for c in self.commands_sent if c.get("type") == "laser_control"]
        assert len(laser_cmds) >= 2

    @pytest.mark.asyncio
    async def test_stop_during_firing(self):
        await self.fc.start_static_firing(1, 320, 240)
        await asyncio.sleep(0.05)
        await self.fc.stop()
        assert self.fc.phase == FiringPhase.IDLE

    @pytest.mark.asyncio
    async def test_trajectory_firing(self):
        points = [(100, 200), (150, 220), (200, 240)]
        timestamps = [0.0, 0.1, 0.2]
        await self.fc.start_trajectory_firing(1, points, timestamps)
        await asyncio.sleep(0.5)
        assert self.fc.phase == FiringPhase.COMPLETED

    def test_get_status(self):
        status = self.fc.get_status()
        assert status["phase"] == "idle"
        assert status["weed_id"] is None


# ==================== TargetingAgent Tests ====================

class TestTargetingAgent:
    @pytest.mark.asyncio
    async def test_initialize(self):
        from agents.targeting.agent import TargetingAgent
        agent = TargetingAgent()
        agent.initialize()
        assert agent._coord_transform is not None
        assert agent._firing_controller is not None
        assert agent._target_selector is not None

    @pytest.mark.asyncio
    async def test_status(self):
        from agents.targeting.agent import TargetingAgent
        agent = TargetingAgent()
        agent.initialize()
        status = agent.get_status()
        assert status["name"] == "Targeting"
        assert "firing_count" in status
        assert "is_firing" in status
