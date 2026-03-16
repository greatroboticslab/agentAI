"""Tests for Navigation Agent sub-components."""

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.navigation.mode_manager import ModeManager, OperationMode
from agents.navigation.vehicle_commands import VehicleCommands


# ==================== ModeManager Tests ====================

class TestModeManager:
    def setup_method(self):
        self.mm = ModeManager()

    def test_initial_mode(self):
        assert self.mm.current_mode == OperationMode.IDLE

    def test_switch_to_swa(self):
        result = self.mm.process_remote_data({"switches": {"SwA": 2000}})
        assert result == OperationMode.SWA

    def test_switch_to_swb(self):
        result = self.mm.process_remote_data({"switches": {"SwB": 2000}})
        assert result == OperationMode.SWB

    def test_swc_emergency_priority(self):
        """SwC should take priority over all other switches."""
        result = self.mm.process_remote_data({
            "switches": {"SwA": 2000, "SwC": 2000}
        })
        assert result == OperationMode.SWC

    def test_swd_manual(self):
        result = self.mm.process_remote_data({"switches": {"SwD": 2000}})
        assert result == OperationMode.SWD

    def test_no_change_returns_none(self):
        self.mm.process_remote_data({"switches": {"SwA": 2000}})
        # Same data again
        result = self.mm.process_remote_data({"switches": {"SwA": 2000}})
        assert result is None

    def test_is_auto_mode(self):
        self.mm.current_mode = OperationMode.SWA
        assert self.mm.is_auto_mode is True
        self.mm.current_mode = OperationMode.SWB
        assert self.mm.is_auto_mode is True
        self.mm.current_mode = OperationMode.IDLE
        assert self.mm.is_auto_mode is False

    def test_is_emergency(self):
        self.mm.current_mode = OperationMode.SWC
        assert self.mm.is_emergency is True
        self.mm.current_mode = OperationMode.SWA
        assert self.mm.is_emergency is False

    def test_empty_switches(self):
        result = self.mm.process_remote_data({"switches": {}})
        assert result is None  # IDLE -> IDLE, no change


# ==================== VehicleCommands Tests ====================

class TestVehicleCommands:
    def setup_method(self):
        self.commands = []

        async def mock_send(cmd):
            self.commands.append(cmd)

        self.vc = VehicleCommands(send_command=mock_send)

    @pytest.mark.asyncio
    async def test_forward(self):
        await self.vc.forward(60)
        assert self.vc.is_moving is True
        assert self.vc.current_speed == 60
        assert len(self.commands) == 1
        assert self.commands[0]["action"] == "FORWARD"

    @pytest.mark.asyncio
    async def test_stop(self):
        await self.vc.forward(50)
        await self.vc.stop()
        assert self.vc.is_moving is False
        assert self.vc.current_speed == 0

    @pytest.mark.asyncio
    async def test_speed_clamping(self):
        await self.vc.forward(200)
        assert self.vc.current_speed == 100

        await self.vc.forward(-10)
        assert self.vc.current_speed == 0

    @pytest.mark.asyncio
    async def test_no_callback(self):
        vc = VehicleCommands(send_command=None)
        await vc.forward(50)  # Should not raise
        assert vc.is_moving is True


# ==================== NavigationAgent Tests ====================

class TestNavigationAgent:
    @pytest.mark.asyncio
    async def test_initialize(self):
        from agents.navigation.agent import NavigationAgent
        agent = NavigationAgent()
        agent.initialize()
        assert agent._mode_manager is not None
        assert agent._vehicle_cmds is not None

    @pytest.mark.asyncio
    async def test_status(self):
        from agents.navigation.agent import NavigationAgent
        agent = NavigationAgent()
        agent.initialize()
        status = agent.get_status()
        assert status["name"] == "Navigation"
        assert status["current_mode"] == "IDLE"
        assert "is_moving" in status

    @pytest.mark.asyncio
    async def test_struck_weed_tracking(self):
        from agents.navigation.agent import NavigationAgent
        agent = NavigationAgent()
        agent.initialize()

        # Not struck yet
        assert agent.is_weed_already_struck(1) is False

        # Record a struck weed
        agent._struck_weeds[1] = (100.0, 200.0)
        assert agent.is_weed_already_struck(1) is True

        # Nearby weed should also be detected
        assert agent.is_weed_already_struck(2, pixel_x=110, pixel_y=200) is True

        # Far away weed should not be detected
        assert agent.is_weed_already_struck(3, pixel_x=500, pixel_y=500) is False
