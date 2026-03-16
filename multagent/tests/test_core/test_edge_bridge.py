"""Tests for EdgeBridge (unit tests, no real WebSocket)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.event_bus import EventBus
from core.edge_bridge import EdgeBridge
from core.events import HardwareCommand
from core.safety import SafetyPolicy


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def bridge(event_bus):
    return EdgeBridge(event_bus, heartbeat_timeout_ms=500)


def test_bridge_init(bridge):
    """Test EdgeBridge initializes correctly."""
    assert bridge.connected_edges == []
    assert bridge.get_connection_status() == {}


def test_video_callback_registration(bridge):
    """Test video callback registration."""
    cb = lambda edge_id, data: None
    bridge.register_video_callback(cb)
    assert len(bridge._video_callbacks) == 1

    bridge.unregister_video_callback(cb)
    assert len(bridge._video_callbacks) == 0


def test_safety_policy():
    """Test safety policy validation."""
    policy = SafetyPolicy()

    # Normal command passes
    cmd = HardwareCommand(command_type="LASER_ON")
    assert policy.validate_command(cmd) is True

    # Excessive laser power blocked
    cmd = HardwareCommand(command_type="LASER_POWER", params={"value": 1.5})
    assert policy.validate_command(cmd) is False

    # Blocked command type
    policy.block_command("LASER_ON")
    cmd = HardwareCommand(command_type="LASER_ON")
    assert policy.validate_command(cmd) is False

    # Unblock
    policy.unblock_command("LASER_ON")
    assert policy.validate_command(cmd) is True


@pytest.mark.asyncio
async def test_send_command_no_connection(bridge):
    """Test send_command with no edge device connected."""
    cmd = HardwareCommand(command_type="LASER_ON")
    result = await bridge.send_command(cmd)
    assert result is False
