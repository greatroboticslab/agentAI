"""Tests for EventBus."""

import asyncio
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.event_bus import EventBus
from core.events import Event


@pytest.fixture
def event_bus():
    return EventBus(history_size=100)


@pytest.mark.asyncio
async def test_publish_subscribe(event_bus):
    """Test basic publish/subscribe."""
    received = []

    async def handler(event: Event):
        received.append(event)

    event_bus.subscribe("test_event", handler)
    await event_bus.publish(Event(event_type="test_event", source="test", data={"msg": "hello"}))

    # Give async tasks time to complete
    await asyncio.sleep(0.05)
    assert len(received) == 1
    assert received[0].data["msg"] == "hello"


@pytest.mark.asyncio
async def test_unsubscribe(event_bus):
    """Test unsubscribe stops delivery."""
    received = []

    async def handler(event: Event):
        received.append(event)

    event_bus.subscribe("test_event", handler)
    event_bus.unsubscribe("test_event", handler)
    await event_bus.publish(Event(event_type="test_event", source="test"))

    await asyncio.sleep(0.05)
    assert len(received) == 0


@pytest.mark.asyncio
async def test_multiple_subscribers(event_bus):
    """Test multiple subscribers receive the same event."""
    results = {"a": [], "b": []}

    async def handler_a(event):
        results["a"].append(event)

    async def handler_b(event):
        results["b"].append(event)

    event_bus.subscribe("test_event", handler_a)
    event_bus.subscribe("test_event", handler_b)
    await event_bus.publish(Event(event_type="test_event", source="test"))

    await asyncio.sleep(0.05)
    assert len(results["a"]) == 1
    assert len(results["b"]) == 1


@pytest.mark.asyncio
async def test_event_history(event_bus):
    """Test event history recording."""
    for i in range(5):
        await event_bus.publish(Event(event_type="log", source="test", data={"i": i}))

    history = event_bus.get_recent_events("log", count=3)
    assert len(history) == 3
    assert history[-1].data["i"] == 4


@pytest.mark.asyncio
async def test_event_filtering(event_bus):
    """Test get_recent_events filters by type."""
    await event_bus.publish(Event(event_type="type_a", source="test"))
    await event_bus.publish(Event(event_type="type_b", source="test"))
    await event_bus.publish(Event(event_type="type_a", source="test"))

    a_events = event_bus.get_recent_events("type_a")
    b_events = event_bus.get_recent_events("type_b")
    assert len(a_events) == 2
    assert len(b_events) == 1


@pytest.mark.asyncio
async def test_dashboard_callback(event_bus):
    """Test dashboard callback receives all events."""
    dashboard_events = []

    async def dashboard_cb(event):
        dashboard_events.append(event)

    event_bus.set_dashboard_callback(dashboard_cb)
    await event_bus.publish(Event(event_type="any", source="test"))

    await asyncio.sleep(0.05)
    assert len(dashboard_events) == 1
