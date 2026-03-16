"""Tests for AgentRegistry."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.event_bus import EventBus
from core.agent_registry import AgentRegistry
from core.embodied_role import EmbodiedRole
from core.events import Event


class DummyAgent(EmbodiedRole):
    """A minimal agent for testing."""
    name: str = "TestAgent"
    profile: str = "Test profile"
    capabilities: list = ["testing"]
    subscribed_events: list = ["test_event"]

    async def on_event(self, event: Event) -> None:
        pass


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def registry(event_bus):
    return AgentRegistry(event_bus)


@pytest.mark.asyncio
async def test_register_agent(registry):
    agent = DummyAgent()
    await registry.register(agent)

    assert registry.agent_count == 1
    assert "TestAgent" in registry.agent_names
    assert agent.event_bus is not None


@pytest.mark.asyncio
async def test_unregister_agent(registry):
    agent = DummyAgent()
    await registry.register(agent)
    await registry.unregister("TestAgent", reason="test")

    assert registry.agent_count == 0


@pytest.mark.asyncio
async def test_get_agent(registry):
    agent = DummyAgent()
    await registry.register(agent)

    found = registry.get_agent("TestAgent")
    assert found is agent
    assert registry.get_agent("NonExistent") is None


@pytest.mark.asyncio
async def test_discover_by_capability(registry):
    agent = DummyAgent()
    await registry.register(agent)

    found = registry.discover("testing")
    assert len(found) == 1
    assert found[0].name == "TestAgent"

    not_found = registry.discover("flying")
    assert len(not_found) == 0


@pytest.mark.asyncio
async def test_list_agents(registry):
    agent = DummyAgent()
    await registry.register(agent)

    agent_list = registry.list_agents()
    assert len(agent_list) == 1
    assert agent_list[0]["name"] == "TestAgent"


@pytest.mark.asyncio
async def test_health_check(registry):
    agent = DummyAgent()
    await registry.register(agent)

    health = await registry.health_check()
    assert "TestAgent" in health
    assert health["TestAgent"]["running"] is False  # not started yet


@pytest.mark.asyncio
async def test_register_publishes_event(event_bus, registry):
    """Registering an agent should publish AgentRegisteredEvent."""
    received = []
    event_bus.subscribe("agent_registered", lambda e: received.append(e))

    agent = DummyAgent()
    await registry.register(agent)

    assert len(received) == 1
    assert received[0].data["agent_name"] == "TestAgent"
