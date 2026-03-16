"""
Agent registry: manages the lifecycle of all active Agents.
Supports:
  - Dynamic registration/deregistration
  - Agent discovery (by name, by capability)
  - Health checks
  - Hot-plugging (add/remove Agents at runtime)
"""

import logging
from typing import Dict, List, Optional

from core.events import AgentRegisteredEvent, AgentRemovedEvent

logger = logging.getLogger(__name__)


class AgentRegistry:
    def __init__(self, event_bus, edge_bridge=None):
        self._agents: Dict[str, "EmbodiedRole"] = {}
        self._event_bus = event_bus
        self._edge_bridge = edge_bridge

    def set_edge_bridge(self, edge_bridge) -> None:
        """Set edge bridge reference (may be set after init)."""
        self._edge_bridge = edge_bridge

    async def register(self, agent) -> None:
        """
        Register a new Agent.
        Auto-connects event_bus, edge_bridge.
        Publishes AgentRegisteredEvent to notify all Agents (especially Brain).
        """
        name = agent.name
        if name in self._agents:
            logger.warning(f"Agent {name} already registered, replacing")

        # Wire up system references
        agent.set_event_bus(self._event_bus)
        if self._edge_bridge:
            agent.set_edge_bridge(self._edge_bridge)
        agent.set_agent_registry(self)

        self._agents[name] = agent
        logger.info(f"Agent registered: {name} ({agent.profile})")

        await self._event_bus.publish(
            AgentRegisteredEvent(
                source="AgentRegistry",
                agent_name=name,
                agent_profile=agent.profile,
            )
        )

    async def unregister(self, agent_name: str, reason: str = "") -> None:
        """Unregister an Agent."""
        agent = self._agents.pop(agent_name, None)
        if agent is None:
            logger.warning(f"Agent {agent_name} not found in registry")
            return

        # Stop the agent
        await agent.stop()

        # Unsubscribe from all events
        for event_type in agent.subscribed_events:
            self._event_bus.unsubscribe(event_type, agent._handle_event)

        logger.info(f"Agent unregistered: {agent_name} ({reason})")

        await self._event_bus.publish(
            AgentRemovedEvent(
                source="AgentRegistry",
                agent_name=agent_name,
                reason=reason,
            )
        )

    def get_agent(self, name: str) -> Optional["EmbodiedRole"]:
        """Get Agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> List[dict]:
        """List all active Agents and their status."""
        return [agent.get_status() for agent in self._agents.values()]

    def discover(self, capability: str) -> List["EmbodiedRole"]:
        """Discover Agents by capability (e.g., find all video-processing Agents)."""
        return [
            agent for agent in self._agents.values()
            if capability in agent.capabilities
        ]

    async def health_check(self) -> dict:
        """Check health status of all Agents."""
        return {
            name: {
                "running": agent._running,
                "status": agent.get_status(),
            }
            for name, agent in self._agents.items()
        }

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    @property
    def agent_names(self) -> List[str]:
        return list(self._agents.keys())
