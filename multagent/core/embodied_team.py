"""
EmbodiedTeam: event-driven Team that orchestrates all agents.
Modification from MetaGPT Team: from turn-based to event-driven + parallel execution.
"""

import asyncio
import logging
from typing import List, Optional

from core.agent_registry import AgentRegistry
from core.config_manager import ConfigManager
from core.edge_bridge import EdgeBridge
from core.embodied_role import EmbodiedRole
from core.event_bus import EventBus

logger = logging.getLogger(__name__)


class EmbodiedTeam:
    """
    The top-level orchestrator for the embodied multi-agent system.
    Manages EventBus, EdgeBridge, AgentRegistry, and all agent lifecycles.
    """

    def __init__(self, config_path: str = "config"):
        self.config = ConfigManager(config_path)
        self.event_bus = EventBus(
            history_size=self.config.get("default", "event_bus.history_size", 1000)
        )
        self.edge_bridge = EdgeBridge(
            self.event_bus,
            heartbeat_timeout_ms=self.config.get("default", "edge_bridge.heartbeat_timeout_ms", 500),
        )
        self.agent_registry = AgentRegistry(self.event_bus, self.edge_bridge)
        self._agent_tasks: List[asyncio.Task] = []
        self._running = False

    async def startup(self, start_edge_bridge: bool = True) -> None:
        """
        Start the entire system:
        1. Load configuration
        2. Start EdgeBridge (wait for edge device connections)
        3. Start all registered Agent run_loops
        """
        logger.info("EmbodiedTeam starting up...")
        self._running = True

        # Start EdgeBridge
        if start_edge_bridge:
            host = self.config.get("default", "edge_bridge.host", "0.0.0.0")
            port = self.config.get("default", "edge_bridge.port", 8765)
            await self.edge_bridge.start(host, port)

        # Start all already-registered agents
        for name in self.agent_registry.agent_names:
            agent = self.agent_registry.get_agent(name)
            if agent:
                task = asyncio.create_task(agent.run_loop())
                agent._task = task
                self._agent_tasks.append(task)

        logger.info(f"EmbodiedTeam started with {self.agent_registry.agent_count} agents")

    async def add_agent(self, agent: EmbodiedRole) -> None:
        """Hot-add Agent at runtime."""
        await self.agent_registry.register(agent)
        task = asyncio.create_task(agent.run_loop())
        agent._task = task
        self._agent_tasks.append(task)
        logger.info(f"Hot-added agent: {agent.name}")

    async def remove_agent(self, agent_name: str) -> None:
        """Hot-remove Agent at runtime."""
        await self.agent_registry.unregister(agent_name)

    async def shutdown(self) -> None:
        """Safely shut down all systems."""
        logger.info("EmbodiedTeam shutting down...")
        self._running = False

        # Stop all agents
        for name in list(self.agent_registry.agent_names):
            await self.agent_registry.unregister(name, reason="system_shutdown")

        # Cancel remaining tasks
        for task in self._agent_tasks:
            if not task.done():
                task.cancel()
        if self._agent_tasks:
            await asyncio.gather(*self._agent_tasks, return_exceptions=True)
        self._agent_tasks.clear()

        # Stop EdgeBridge
        await self.edge_bridge.stop()

        logger.info("EmbodiedTeam shutdown complete")

    @property
    def is_running(self) -> bool:
        return self._running
