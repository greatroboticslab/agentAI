"""
EmbodiedRole: extends MetaGPT Role with real-time embodied capabilities.

New capabilities:
  1. Event-driven response (on_event)
  2. Hardware I/O interface (send_hardware_command)
  3. Latency awareness (latency tracking)
  4. Periodic heartbeat (heartbeat)
  5. Runtime parameter dynamic adjustment (update_params)
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Set

from pydantic import Field

from metagpt.roles import Role

from core.events import Event, HardwareCommand

logger = logging.getLogger(__name__)


class EmbodiedRole(Role):
    """
    Base class for all embodied agents.
    Extends MetaGPT Role with event-driven, real-time capabilities.
    """

    # Pydantic fields
    params: Dict[str, Any] = Field(default_factory=dict)
    subscribed_events: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)

    # Non-serialized runtime references (set by AgentRegistry during registration)
    _event_bus: Any = None
    _edge_bridge: Any = None
    _agent_registry: Any = None
    _latency_tracker: Dict[str, Deque] = {}
    _running: bool = False
    _task: Optional[asyncio.Task] = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._latency_tracker = defaultdict(lambda: deque(maxlen=100))
        self._running = False

    # --- System references (set by AgentRegistry) ---

    def set_event_bus(self, event_bus) -> None:
        self._event_bus = event_bus

    def set_edge_bridge(self, edge_bridge) -> None:
        self._edge_bridge = edge_bridge

    def set_agent_registry(self, registry) -> None:
        self._agent_registry = registry

    @property
    def event_bus(self):
        return self._event_bus

    @property
    def edge_bridge(self):
        return self._edge_bridge

    @property
    def agent_registry(self):
        return self._agent_registry

    # --- Event-driven methods ---

    async def on_event(self, event: Event) -> None:
        """
        Event-driven response (core extension).
        Unlike MetaGPT's turn-based approach, this is real-time.
        Subclasses must implement this method.
        """
        raise NotImplementedError(f"{self.name} must implement on_event()")

    async def run_loop(self) -> None:
        """
        Agent main loop.
        Subscribes to events on event_bus, dispatches to on_event.
        Also runs periodic tasks via on_tick().
        """
        self._running = True
        logger.info(f"Agent {self.name} run_loop started")

        # Subscribe to declared event types
        if self._event_bus:
            for event_type in self.subscribed_events:
                self._event_bus.subscribe(event_type, self._handle_event)

        try:
            while self._running:
                await self.on_tick()
                await asyncio.sleep(0.01)  # 10ms tick
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            logger.info(f"Agent {self.name} run_loop stopped")

    async def _handle_event(self, event: Event) -> None:
        """Internal event handler with error catching."""
        try:
            start = time.perf_counter()
            await self.on_event(event)
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.report_latency(f"event_{event.event_type}", elapsed_ms)
        except Exception as e:
            logger.error(f"Agent {self.name} error handling {event.event_type}: {e}")

    async def on_tick(self) -> None:
        """
        Periodic tick callback. Override in subclasses for periodic work.
        Default: no-op. E.g., PerceptionAgent runs detection every tick.
        """
        pass

    async def stop(self) -> None:
        """Stop the agent's run loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    # --- Hardware I/O ---

    async def send_hardware_command(self, command: HardwareCommand) -> bool:
        """Send hardware command via EdgeBridge."""
        if self._edge_bridge:
            return await self._edge_bridge.send_command(command)
        logger.warning(f"Agent {self.name}: no EdgeBridge connected")
        return False

    # --- Parameter management ---

    def update_params(self, new_params: dict) -> None:
        """Brain dynamically adjusts this Agent's parameters."""
        old_params = dict(self.params)
        self.params.update(new_params)
        changed = {k: v for k, v in new_params.items() if old_params.get(k) != v}
        if changed:
            logger.info(f"Agent {self.name} params updated: {changed}")
            self.on_params_updated(changed)

    def on_params_updated(self, changed_params: dict) -> None:
        """Parameter change callback, subclasses can override."""
        pass

    # --- Latency tracking ---

    def report_latency(self, metric: str, value_ms: float) -> None:
        """Record latency metrics for Brain analysis."""
        self._latency_tracker[metric].append(value_ms)

    def get_avg_latency(self, metric: str) -> float:
        """Get average latency for a metric."""
        values = self._latency_tracker.get(metric, deque())
        return sum(values) / len(values) if values else 0.0

    # --- Status ---

    def get_status(self) -> dict:
        """Return current Agent status summary (for Dashboard display)."""
        return {
            "name": self.name,
            "profile": self.profile,
            "running": self._running,
            "params": dict(self.params),
            "capabilities": list(self.capabilities),
            "latency": {
                k: round(sum(v) / len(v), 2) if v else 0.0
                for k, v in self._latency_tracker.items()
            },
        }
