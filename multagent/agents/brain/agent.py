"""
BrainAgent: Cognitive decision maker with natural language interface and self-optimization.
Entirely new code (no lasercar.py equivalent).

Subscribes to all key events, aggregates them, periodically calls LLM for analysis,
and publishes parameter adjustment decisions.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import Field

from core.embodied_role import EmbodiedRole
from core.events import Event

from agents.brain.events import BrainResponseEvent, BrainThoughtEvent
from agents.brain.memory import BrainMemory
from agents.brain.optimizer import SelfOptimizer
from agents.brain.prompts.system_prompt import (
    ANALYSIS_PROMPT_TEMPLATE,
    BRAIN_SYSTEM_PROMPT,
    CHAT_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "analysis_interval": 2.0,
    "max_buffer_size": 50,
    "llm_timeout": 30.0,
}


class BrainAgent(EmbodiedRole):
    """
    Brain Agent — the system's cognitive center.
    Aggregates events, calls LLM for analysis, adjusts parameters, handles user chat.
    """

    name: str = "Brain"
    profile: str = "Cognitive decision maker with natural language interface"
    goal: str = "Optimize system performance through intelligent decision making"
    capabilities: list = ["cognition", "optimization", "user_interface"]
    subscribed_events: list = [
        "weed_detection", "firing_complete", "firing_started",
        "mode_change", "user_chat", "agent_registered",
        "edge_disconnected",
    ]
    params: Dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_PARAMS))

    # Sub-components (set in initialize)
    _memory: Optional[BrainMemory] = None
    _optimizer: Optional[SelfOptimizer] = None

    # Event buffer
    _event_buffer: list = []
    _last_analysis_time: float = 0.0

    # LLM interface (injected)
    _llm_call: Optional[Any] = None  # async callable(system_prompt, user_prompt) -> str

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._event_buffer = []
        self._last_analysis_time = time.time()

    def initialize(self, llm_call: Optional[Any] = None) -> None:
        """Initialize sub-components."""
        self._memory = BrainMemory()
        self._optimizer = SelfOptimizer()
        self._llm_call = llm_call
        logger.info("BrainAgent initialized")

    async def run_loop(self) -> None:
        if self._memory is None:
            self.initialize()
        await super().run_loop()

    async def on_tick(self) -> None:
        """Periodic analysis of buffered events."""
        now = time.time()
        interval = self.params.get("analysis_interval", 2.0)

        if now - self._last_analysis_time >= interval and self._event_buffer:
            await self._analyze_events()
            self._last_analysis_time = now

    async def on_event(self, event: Event) -> None:
        """Buffer events for periodic analysis; handle high-priority immediately."""
        if event.event_type == "user_chat":
            await self._on_user_chat(event)
            return

        if event.event_type == "edge_disconnected":
            await self._on_emergency(event)
            return

        if event.event_type == "param_update":
            if event.data.get("target_agent") == self.name:
                self.update_params(event.data["params"])
            return

        # Buffer for periodic analysis
        max_buf = self.params.get("max_buffer_size", 50)
        if len(self._event_buffer) < max_buf:
            self._event_buffer.append(event)

    async def _analyze_events(self) -> None:
        """Periodic event analysis — core cognitive function."""
        events = self._event_buffer.copy()
        self._event_buffer.clear()

        # Update optimizer with event data
        self._optimizer.update_from_events(events)

        # Build event summary
        summary = self._build_event_summary(events)
        system_status = self._get_system_status()
        metrics = self._optimizer.get_metrics()
        memory_context = self._memory.build_context()

        # Call LLM if available
        decision = {"reasoning": "No LLM configured", "actions": []}
        if self._llm_call:
            prompt = ANALYSIS_PROMPT_TEMPLATE.format(
                system_status=json.dumps(system_status, indent=2),
                time_window=f"{self.params.get('analysis_interval', 2.0)}s",
                event_summary=json.dumps(summary, indent=2),
                metrics=json.dumps(metrics, indent=2),
                memory_context=memory_context,
            )
            try:
                response = await asyncio.wait_for(
                    self._llm_call(BRAIN_SYSTEM_PROMPT, prompt),
                    timeout=self.params.get("llm_timeout", 30.0),
                )
                decision = self._parse_llm_response(response)
            except asyncio.TimeoutError:
                logger.warning("LLM analysis timed out")
            except Exception as e:
                logger.error(f"LLM analysis error: {e}")
        else:
            # Rule-based fallback: use optimizer suggestions
            suggestions = self._optimizer.get_suggestions()
            if suggestions:
                decision = {
                    "reasoning": "Rule-based optimization",
                    "actions": [
                        {
                            "type": "ADJUST_PARAM",
                            "target": s["target"],
                            "params": {s["param"]: s["direction"]},
                            "reason": s["reason"],
                        }
                        for s in suggestions
                    ],
                }

        # Execute decisions
        await self._execute_decision(decision)

        # Record to memory
        self._memory.record(summary, decision)

        # Publish thought event
        if self._event_bus:
            await self._event_bus.publish(BrainThoughtEvent(
                source=self.name,
                summary=summary,
                thought=decision.get("reasoning", ""),
                actions=decision.get("actions", []),
            ))

    async def _on_user_chat(self, event: Event) -> None:
        """Handle user chat — immediate LLM response."""
        user_message = event.data.get("message", "")
        system_status = self._get_system_status()
        memory_context = self._memory.build_context()

        reply = f"Received: {user_message}"
        actions = []

        if self._llm_call:
            prompt = CHAT_PROMPT_TEMPLATE.format(
                system_status=json.dumps(system_status, indent=2),
                memory_context=memory_context,
                user_message=user_message,
            )
            try:
                response = await asyncio.wait_for(
                    self._llm_call(BRAIN_SYSTEM_PROMPT, prompt),
                    timeout=self.params.get("llm_timeout", 30.0),
                )
                parsed = self._parse_llm_response(response)
                reply = parsed.get("reasoning", reply)
                actions = parsed.get("actions", [])
                await self._execute_decision(parsed)
            except Exception as e:
                logger.error(f"LLM chat error: {e}")
                reply = f"Error processing your message: {e}"

        if self._event_bus:
            await self._event_bus.publish(BrainResponseEvent(
                source=self.name,
                reply=reply,
                actions_taken=actions,
            ))

    async def _on_emergency(self, event: Event) -> None:
        """Handle emergency events — immediate response."""
        logger.warning(f"Emergency event: {event.event_type}")
        # Publish laser OFF command
        if self._event_bus:
            from core.events import Event as BaseEvent
            await self._event_bus.publish(BaseEvent(
                event_type="param_update",
                source=self.name,
                data={
                    "target_agent": "Targeting",
                    "params": {"targeting_enabled": False},
                    "reason": f"Emergency: {event.event_type}",
                },
            ))

    async def _execute_decision(self, decision: dict) -> None:
        """Execute LLM's decisions by publishing events."""
        for action in decision.get("actions", []):
            action_type = action.get("type", "")

            if action_type == "ADJUST_PARAM" and self._event_bus:
                await self._event_bus.publish(Event(
                    event_type="param_update",
                    source=self.name,
                    data={
                        "target_agent": action.get("target", ""),
                        "params": action.get("params", {}),
                        "reason": action.get("reason", ""),
                    },
                ))
                self._memory.record_param_change(
                    agent=action.get("target", ""),
                    params=action.get("params", {}),
                    reason=action.get("reason", ""),
                )
            elif action_type == "CHANGE_MODE" and self._event_bus:
                await self._event_bus.publish(Event(
                    event_type="mode_command",
                    source=self.name,
                    data={"mode": action.get("mode", "IDLE")},
                ))

    def _build_event_summary(self, events: list) -> dict:
        """Aggregate raw events into LLM-friendly summary."""
        summary = {
            "time_window": f"last {self.params.get('analysis_interval', 2.0)}s",
            "total_events": len(events),
            "detection_count": 0,
            "firings_started": 0,
            "firings_completed": 0,
            "mode_changes": [],
            "avg_yolo_latency_ms": 0,
        }

        latencies = []
        for e in events:
            if e.event_type == "weed_detection":
                summary["detection_count"] += 1
                lat = e.data.get("yolo_latency_ms", 0)
                if lat:
                    latencies.append(lat)
            elif e.event_type == "firing_started":
                summary["firings_started"] += 1
            elif e.event_type == "firing_complete":
                summary["firings_completed"] += 1
            elif e.event_type == "mode_change":
                summary["mode_changes"].append(e.data)

        if latencies:
            summary["avg_yolo_latency_ms"] = round(sum(latencies) / len(latencies), 1)

        return summary

    def _get_system_status(self) -> dict:
        """Get snapshot of entire system state."""
        status = {"agents": [], "performance": self._optimizer.get_metrics()}

        if self._agent_registry:
            status["agents"] = self._agent_registry.list_agents()

        return status

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM JSON response."""
        try:
            # Try to extract JSON from response
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return {"reasoning": response, "actions": []}

    def get_status(self) -> dict:
        status = super().get_status()
        status.update({
            "buffered_events": len(self._event_buffer),
            "short_term_memory": self._memory.short_term_count if self._memory else 0,
            "long_term_memory": self._memory.long_term_count if self._memory else 0,
            "has_llm": self._llm_call is not None,
            "performance": self._optimizer.get_metrics() if self._optimizer else {},
        })
        return status
