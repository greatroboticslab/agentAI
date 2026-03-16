"""Tests for Brain Agent sub-components."""

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.brain.memory import BrainMemory
from agents.brain.optimizer import SelfOptimizer, PerformanceMetrics
from core.events import Event


# ==================== BrainMemory Tests ====================

class TestBrainMemory:
    def setup_method(self):
        self.memory = BrainMemory(short_term_capacity=10, long_term_capacity=5)

    def test_record_and_retrieve(self):
        self.memory.record({"detection_count": 5}, {"reasoning": "ok"})
        recent = self.memory.get_recent(10)
        assert len(recent) == 1
        assert recent[0]["summary"]["detection_count"] == 5

    def test_short_term_capacity(self):
        for i in range(15):
            self.memory.record({"count": i}, {})
        assert self.memory.short_term_count == 10

    def test_promote_to_long_term(self):
        entry = {"insight": "high latency pattern"}
        self.memory.promote_to_long_term(entry)
        lt = self.memory.get_long_term()
        assert len(lt) == 1
        assert lt[0]["insight"] == "high latency pattern"

    def test_param_history(self):
        self.memory.record_param_change("Perception", {"yolo_confidence": 0.5}, "test")
        history = self.memory.get_param_history("Perception")
        assert len(history) == 1
        assert history[0]["params"]["yolo_confidence"] == 0.5

    def test_build_context(self):
        self.memory.record({"detection_count": 3, "firings_completed": 1}, {"reasoning": "ok"})
        ctx = self.memory.build_context()
        assert "Detections: 3" in ctx

    def test_empty_context(self):
        assert self.memory.build_context() == "No recent activity."


# ==================== PerformanceMetrics Tests ====================

class TestPerformanceMetrics:
    def setup_method(self):
        self.metrics = PerformanceMetrics(window_size=10)

    def test_record_firing(self):
        self.metrics.record_firing(1, 5.0, "static")
        assert self.metrics.total_firings == 1
        assert self.metrics.avg_firing_duration == 5.0

    def test_record_latency(self):
        self.metrics.record_latency(50.0)
        self.metrics.record_latency(100.0)
        assert self.metrics.avg_latency == 75.0

    def test_summary(self):
        self.metrics.record_firing(1, 10.0, "trajectory")
        self.metrics.record_latency(80.0)
        self.metrics.record_detections(5)
        summary = self.metrics.get_summary()
        assert summary["total_firings"] == 1
        assert summary["avg_yolo_latency_ms"] == 80.0
        assert summary["avg_detections_per_frame"] == 5.0


# ==================== SelfOptimizer Tests ====================

class TestSelfOptimizer:
    def setup_method(self):
        self.opt = SelfOptimizer()

    def test_update_from_events(self):
        events = [
            Event(event_type="weed_detection", source="test",
                  data={"weeds": [1, 2, 3], "yolo_latency_ms": 80}),
            Event(event_type="firing_complete", source="test",
                  data={"weed_id": 1, "duration": 10, "mode": "static"}),
        ]
        self.opt.update_from_events(events)
        assert self.opt.metrics.total_firings == 1
        assert self.opt.metrics.avg_latency == 80.0

    def test_suggestions_high_latency(self):
        for _ in range(5):
            self.opt.metrics.record_latency(150.0)
        suggestions = self.opt.get_suggestions()
        assert len(suggestions) > 0
        assert suggestions[0]["target"] == "Perception"

    def test_no_suggestions_when_ok(self):
        self.opt.metrics.record_latency(30.0)
        self.opt.metrics.record_detections(3)
        suggestions = self.opt.get_suggestions()
        # Low latency + low detections = no suggestions
        assert len(suggestions) == 0


# ==================== BrainAgent Tests ====================

class TestBrainAgent:
    @pytest.mark.asyncio
    async def test_initialize(self):
        from agents.brain.agent import BrainAgent
        agent = BrainAgent()
        agent.initialize()
        assert agent._memory is not None
        assert agent._optimizer is not None

    @pytest.mark.asyncio
    async def test_status(self):
        from agents.brain.agent import BrainAgent
        agent = BrainAgent()
        agent.initialize()
        status = agent.get_status()
        assert status["name"] == "Brain"
        assert "buffered_events" in status
        assert "has_llm" in status
        assert status["has_llm"] is False

    @pytest.mark.asyncio
    async def test_event_buffering(self):
        from agents.brain.agent import BrainAgent
        agent = BrainAgent()
        agent.initialize()

        event = Event(event_type="weed_detection", source="test",
                      data={"weeds": [], "yolo_latency_ms": 50})
        await agent.on_event(event)
        assert len(agent._event_buffer) == 1

    @pytest.mark.asyncio
    async def test_build_event_summary(self):
        from agents.brain.agent import BrainAgent
        agent = BrainAgent()
        agent.initialize()

        events = [
            Event(event_type="weed_detection", source="test",
                  data={"weeds": [1], "yolo_latency_ms": 60}),
            Event(event_type="firing_started", source="test",
                  data={"weed_id": 1}),
            Event(event_type="firing_complete", source="test",
                  data={"weed_id": 1, "duration": 5}),
        ]
        summary = agent._build_event_summary(events)
        assert summary["detection_count"] == 1
        assert summary["firings_started"] == 1
        assert summary["firings_completed"] == 1
        assert summary["avg_yolo_latency_ms"] == 60.0

    @pytest.mark.asyncio
    async def test_analyze_without_llm(self):
        """Analysis should work with rule-based fallback when no LLM."""
        from agents.brain.agent import BrainAgent
        from core.event_bus import EventBus

        event_bus = EventBus()
        thoughts = []
        event_bus.subscribe("brain_thought", lambda e: thoughts.append(e))

        agent = BrainAgent()
        agent.initialize()
        agent.set_event_bus(event_bus)

        # Buffer some events
        for _ in range(3):
            agent._event_buffer.append(
                Event(event_type="weed_detection", source="test",
                      data={"weeds": [1], "yolo_latency_ms": 50})
            )

        # Trigger analysis
        await agent._analyze_events()

        assert len(thoughts) == 1
        assert thoughts[0].event_type == "brain_thought"
