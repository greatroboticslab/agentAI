"""
Brain memory: short-term event buffer + long-term learned patterns.
"""

import time
from collections import deque
from typing import Any, Dict, List, Optional


class BrainMemory:
    """Manages Brain's short-term and long-term memory."""

    def __init__(self, short_term_capacity: int = 100, long_term_capacity: int = 500):
        self._short_term: deque = deque(maxlen=short_term_capacity)
        self._long_term: deque = deque(maxlen=long_term_capacity)
        self._param_history: List[dict] = []  # Track parameter changes

    def record(self, summary: dict, decision: dict) -> None:
        """Record an analysis cycle to short-term memory."""
        entry = {
            "timestamp": time.time(),
            "summary": summary,
            "decision": decision,
        }
        self._short_term.append(entry)

    def record_param_change(
        self, agent: str, params: dict, reason: str, result: str = ""
    ) -> None:
        """Record a parameter change for optimization tracking."""
        self._param_history.append({
            "timestamp": time.time(),
            "agent": agent,
            "params": params,
            "reason": reason,
            "result": result,
        })

    def promote_to_long_term(self, entry: dict) -> None:
        """Move an important insight to long-term memory."""
        entry["promoted_at"] = time.time()
        self._long_term.append(entry)

    def get_recent(self, count: int = 20) -> List[dict]:
        """Get recent short-term memory entries."""
        items = list(self._short_term)
        return items[-count:]

    def get_long_term(self, count: int = 10) -> List[dict]:
        """Get long-term memory entries."""
        items = list(self._long_term)
        return items[-count:]

    def get_param_history(self, agent: Optional[str] = None, count: int = 10) -> List[dict]:
        """Get parameter change history, optionally filtered by agent."""
        history = self._param_history
        if agent:
            history = [h for h in history if h["agent"] == agent]
        return history[-count:]

    def build_context(self, max_entries: int = 20) -> str:
        """Build a text context from recent memory for LLM prompts."""
        recent = self.get_recent(max_entries)
        if not recent:
            return "No recent activity."

        lines = []
        for entry in recent:
            ts = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
            summary = entry.get("summary", {})
            decision = entry.get("decision", {})
            lines.append(
                f"[{ts}] Detections: {summary.get('detection_count', 0)}, "
                f"Firings: {summary.get('firings_completed', 0)}, "
                f"Decision: {decision.get('reasoning', 'N/A')[:80]}"
            )
        return "\n".join(lines)

    @property
    def short_term_count(self) -> int:
        return len(self._short_term)

    @property
    def long_term_count(self) -> int:
        return len(self._long_term)
