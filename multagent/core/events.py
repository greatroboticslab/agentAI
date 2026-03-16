"""
Standard event type definitions.
All events inherit from Event base class.
Domain-specific events are defined in agents/; only generic events here.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Event:
    """Base event class for all inter-agent communication."""
    event_type: str
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: Optional[str] = None

    def __post_init__(self):
        if self.event_id is None:
            self.event_id = f"{self.event_type}_{self.timestamp:.6f}"


# --- Generic System Events ---

class AgentRegisteredEvent(Event):
    """New Agent registered."""
    # data: {"agent_name": str, "agent_profile": str}
    def __init__(self, source: str, agent_name: str, agent_profile: str, **kwargs):
        super().__init__(
            event_type="agent_registered",
            source=source,
            data={"agent_name": agent_name, "agent_profile": agent_profile},
            **kwargs,
        )


class AgentRemovedEvent(Event):
    """Agent removed."""
    # data: {"agent_name": str, "reason": str}
    def __init__(self, source: str, agent_name: str, reason: str = "", **kwargs):
        super().__init__(
            event_type="agent_removed",
            source=source,
            data={"agent_name": agent_name, "reason": reason},
            **kwargs,
        )


class ParamUpdateEvent(Event):
    """Parameter update command (typically issued by Brain)."""
    # data: {"target_agent": str, "params": dict, "reason": str}
    def __init__(self, source: str, target_agent: str, params: dict, reason: str = "", **kwargs):
        super().__init__(
            event_type="param_update",
            source=source,
            data={"target_agent": target_agent, "params": params, "reason": reason},
            **kwargs,
        )


class EdgeConnectedEvent(Event):
    """Edge device connected."""
    # data: {"edge_id": str, "capabilities": list}
    def __init__(self, source: str, edge_id: str, capabilities: list = None, **kwargs):
        super().__init__(
            event_type="edge_connected",
            source=source,
            data={"edge_id": edge_id, "capabilities": capabilities or []},
            **kwargs,
        )


class EdgeDisconnectedEvent(Event):
    """Edge device disconnected."""
    # data: {"edge_id": str, "reason": str}
    def __init__(self, source: str, edge_id: str, reason: str = "", **kwargs):
        super().__init__(
            event_type="edge_disconnected",
            source=source,
            data={"edge_id": edge_id, "reason": reason},
            **kwargs,
        )


class HeartbeatEvent(Event):
    """Heartbeat event."""
    # data: {"edge_id": str, "latency_ms": float}
    def __init__(self, source: str, edge_id: str, latency_ms: float, **kwargs):
        super().__init__(
            event_type="heartbeat",
            source=source,
            data={"edge_id": edge_id, "latency_ms": latency_ms},
            **kwargs,
        )


class UserChatEvent(Event):
    """User sends a message."""
    # data: {"message": str, "user_id": str}
    def __init__(self, source: str, message: str, user_id: str = "anonymous", **kwargs):
        super().__init__(
            event_type="user_chat",
            source=source,
            data={"message": message, "user_id": user_id},
            **kwargs,
        )


class SystemStatusEvent(Event):
    """System status summary (published periodically)."""
    # data: {"agents": dict, "edge_status": dict, "performance": dict}
    def __init__(self, source: str, agents: dict, edge_status: dict, performance: dict, **kwargs):
        super().__init__(
            event_type="system_status",
            source=source,
            data={"agents": agents, "edge_status": edge_status, "performance": performance},
            **kwargs,
        )


# --- Hardware Command (used by EdgeBridge) ---

@dataclass
class HardwareCommand:
    """Command sent to edge device hardware."""
    command_type: str  # e.g. LASER_ON, DAC_POSITION, VEHICLE_FORWARD
    params: Dict[str, Any] = field(default_factory=dict)
    target_edge: str = "default"
    timestamp: float = field(default_factory=time.time)
