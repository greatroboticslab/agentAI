"""
REST API endpoints for the Dashboard.
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ParamUpdateRequest(BaseModel):
    params: Dict[str, Any]


class ChatRequest(BaseModel):
    message: str


class ModeRequest(BaseModel):
    mode: str


def create_api_router(server) -> APIRouter:
    """Create REST API routes bound to the DashboardServer instance."""
    router = APIRouter()

    @router.get("/agents")
    async def get_agents():
        """Get all agent statuses."""
        if not server._agent_registry:
            return {"agents": []}

        return {"agents": server._agent_registry.list_agents()}

    @router.post("/agents/{name}/params")
    async def update_agent_params(name: str, req: ParamUpdateRequest):
        """Update parameters for a specific agent."""
        if not server._agent_registry:
            raise HTTPException(status_code=503, detail="System not ready")

        agent = server._agent_registry.get_agent(name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")

        agent.update_params(req.params)

        # Also publish as event so Brain can track it
        if server._event_bus:
            from core.events import Event
            await server._event_bus.publish(Event(
                event_type="param_update",
                source="dashboard",
                data={"target_agent": name, "params": req.params, "reason": "Manual dashboard update"},
            ))

        return {"status": "ok", "agent": name, "updated_params": req.params}

    @router.get("/performance")
    async def get_performance():
        """Get performance metrics."""
        if not server._agent_registry:
            return {"metrics": {}}

        brain = server._agent_registry.get_agent("Brain")
        if brain and hasattr(brain, "_optimizer"):
            return {"metrics": brain._optimizer.get_metrics()}
        return {"metrics": {}}

    @router.get("/brain/memory")
    async def get_brain_memory():
        """Get Brain's memory contents."""
        if not server._agent_registry:
            return {"short_term": [], "long_term": [], "param_history": []}

        brain = server._agent_registry.get_agent("Brain")
        if not brain or not hasattr(brain, "_memory"):
            return {"short_term": [], "long_term": [], "param_history": []}

        return {
            "short_term": brain._memory.get_recent(20),
            "long_term": brain._memory.get_long_term(10),
            "param_history": brain._memory.get_param_history(count=20),
        }

    @router.post("/brain/chat")
    async def brain_chat(req: ChatRequest):
        """Send a chat message to Brain (REST fallback)."""
        if not server._event_bus:
            raise HTTPException(status_code=503, detail="System not ready")

        from agents.brain.events import UserChatEvent
        await server._event_bus.publish(UserChatEvent(
            source="dashboard",
            message=req.message,
        ))
        return {"status": "sent", "message": req.message}

    @router.get("/system/status")
    async def get_system_status():
        """Get overall system status."""
        status = {
            "system_name": "LaserCar EMACF",
            "version": "1.0.0",
            "timestamp": time.time(),
            "agents_running": 0,
            "edge_connected": False,
        }

        if server._agent_registry:
            agents = server._agent_registry.list_agents()
            status["agents_running"] = len(agents)

        if server._edge_bridge:
            status["edge_connected"] = server._edge_bridge.is_connected

        return status

    @router.post("/system/mode")
    async def switch_mode(req: ModeRequest):
        """Switch operation mode."""
        if not server._event_bus:
            raise HTTPException(status_code=503, detail="System not ready")

        from core.events import Event
        await server._event_bus.publish(Event(
            event_type="mode_command",
            source="dashboard",
            data={"mode": req.mode},
        ))
        return {"status": "ok", "mode": req.mode}

    return router
