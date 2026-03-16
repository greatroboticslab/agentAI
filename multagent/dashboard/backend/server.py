"""
Dashboard FastAPI server.
Hosts WebSocket endpoints, REST API, and serves Vue frontend.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


class DashboardServer:
    """
    FastAPI-based dashboard server.
    Integrates with the EmbodiedTeam to expose real-time data.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        cors_origins: Optional[list] = None,
    ):
        self.host = host
        self.port = port
        self.app = FastAPI(title="EMACF Dashboard", version="1.0.0")

        # CORS
        origins = cors_origins or ["http://localhost:5173", "http://localhost:3000"]
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # References to system components (set by attach_system)
        self._event_bus = None
        self._agent_registry = None
        self._edge_bridge = None
        self._stream_proxy = None

        # WebSocket client sets
        self._ws_clients: dict = {
            "video": set(),
            "agents": set(),
            "messages": set(),
            "brain": set(),
            "chat": set(),
            "performance": set(),
        }

        self._setup_routes()

    def attach_system(self, event_bus, agent_registry, edge_bridge=None) -> None:
        """Attach the running agent system to the dashboard."""
        self._event_bus = event_bus
        self._agent_registry = agent_registry
        self._edge_bridge = edge_bridge

        # Subscribe to events for real-time push
        self._event_bus.subscribe("*", self._on_any_event)
        self._event_bus.subscribe("brain_thought", self._on_brain_thought)
        self._event_bus.subscribe("brain_response", self._on_brain_response)

        # Set up stream proxy
        from dashboard.backend.stream_proxy import StreamProxy
        self._stream_proxy = StreamProxy(self._ws_clients["video"])

        if self._edge_bridge:
            self._edge_bridge.register_video_callback(self._stream_proxy.on_video_frame)

        logger.info("Dashboard attached to system")

    def _setup_routes(self) -> None:
        """Register all routes."""
        from dashboard.backend.api_routes import create_api_router
        from dashboard.backend.ws_handlers import create_ws_router

        api_router = create_api_router(self)
        ws_router = create_ws_router(self)

        self.app.include_router(api_router, prefix="/api")
        self.app.include_router(ws_router)

        # Serve Vue frontend static files (production)
        frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
        if frontend_dist.exists():
            self.app.mount("/", StaticFiles(directory=str(frontend_dist), html=True))

    async def _on_any_event(self, event) -> None:
        """Forward all events to message flow WebSocket clients."""
        import json
        msg = json.dumps({
            "from": event.source,
            "type": event.event_type,
            "data": event.data,
            "timestamp": event.timestamp,
        })
        await self._broadcast("messages", msg)

    async def _on_brain_thought(self, event) -> None:
        import json
        msg = json.dumps({
            "type": "thought",
            "summary": event.data.get("summary", {}),
            "thought": event.data.get("thought", ""),
            "actions": event.data.get("actions", []),
            "timestamp": event.timestamp,
        })
        await self._broadcast("brain", msg)

    async def _on_brain_response(self, event) -> None:
        import json
        msg = json.dumps({
            "type": "response",
            "role": "brain",
            "message": event.data.get("reply", ""),
            "actions": event.data.get("actions_taken", []),
            "timestamp": event.timestamp,
        })
        await self._broadcast("chat", msg)

    async def _broadcast(self, channel: str, message: str) -> None:
        """Broadcast message to all connected WebSocket clients on a channel."""
        dead = set()
        for ws in self._ws_clients.get(channel, set()):
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self._ws_clients[channel] -= dead

    async def start_background_tasks(self) -> None:
        """Start periodic push tasks."""
        asyncio.create_task(self._push_agent_status_loop())
        asyncio.create_task(self._push_performance_loop())

    async def _push_agent_status_loop(self) -> None:
        """Push agent status every 500ms."""
        import json
        import time

        while True:
            if self._agent_registry and self._ws_clients["agents"]:
                msg = json.dumps({
                    "agents": self._agent_registry.list_agents(),
                    "timestamp": time.time(),
                })
                await self._broadcast("agents", msg)

            await asyncio.sleep(0.5)

    async def _push_performance_loop(self) -> None:
        """Push performance metrics every 500ms."""
        import json
        import time

        while True:
            if self._agent_registry and self._ws_clients["performance"]:
                brain = self._agent_registry.get_agent("Brain")
                metrics = {}
                if brain and hasattr(brain, "_optimizer"):
                    metrics = brain._optimizer.get_metrics()

                msg = json.dumps({
                    "metrics": metrics,
                    "timestamp": time.time(),
                })
                await self._broadcast("performance", msg)

            await asyncio.sleep(0.5)

    def run(self) -> None:
        """Run the dashboard server (blocking)."""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)
