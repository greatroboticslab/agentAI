"""Tests for Dashboard backend components."""

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.backend.server import DashboardServer
from dashboard.backend.stream_proxy import StreamProxy


# ==================== DashboardServer Tests ====================

class TestDashboardServer:
    def test_init(self):
        server = DashboardServer(port=8001)
        assert server.host == "0.0.0.0"
        assert server.port == 8001
        assert server.app is not None

    def test_ws_client_channels(self):
        server = DashboardServer()
        assert "video" in server._ws_clients
        assert "agents" in server._ws_clients
        assert "messages" in server._ws_clients
        assert "brain" in server._ws_clients
        assert "chat" in server._ws_clients
        assert "performance" in server._ws_clients

    def test_attach_system(self):
        from core.event_bus import EventBus
        from core.agent_registry import AgentRegistry

        server = DashboardServer()
        event_bus = EventBus()
        registry = AgentRegistry(event_bus=event_bus)

        server.attach_system(event_bus, registry)
        assert server._event_bus is event_bus
        assert server._agent_registry is registry
        assert server._stream_proxy is not None

    @pytest.mark.asyncio
    async def test_broadcast_handles_dead_clients(self):
        server = DashboardServer()

        # Create mock WebSocket that raises on send
        dead_ws = AsyncMock()
        dead_ws.send_text = AsyncMock(side_effect=Exception("disconnected"))
        server._ws_clients["agents"].add(dead_ws)

        # Should not raise
        await server._broadcast("agents", '{"test": true}')
        # Dead client should be removed
        assert dead_ws not in server._ws_clients["agents"]


# ==================== StreamProxy Tests ====================

class TestStreamProxy:
    def test_init(self):
        clients = set()
        proxy = StreamProxy(clients, target_fps=15)
        assert proxy.frame_count == 0

    @pytest.mark.asyncio
    async def test_forward_frame(self):
        # Create mock WebSocket
        mock_ws = AsyncMock()
        clients = {mock_ws}
        proxy = StreamProxy(clients, target_fps=1000)  # High FPS = no rate limit

        # Send a fake JPEG frame
        fake_jpeg = b'\xff\xd8\xff\xe0' + b'\x00' * 100
        await proxy.on_video_frame("edge1", fake_jpeg)

        assert proxy.frame_count == 1
        mock_ws.send_text.assert_called_once()

        # Verify the message is valid JSON with base64 data
        sent_data = json.loads(mock_ws.send_text.call_args[0][0])
        assert sent_data["type"] == "video_frame"
        assert "jpeg_base64" in sent_data

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        mock_ws = AsyncMock()
        clients = {mock_ws}
        proxy = StreamProxy(clients, target_fps=2)  # Very low FPS

        fake_jpeg = b'\xff\xd8\xff\xe0' + b'\x00' * 50

        # Send multiple frames rapidly
        for _ in range(5):
            await proxy.on_video_frame("edge1", fake_jpeg)

        # Should have sent only 1 frame due to rate limiting
        assert proxy.frame_count == 1

    @pytest.mark.asyncio
    async def test_no_clients_no_crash(self):
        clients = set()
        proxy = StreamProxy(clients)
        # Should not crash with no clients
        await proxy.on_video_frame("edge1", b'\xff\xd8\xff\xe0')

    def test_set_detections(self):
        clients = set()
        proxy = StreamProxy(clients)
        proxy.set_detections([{"id": 1, "x": 100}])
        assert len(proxy._latest_detections) == 1


# ==================== API Routes Tests ====================

class TestAPIRoutes:
    @pytest.mark.asyncio
    async def test_api_routes_creation(self):
        """Verify API routes are registered on the FastAPI app."""
        server = DashboardServer()
        routes = [r.path for r in server.app.routes]
        assert "/api/agents" in routes
        assert "/api/system/status" in routes

    @pytest.mark.asyncio
    async def test_system_status_endpoint(self):
        """Test the system status via TestClient."""
        from fastapi.testclient import TestClient

        server = DashboardServer()
        client = TestClient(server.app)

        response = client.get("/api/system/status")
        assert response.status_code == 200
        data = response.json()
        assert data["system_name"] == "LaserCar EMACF"
        assert "agents_running" in data

    @pytest.mark.asyncio
    async def test_get_agents_empty(self):
        from fastapi.testclient import TestClient

        server = DashboardServer()
        client = TestClient(server.app)

        response = client.get("/api/agents")
        assert response.status_code == 200
        assert response.json()["agents"] == []

    @pytest.mark.asyncio
    async def test_get_agents_with_registry(self):
        from fastapi.testclient import TestClient
        from core.event_bus import EventBus
        from core.agent_registry import AgentRegistry
        from agents.brain.agent import BrainAgent

        server = DashboardServer()
        event_bus = EventBus()
        registry = AgentRegistry(event_bus=event_bus)

        brain = BrainAgent()
        brain.initialize()
        await registry.register(brain)

        server.attach_system(event_bus, registry)
        client = TestClient(server.app)

        response = client.get("/api/agents")
        assert response.status_code == 200
        agents = response.json()["agents"]
        assert len(agents) >= 1
        assert any(a["name"] == "Brain" for a in agents)

    @pytest.mark.asyncio
    async def test_switch_mode(self):
        from fastapi.testclient import TestClient
        from core.event_bus import EventBus

        server = DashboardServer()
        event_bus = EventBus()
        server._event_bus = event_bus

        received = []
        event_bus.subscribe("mode_command", lambda e: received.append(e))

        client = TestClient(server.app)
        response = client.post("/api/system/mode", json={"mode": "SWA"})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_performance_empty(self):
        from fastapi.testclient import TestClient

        server = DashboardServer()
        client = TestClient(server.app)

        response = client.get("/api/performance")
        assert response.status_code == 200
        assert response.json()["metrics"] == {}
