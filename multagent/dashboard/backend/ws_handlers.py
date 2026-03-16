"""
WebSocket endpoint handlers for the Dashboard.
"""

import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


def create_ws_router(server) -> APIRouter:
    """Create WebSocket routes bound to the DashboardServer instance."""
    router = APIRouter()

    @router.websocket("/ws/video")
    async def ws_video(ws: WebSocket):
        await ws.accept()
        server._ws_clients["video"].add(ws)
        logger.info("Video client connected")
        try:
            while True:
                # Keep connection alive; video is server-push only
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            server._ws_clients["video"].discard(ws)

    @router.websocket("/ws/agents")
    async def ws_agents(ws: WebSocket):
        await ws.accept()
        server._ws_clients["agents"].add(ws)
        logger.info("Agent status client connected")
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            server._ws_clients["agents"].discard(ws)

    @router.websocket("/ws/messages")
    async def ws_messages(ws: WebSocket):
        await ws.accept()
        server._ws_clients["messages"].add(ws)
        logger.info("Message flow client connected")
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            server._ws_clients["messages"].discard(ws)

    @router.websocket("/ws/brain")
    async def ws_brain(ws: WebSocket):
        await ws.accept()
        server._ws_clients["brain"].add(ws)
        logger.info("Brain thought client connected")
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            server._ws_clients["brain"].discard(ws)

    @router.websocket("/ws/chat")
    async def ws_chat(ws: WebSocket):
        """Bidirectional chat with Brain agent."""
        await ws.accept()
        server._ws_clients["chat"].add(ws)
        logger.info("Chat client connected")
        try:
            while True:
                data = await ws.receive_text()
                msg = json.loads(data)

                # Forward user message to Brain via EventBus
                if server._event_bus:
                    from agents.brain.events import UserChatEvent
                    await server._event_bus.publish(UserChatEvent(
                        source="dashboard",
                        message=msg.get("message", ""),
                    ))

                # Echo user message to all chat clients
                echo = json.dumps({
                    "role": "user",
                    "message": msg.get("message", ""),
                    "timestamp": time.time(),
                })
                await server._broadcast("chat", echo)

        except WebSocketDisconnect:
            pass
        finally:
            server._ws_clients["chat"].discard(ws)

    @router.websocket("/ws/performance")
    async def ws_performance(ws: WebSocket):
        await ws.accept()
        server._ws_clients["performance"].add(ws)
        logger.info("Performance client connected")
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            server._ws_clients["performance"].discard(ws)

    return router
