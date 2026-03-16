"""
Cloud-side edge communication manager.
Responsibilities:
  1. Manage WebSocket connections with edge devices
  2. Receive video stream and dispatch to PerceptionAgent
  3. Receive sensor data
  4. Send control commands to edge devices
  5. Heartbeat monitoring
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import websockets

from core.events import (
    EdgeConnectedEvent,
    EdgeDisconnectedEvent,
    HardwareCommand,
    HeartbeatEvent,
    Event,
)
from core.safety import SafetyPolicy

logger = logging.getLogger(__name__)

# Message type prefixes (first byte of binary message)
MSG_VIDEO_FRAME = 0x01
MSG_SENSOR_DATA = 0x02
MSG_HEARTBEAT = 0x03
MSG_REMOTE_CONTROL = 0x04
MSG_JSON = 0x00  # JSON text message


class EdgeConnection:
    """Represents a single edge device connection."""
    def __init__(self, edge_id: str, ws):
        self.edge_id = edge_id
        self.ws = ws
        self.connected_at = time.time()
        self.last_heartbeat = time.time()
        self.latency_ms: float = 0.0


class EdgeBridge:
    def __init__(self, event_bus, heartbeat_timeout_ms: float = 500):
        self._event_bus = event_bus
        self._connections: Dict[str, EdgeConnection] = {}
        self._video_callbacks: List[Callable] = []
        self._heartbeat_timeout = heartbeat_timeout_ms / 1000.0
        self._safety = SafetyPolicy()
        self._server = None
        self._running = False

    async def start(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        """Start WebSocket server, wait for edge device connections."""
        self._running = True
        self._server = await websockets.serve(
            self._handle_connection, host, port
        )
        logger.info(f"EdgeBridge started on ws://{host}:{port}")
        # Start heartbeat monitor
        asyncio.create_task(self._heartbeat_monitor())

    async def _handle_connection(self, ws, path=None):
        """Handle a new WebSocket connection from an edge device."""
        edge_id = None
        try:
            # First message should be identification
            init_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            init_data = json.loads(init_msg)
            edge_id = init_data.get("edge_id", f"edge_{len(self._connections)}")
            capabilities = init_data.get("capabilities", [])

            await self._on_edge_connected(edge_id, ws, capabilities)

            async for message in ws:
                await self._on_edge_message(edge_id, message)

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Edge connection error: {e}")
        finally:
            if edge_id:
                await self._on_edge_disconnected(edge_id, "connection_closed")

    async def _on_edge_connected(self, edge_id: str, ws, capabilities: list) -> None:
        """Edge device connection callback."""
        self._connections[edge_id] = EdgeConnection(edge_id, ws)
        logger.info(f"Edge device connected: {edge_id}")
        await self._event_bus.publish(
            EdgeConnectedEvent(source="EdgeBridge", edge_id=edge_id, capabilities=capabilities)
        )

    async def _on_edge_disconnected(self, edge_id: str, reason: str) -> None:
        """Edge device disconnection callback."""
        self._connections.pop(edge_id, None)
        logger.warning(f"Edge device disconnected: {edge_id} ({reason})")
        await self._event_bus.publish(
            EdgeDisconnectedEvent(source="EdgeBridge", edge_id=edge_id, reason=reason)
        )

    async def _on_edge_message(self, edge_id: str, message) -> None:
        """
        Handle messages from edge device.
        Binary messages: first byte = message type, rest = payload.
        Text messages: JSON.
        """
        conn = self._connections.get(edge_id)
        if not conn:
            return

        if isinstance(message, bytes) and len(message) > 1:
            msg_type = message[0]
            payload = message[1:]

            if msg_type == MSG_VIDEO_FRAME:
                for callback in self._video_callbacks:
                    try:
                        result = callback(edge_id, payload)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"Video callback error: {e}")

            elif msg_type == MSG_HEARTBEAT:
                conn.last_heartbeat = time.time()
                latency = float(payload.decode("utf-8")) if payload else 0.0
                conn.latency_ms = latency
                await self._event_bus.publish(
                    HeartbeatEvent(source="EdgeBridge", edge_id=edge_id, latency_ms=latency)
                )

            elif msg_type == MSG_SENSOR_DATA:
                data = json.loads(payload.decode("utf-8"))
                await self._event_bus.publish(Event(
                    event_type="sensor_data",
                    source=edge_id,
                    data=data,
                ))

            elif msg_type == MSG_REMOTE_CONTROL:
                data = json.loads(payload.decode("utf-8"))
                await self._event_bus.publish(Event(
                    event_type="remote_control",
                    source=edge_id,
                    data=data,
                ))

        elif isinstance(message, str):
            # JSON text message
            data = json.loads(message)
            msg_type = data.get("type", "unknown")
            if msg_type == "heartbeat":
                conn.last_heartbeat = time.time()
            else:
                await self._event_bus.publish(Event(
                    event_type=msg_type,
                    source=edge_id,
                    data=data,
                ))

    async def send_command(self, command: HardwareCommand) -> bool:
        """Send control command to edge device. Returns True if sent."""
        if not self._safety.validate_command(command):
            return False

        target = command.target_edge
        conn = self._connections.get(target)
        if not conn:
            # Try first available connection
            if self._connections:
                conn = next(iter(self._connections.values()))
            else:
                logger.warning("No edge device connected, command dropped")
                return False

        msg = json.dumps({
            "type": "command",
            "command": command.command_type,
            "params": command.params,
            "timestamp": command.timestamp,
        })
        try:
            await conn.ws.send(msg)
            return True
        except Exception as e:
            logger.error(f"Failed to send command to {conn.edge_id}: {e}")
            return False

    def register_video_callback(self, callback: Callable) -> None:
        """Register video frame callback (PerceptionAgent will register)."""
        if callback not in self._video_callbacks:
            self._video_callbacks.append(callback)

    def unregister_video_callback(self, callback: Callable) -> None:
        """Unregister video frame callback."""
        try:
            self._video_callbacks.remove(callback)
        except ValueError:
            pass

    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status of all edge devices."""
        return {
            edge_id: {
                "connected_at": conn.connected_at,
                "last_heartbeat": conn.last_heartbeat,
                "latency_ms": conn.latency_ms,
                "alive": (time.time() - conn.last_heartbeat) < self._heartbeat_timeout,
            }
            for edge_id, conn in self._connections.items()
        }

    async def _heartbeat_monitor(self) -> None:
        """Monitor heartbeats and disconnect stale connections."""
        while self._running:
            await asyncio.sleep(self._heartbeat_timeout)
            now = time.time()
            stale = [
                eid for eid, conn in self._connections.items()
                if (now - conn.last_heartbeat) > self._heartbeat_timeout
            ]
            for edge_id in stale:
                logger.warning(f"Heartbeat timeout for {edge_id}")
                await self._on_edge_disconnected(edge_id, "heartbeat_timeout")

    async def stop(self) -> None:
        """Stop the EdgeBridge server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        # Close all connections
        for conn in list(self._connections.values()):
            try:
                await conn.ws.close()
            except Exception:
                pass
        self._connections.clear()
        logger.info("EdgeBridge stopped")

    @property
    def safety(self) -> SafetyPolicy:
        return self._safety

    @property
    def connected_edges(self) -> List[str]:
        return list(self._connections.keys())
