"""
Video stream proxy: receives frames from EdgeBridge, forwards to dashboard clients.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Set

logger = logging.getLogger(__name__)


class StreamProxy:
    """
    Proxies video frames from EdgeBridge to Dashboard WebSocket clients.
    Optionally adds detection overlays as metadata.
    """

    def __init__(self, ws_clients: Set, target_fps: int = 15):
        self._ws_clients = ws_clients
        self._target_fps = target_fps
        self._frame_interval = 1.0 / target_fps
        self._last_send_time: float = 0
        self._frame_count: int = 0
        self._latest_detections: list = []

    def set_detections(self, detections: list) -> None:
        """Update latest detection overlay data."""
        self._latest_detections = detections

    async def on_video_frame(self, edge_id: str, frame_bytes: bytes) -> None:
        """
        Called when a video frame arrives from EdgeBridge.
        Forwards to all connected dashboard video clients.
        """
        now = time.time()

        # Rate limit to target FPS
        if now - self._last_send_time < self._frame_interval:
            return

        self._last_send_time = now
        self._frame_count += 1

        if not self._ws_clients:
            return

        # Build message: JSON metadata + base64 frame
        msg = json.dumps({
            "type": "video_frame",
            "frame_number": self._frame_count,
            "timestamp": now,
            "jpeg_base64": base64.b64encode(frame_bytes).decode("ascii"),
            "detections": self._latest_detections,
        })

        # Broadcast to all video clients
        dead = set()
        for ws in self._ws_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self._ws_clients -= dead

    @property
    def frame_count(self) -> int:
        return self._frame_count
