"""
FlySky remote control receiver.
Extracted from lasercar.py FlySkyRemoteControl network communication.
"""

import asyncio
import json
import logging
import struct
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class FlySkyReceiver:
    """Receives and parses FlySky remote control data via UDP."""

    # Channel mapping
    CH_THROTTLE = 2    # Channel 3 (0-indexed)
    CH_STEERING = 0    # Channel 1
    CH_AUX1 = 4        # Channel 5
    CH_AUX2 = 5        # Channel 6

    def __init__(self, listen_port: int = 9090):
        self.listen_port = listen_port
        self._running = False
        self._latest_data: Dict = {}
        self._on_data_callback: Optional[Callable] = None

    def set_callback(self, callback: Callable) -> None:
        """Set callback for when new data arrives."""
        self._on_data_callback = callback

    async def receive_loop(self) -> None:
        """Listen for FlySky data via UDP."""
        self._running = True

        transport, protocol = await asyncio.get_event_loop().create_datagram_endpoint(
            lambda: _FlySkyProtocol(self),
            local_addr=("0.0.0.0", self.listen_port),
        )

        logger.info(f"FlySky receiver listening on port {self.listen_port}")

        try:
            while self._running:
                await asyncio.sleep(0.1)
        finally:
            transport.close()

    def process_data(self, data: bytes) -> Optional[dict]:
        """Parse incoming FlySky data packet."""
        try:
            decoded = json.loads(data.decode())
            channels = decoded.get("channels", [])
            switches = decoded.get("switches", {})

            self._latest_data = {
                "channels": channels,
                "switches": switches,
                "timestamp": decoded.get("timestamp", 0),
            }

            if self._on_data_callback:
                self._on_data_callback(self._latest_data)

            return self._latest_data

        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def stop(self) -> None:
        self._running = False

    @property
    def latest_data(self) -> dict:
        return self._latest_data


class _FlySkyProtocol(asyncio.DatagramProtocol):
    """UDP protocol handler for FlySky data."""

    def __init__(self, receiver: FlySkyReceiver):
        self._receiver = receiver

    def datagram_received(self, data: bytes, addr) -> None:
        self._receiver.process_data(data)
