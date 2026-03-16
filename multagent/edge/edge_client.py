"""
EdgeClient: main program for the edge device.
Connects to cloud, streams video, receives commands, runs safety monitor.
"""

import asyncio
import json
import logging
import struct
import time
from typing import Optional

logger = logging.getLogger(__name__)


class EdgeClient:
    """
    Main edge device client.
    Connects to cloud WebSocket, streams video, receives and executes commands.
    """

    def __init__(self, cloud_url: str = "ws://localhost:8765"):
        self.cloud_url = cloud_url
        self._ws = None
        self._running = False

        # Components (lazily initialized)
        self._camera = None
        self._executor = None
        self._flysky = None
        self._safety = None

    def initialize(
        self,
        camera_index: int = 0,
        esp32_port: Optional[str] = None,
        dac_lib_path: Optional[str] = None,
        flysky_port: int = 9090,
    ) -> None:
        """Initialize all edge components."""
        from edge.camera_streamer import CameraStreamer
        from edge.command_executor import CommandExecutor
        from edge.flysky_receiver import FlySkyReceiver
        from edge.hardware_driver import ESP32Controller, HeliosDAC
        from edge.safety_monitor import SafetyMonitor

        # Hardware
        dac = HeliosDAC(lib_path=dac_lib_path)
        dac.initialize()

        esp32 = ESP32Controller(port=esp32_port)
        esp32.connect()

        # Components
        self._camera = CameraStreamer(camera_index=camera_index)
        self._executor = CommandExecutor(dac=dac, esp32=esp32)
        self._flysky = FlySkyReceiver(listen_port=flysky_port)
        self._safety = SafetyMonitor(esp32=esp32)

        logger.info("EdgeClient initialized")

    async def run(self) -> None:
        """Main loop: connect to cloud and run all tasks."""
        import websockets

        if self._camera is None:
            logger.error("EdgeClient not initialized. Call initialize() first.")
            return

        self._running = True

        while self._running:
            try:
                async with websockets.connect(self.cloud_url) as ws:
                    self._ws = ws
                    logger.info(f"Connected to cloud: {self.cloud_url}")

                    # Open camera
                    self._camera.open()

                    # Run all tasks in parallel
                    await asyncio.gather(
                        self._safety.run(),
                        self._camera.stream_loop(ws),
                        self._receive_commands(ws),
                        self._send_heartbeat(ws),
                        self._flysky.receive_loop(),
                        return_exceptions=True,
                    )

            except Exception as e:
                logger.error(f"Connection error: {e}")
                await asyncio.sleep(2)  # Retry delay

    async def _receive_commands(self, ws) -> None:
        """Receive and execute commands from cloud."""
        async for message in ws:
            try:
                if isinstance(message, bytes):
                    # Binary protocol: first byte = type
                    msg_type = message[0]
                    if msg_type == 0x03:
                        # Heartbeat ACK
                        self._safety.update_heartbeat()
                    continue

                # JSON command
                result = self._executor.execute(message)
                await ws.send(json.dumps(result))

            except Exception as e:
                logger.error(f"Command handling error: {e}")

    async def _send_heartbeat(self, ws) -> None:
        """Send periodic heartbeat to cloud."""
        while self._running:
            try:
                heartbeat = struct.pack("!Bd", 0x03, time.time())
                await ws.send(heartbeat)
            except Exception:
                break
            await asyncio.sleep(0.2)  # 200ms interval

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._camera:
            self._camera.stop()
        if self._flysky:
            self._flysky.stop()
        if self._safety:
            self._safety.stop()
        logger.info("EdgeClient shutdown complete")


def main():
    """Entry point for edge device."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="EMACF Edge Client")
    parser.add_argument("--config", default="edge/config.yaml")
    parser.add_argument("--cloud-url", default="ws://localhost:8765")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    client = EdgeClient(cloud_url=args.cloud_url)
    client.initialize()

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        asyncio.run(client.shutdown())


if __name__ == "__main__":
    main()
