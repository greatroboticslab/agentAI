"""
Camera capture and streaming to cloud.
Captures frames via OpenCV, JPEG encodes, sends over WebSocket.
"""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class CameraStreamer:
    """Captures camera frames and streams to cloud via WebSocket."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        jpeg_quality: int = 70,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self._cap = None
        self._running = False

    def open(self) -> bool:
        """Open camera device."""
        import cv2

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_index}")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened: {actual_w}x{actual_h}")
        return True

    def read_frame(self) -> Optional[bytes]:
        """Capture and JPEG-encode a single frame. Returns bytes or None."""
        import cv2

        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        _, jpeg_data = cv2.imencode(".jpg", frame, encode_params)
        return jpeg_data.tobytes()

    async def stream_loop(self, ws) -> None:
        """Continuously capture frames and send over WebSocket."""
        self._running = True
        interval = 1.0 / self.fps

        while self._running:
            start = time.time()
            frame_data = self.read_frame()

            if frame_data:
                try:
                    # Protocol: 0x01 (video) + timestamp (8 bytes) + JPEG data
                    import struct
                    header = struct.pack("!Bd", 0x01, time.time())
                    await ws.send(header + frame_data)
                except Exception as e:
                    logger.error(f"Stream send error: {e}")
                    break

            elapsed = time.time() - start
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)

    def stop(self) -> None:
        """Stop streaming and release camera."""
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None

    def set_resolution(self, width: int, height: int) -> None:
        """Update camera resolution."""
        import cv2

        self.width = width
        self.height = height
        if self._cap and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
