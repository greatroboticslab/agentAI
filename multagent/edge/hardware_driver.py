"""
Hardware driver: Helios DAC + ESP32 communication.
Extracted from lasercar.py HeliosPoint, DAC init, ESP32 serial.
"""

import ctypes
import logging
import platform
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class HeliosPoint(ctypes.Structure):
    """Ctypes structure for Helios DAC laser point."""
    _fields_ = [
        ("x", ctypes.c_uint16),    # 0-0xFFF
        ("y", ctypes.c_uint16),    # 0-0xFFF
        ("r", ctypes.c_uint8),
        ("g", ctypes.c_uint8),
        ("b", ctypes.c_uint8),
        ("i", ctypes.c_uint8),     # Intensity
    ]


# Constants
LASER_MAX = 0xFFF
POINTS_PER_FRAME = 1000
FRAME_DURATION = 30000
COLOR_VALUE = 255
INTENSITY = 130


class HeliosDAC:
    """Helios laser DAC controller."""

    def __init__(self, lib_path: Optional[str] = None):
        self._lib = None
        self._num_devices = 0
        self._lib_path = lib_path

    def initialize(self) -> int:
        """Initialize Helios DAC library and return number of devices."""
        try:
            if self._lib_path:
                self._lib = ctypes.cdll.LoadLibrary(self._lib_path)
            else:
                # Auto-detect library path
                system = platform.system()
                if system == "Windows":
                    self._lib = ctypes.cdll.LoadLibrary("HeliosLaserDAC.dll")
                elif system == "Darwin":
                    self._lib = ctypes.cdll.LoadLibrary("libHeliosLaserDAC.dylib")
                else:
                    self._lib = ctypes.cdll.LoadLibrary("libHeliosLaserDAC.so")

            self._num_devices = self._lib.OpenDevices()
            logger.info(f"Helios DAC: {self._num_devices} device(s) found")
            return self._num_devices

        except Exception as e:
            logger.error(f"Helios DAC init failed: {e}")
            return 0

    def create_single_point_frame(
        self, x: int, y: int, r: int = COLOR_VALUE, g: int = COLOR_VALUE,
        b: int = COLOR_VALUE, intensity: int = INTENSITY,
    ) -> ctypes.Array:
        """Create a frame with a single repeated point."""
        frame = (HeliosPoint * POINTS_PER_FRAME)()
        point = HeliosPoint(
            x=max(0, min(LASER_MAX, x)),
            y=max(0, min(LASER_MAX, y)),
            r=r, g=g, b=b, i=intensity,
        )
        for j in range(POINTS_PER_FRAME):
            frame[j] = point
        return frame

    def create_pattern_frame(
        self, points: List[Tuple[int, int]],
        r: int = COLOR_VALUE, g: int = COLOR_VALUE,
        b: int = COLOR_VALUE, intensity: int = INTENSITY,
    ) -> ctypes.Array:
        """Create a frame from pattern points."""
        frame = (HeliosPoint * POINTS_PER_FRAME)()

        if not points:
            return frame

        points_per_pattern = max(1, POINTS_PER_FRAME // len(points))

        idx = 0
        for px, py in points:
            hp = HeliosPoint(
                x=max(0, min(LASER_MAX, int(px))),
                y=max(0, min(LASER_MAX, int(py))),
                r=r, g=g, b=b, i=intensity,
            )
            for _ in range(points_per_pattern):
                if idx >= POINTS_PER_FRAME:
                    break
                frame[idx] = hp
                idx += 1

        # Pad remaining with last point
        if idx < POINTS_PER_FRAME and points:
            last = HeliosPoint(
                x=max(0, min(LASER_MAX, int(points[-1][0]))),
                y=max(0, min(LASER_MAX, int(points[-1][1]))),
                r=r, g=g, b=b, i=intensity,
            )
            while idx < POINTS_PER_FRAME:
                frame[idx] = last
                idx += 1

        return frame

    def send_frame(self, motor_index: int, frame: ctypes.Array) -> bool:
        """Send frame to a specific motor/device."""
        if not self._lib or motor_index >= self._num_devices:
            return False

        # Wait for device ready (up to 32ms)
        for _ in range(32):
            if self._lib.GetStatus(motor_index) == 1:
                break
            time.sleep(0.001)

        try:
            self._lib.WriteFrame(
                motor_index, FRAME_DURATION, 0,
                ctypes.pointer(frame), POINTS_PER_FRAME,
            )
            return True
        except Exception as e:
            logger.error(f"DAC write error motor {motor_index}: {e}")
            return False

    def close(self) -> None:
        if self._lib:
            try:
                self._lib.CloseDevices()
            except Exception:
                pass

    @property
    def num_devices(self) -> int:
        return self._num_devices


class ESP32Controller:
    """ESP32 serial controller for laser on/off/power and vehicle commands."""

    COMMON_PORTS = [
        "/dev/cu.usbserial-0001", "/dev/ttyUSB0",
        "COM3", "COM4", "COM5",
    ]
    IDENTIFIERS = ["usbserial", "ch340", "cp210"]

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self._serial = None
        self._port = port
        self._baudrate = baudrate
        self._laser_enabled = False
        self._laser_power = 128

    def connect(self) -> bool:
        """Connect to ESP32 via serial."""
        import serial
        import serial.tools.list_ports

        if self._port:
            ports_to_try = [self._port]
        else:
            # Auto-detect
            ports_to_try = []
            for port_info in serial.tools.list_ports.comports():
                for ident in self.IDENTIFIERS:
                    if ident in port_info.device.lower():
                        ports_to_try.append(port_info.device)
            ports_to_try.extend(self.COMMON_PORTS)

        for port in ports_to_try:
            try:
                self._serial = serial.Serial(port, self._baudrate, timeout=1)
                time.sleep(1)  # Wait for Arduino reset
                self.send_command("OFF")  # Laser off by default
                logger.info(f"ESP32 connected: {port}")
                return True
            except Exception:
                continue

        logger.warning("ESP32 not found")
        return False

    def send_command(self, cmd: str) -> Optional[str]:
        """Send command to ESP32 and return response."""
        if not self._serial:
            return None
        try:
            self._serial.write(f"{cmd}\n".encode())
            time.sleep(0.05)
            if self._serial.in_waiting:
                return self._serial.readline().decode().strip()
            return None
        except Exception as e:
            logger.error(f"ESP32 command error: {e}")
            return None

    def laser_on(self) -> None:
        self.send_command("ON")
        self._laser_enabled = True

    def laser_off(self) -> None:
        self.send_command("OFF")
        self._laser_enabled = False

    def set_laser_power(self, power: int) -> None:
        power = max(0, min(255, power))
        self.send_command(f"POWER {power}")
        self._laser_power = power

    def vehicle_forward(self, speed: int = 50) -> None:
        self.send_command(f"FORWARD {speed}")

    def vehicle_stop(self) -> None:
        self.send_command("STOP")

    def close(self) -> None:
        if self._serial:
            try:
                self.laser_off()
                self._serial.close()
            except Exception:
                pass

    @property
    def laser_enabled(self) -> bool:
        return self._laser_enabled

    @property
    def laser_power(self) -> int:
        return self._laser_power
