"""
YOLO inference wrapper.
Extracted from lasercar.py detection_thread() YOLO logic.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single YOLO detection result."""
    pixel_x: int  # Center x
    pixel_y: int  # Center y
    confidence: float
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area_fraction: float = 0.0
    aspect_ratio: float = 1.0
    cls: int = 0


class YoloDetector:
    """YOLO inference wrapper with area/aspect ratio filtering."""

    def __init__(self, model_path: str = "yolo11nweed.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self._last_inference_ms: float = 0.0
        self._yolo_delay: float = 1.0  # Moving average of inference time
        self._min_delay: float = 0.2
        self._max_delay: float = 3.0
        logger.info(f"YOLO model loaded: {model_path}")

    def detect(
        self,
        frame: np.ndarray,
        params: Dict[str, Any],
    ) -> List[Detection]:
        """
        Run YOLO inference on a frame with filtering.

        Args:
            frame: BGR image (numpy array)
            params: Detection parameters including:
                - yolo_confidence (float): Min confidence threshold
                - yolo_iou (float): NMS IOU threshold
                - max_area_fraction (float): Max bounding box area as fraction of frame
                - min_area_fraction (float): Min bounding box area as fraction of frame
                - max_aspect_ratio (float): Max width/height ratio
                - min_aspect_ratio (float): Min width/height ratio

        Returns:
            List of Detection objects that pass all filters.
        """
        conf = params.get("yolo_confidence", 0.4)
        iou = params.get("yolo_iou", 0.4)
        max_area_frac = params.get("max_area_fraction", 0.18)
        min_area_frac = params.get("min_area_fraction", 0.0008)
        max_aspect = params.get("max_aspect_ratio", 4.0)
        min_aspect = params.get("min_aspect_ratio", 0.25)

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_w * frame_h

        # Run YOLO inference
        start = time.perf_counter()
        try:
            results = self.model.predict(
                frame,
                conf=conf,
                iou=iou,
                agnostic_nms=True,
                max_det=100,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._last_inference_ms = elapsed_ms

        # Update YOLO delay moving average (in seconds)
        actual_delay = elapsed_ms / 1000.0
        self._yolo_delay = 0.8 * self._yolo_delay + 0.2 * actual_delay
        self._yolo_delay = max(self._min_delay, min(self._yolo_delay, self._max_delay))

        # Process results
        detections = []
        if not results or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf_val = float(box.conf.cpu().numpy()[0])
                cls = int(box.cls.cpu().numpy()[0])

                if cls != 0:  # Only class 0 (weed)
                    continue

                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                area_fraction = (w * h) / frame_area

                # Area filtering
                if area_fraction > max_area_frac or area_fraction < min_area_frac:
                    continue

                # Aspect ratio filtering
                aspect_ratio = w / h if h > 0 else 999
                if aspect_ratio > max_aspect or aspect_ratio < min_aspect:
                    continue

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if 0 <= cx < frame_w and 0 <= cy < frame_h:
                    detections.append(Detection(
                        pixel_x=cx,
                        pixel_y=cy,
                        confidence=conf_val,
                        box=(x1, y1, x2, y2),
                        area_fraction=area_fraction,
                        aspect_ratio=aspect_ratio,
                        cls=cls,
                    ))
            except Exception:
                continue

        return detections

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    @property
    def yolo_delay(self) -> float:
        """Current moving-average YOLO delay in seconds."""
        return self._yolo_delay
