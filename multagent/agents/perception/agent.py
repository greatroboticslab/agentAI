"""
PerceptionAgent: Weed detection, tracking, and trajectory prediction.
Receives video frames from EdgeBridge, runs YOLO, publishes detection events.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from pydantic import Field

from core.embodied_role import EmbodiedRole
from core.events import Event

from agents.perception.events import NewWeedEvent, WeedDetectionEvent, WeedLostEvent
from agents.perception.noise_filter import AdvancedNoiseFilter
from agents.perception.trajectory_predictor import WeedTrajectoryPredictor
from agents.perception.weed_tracker import WeedTracker
from agents.perception.yolo_detector import YoloDetector

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "yolo_confidence": 0.4,
    "yolo_iou": 0.4,
    "max_area_fraction": 0.18,
    "min_area_fraction": 0.0008,
    "max_aspect_ratio": 4.0,
    "min_aspect_ratio": 0.25,
    "noise_filter_strength": 0.3,
    "noise_smoothing_window": 5,
    "noise_movement_threshold": 3.0,
    "noise_outlier_threshold": 50.0,
}


class PerceptionAgent(EmbodiedRole):
    """
    Perception Agent — the system's eyes.
    Pipeline: Video Frame -> YOLO -> Tracking -> Noise Filter -> Trajectory Prediction -> Events
    """

    name: str = "Perception"
    profile: str = "Weed detection, tracking, and trajectory prediction"
    goal: str = "Detect and track weeds in real-time with high accuracy"
    capabilities: list = ["perception", "detection", "tracking"]
    subscribed_events: list = ["param_update"]
    params: Dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_PARAMS))

    # Sub-components (initialized lazily)
    _detector: Optional[YoloDetector] = None
    _tracker: Optional[WeedTracker] = None
    _noise_filter: Optional[AdvancedNoiseFilter] = None
    _trajectory_predictor: Optional[WeedTrajectoryPredictor] = None
    _last_cleanup_time: float = 0.0
    _frame_count: int = 0

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._last_cleanup_time = time.time()
        self._frame_count = 0

    def initialize(self, model_path: str = "yolo11nweed.pt") -> None:
        """Initialize sub-components. Call after construction."""
        self._detector = YoloDetector(model_path=model_path)
        self._tracker = WeedTracker()
        self._noise_filter = AdvancedNoiseFilter(
            filter_strength=self.params.get("noise_filter_strength", 0.3),
            smoothing_window=self.params.get("noise_smoothing_window", 5),
        )
        self._trajectory_predictor = WeedTrajectoryPredictor()
        logger.info("PerceptionAgent initialized")

    async def run_loop(self) -> None:
        """Main loop: register video callback and run."""
        if self._detector is None:
            self.initialize()

        # Register for video frames from EdgeBridge
        if self._edge_bridge:
            self._edge_bridge.register_video_callback(self._on_video_frame_raw)

        await super().run_loop()

    async def _on_video_frame_raw(self, edge_id: str, frame_bytes: bytes) -> None:
        """Handle raw JPEG bytes from EdgeBridge."""
        import cv2
        import numpy as np

        # Decode JPEG
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            await self.process_frame(frame, time.time())

    async def process_frame(self, frame, timestamp: float) -> None:
        """
        Process a single video frame through the full pipeline.
        Can be called directly for testing without EdgeBridge.
        """
        import numpy as np

        if self._detector is None:
            return

        self._frame_count += 1

        # 1. YOLO detection
        detections = self._detector.detect(frame, self.params)
        yolo_latency = self._detector.last_inference_ms
        self.report_latency("yolo_inference_ms", yolo_latency)

        # Update trajectory predictor with actual YOLO delay
        self._trajectory_predictor.set_yolo_delay(self._detector.yolo_delay)

        # 2. Track weeds (assign IDs, match across frames)
        prev_ids = {w.weed_id for w in self._tracker.tracked_weeds}
        tracked_weeds = self._tracker.update(detections, timestamp)
        curr_ids = {w.weed_id for w in tracked_weeds}

        # 3. Publish NewWeedEvent for newly appeared weeds
        new_ids = curr_ids - prev_ids
        for weed in tracked_weeds:
            if weed.weed_id in new_ids:
                if self._event_bus:
                    await self._event_bus.publish(NewWeedEvent(
                        source=self.name,
                        weed_id=weed.weed_id,
                        position=(weed.pixel_x, weed.pixel_y),
                        confidence=weed.confidence,
                    ))

        # 4. Noise filter + trajectory prediction for visible weeds
        for weed in tracked_weeds:
            if weed.visible_this_frame:
                filtered_pos = self._noise_filter.filter_position(
                    weed.weed_id, weed.pixel_x, weed.pixel_y
                )
                weed.filtered_x = float(filtered_pos[0])
                weed.filtered_y = float(filtered_pos[1])

                self._trajectory_predictor.update_weed_position(
                    weed.weed_id, filtered_pos[0], filtered_pos[1], timestamp
                )

        # 5. Periodic cleanup
        if timestamp - self._last_cleanup_time > 5.0:
            removed_ids = self._trajectory_predictor.cleanup_old_trajectories(timestamp)
            removed_tracker_ids = self._tracker.cleanup(timestamp)

            # Clean up noise filter for removed weeds
            for wid in set(removed_ids) | set(removed_tracker_ids):
                self._noise_filter.remove_weed(wid)
                if self._event_bus:
                    await self._event_bus.publish(WeedLostEvent(
                        source=self.name, weed_id=wid
                    ))

            self._last_cleanup_time = timestamp

        # 6. Publish main detection event
        weed_data = [
            {
                "weed_id": w.weed_id,
                "pixel_x": w.pixel_x,
                "pixel_y": w.pixel_y,
                "filtered_x": w.filtered_x,
                "filtered_y": w.filtered_y,
                "confidence": w.confidence,
                "box": w.box,
                "visible": w.visible_this_frame,
            }
            for w in tracked_weeds
            if w.visible_this_frame
        ]

        if self._event_bus:
            await self._event_bus.publish(WeedDetectionEvent(
                source=self.name,
                weeds=weed_data,
                frame_shape=frame.shape[:2],
                yolo_latency_ms=yolo_latency,
                frame_timestamp=timestamp,
            ))

    async def on_event(self, event: Event) -> None:
        """Handle parameter update events from Brain."""
        if event.event_type == "param_update":
            if event.data.get("target_agent") == self.name:
                self.update_params(event.data["params"])

    def on_params_updated(self, changed_params: dict) -> None:
        """Apply parameter changes to sub-components."""
        if self._noise_filter:
            if "noise_filter_strength" in changed_params:
                self._noise_filter.set_filter_strength(changed_params["noise_filter_strength"])
            if "noise_smoothing_window" in changed_params:
                self._noise_filter.set_smoothing_window(changed_params["noise_smoothing_window"])
            if "noise_movement_threshold" in changed_params:
                self._noise_filter.set_movement_threshold(changed_params["noise_movement_threshold"])
            if "noise_outlier_threshold" in changed_params:
                self._noise_filter.set_outlier_threshold(changed_params["noise_outlier_threshold"])

    # --- Query methods for other Agents ---

    def get_trajectory_prediction(self, weed_id: int, duration: float, speed_scale: float = 1.0):
        """Get trajectory prediction for a specific weed."""
        if self._trajectory_predictor:
            return self._trajectory_predictor.predict_complete_trajectory(
                weed_id, duration, speed_scale
            )
        return None

    def get_movement_info(self, weed_id: int):
        """Get movement info for a weed."""
        if self._trajectory_predictor:
            return self._trajectory_predictor.get_movement_info(weed_id)
        return None

    def get_status(self) -> dict:
        """Extended status with perception-specific info."""
        status = super().get_status()
        status.update({
            "tracked_weeds": self._tracker.count if self._tracker else 0,
            "visible_weeds": self._tracker.visible_count if self._tracker else 0,
            "frames_processed": self._frame_count,
        })
        return status
