"""Tests for Perception Agent sub-components."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.perception.noise_filter import AdvancedNoiseFilter
from agents.perception.trajectory_predictor import WeedTrajectoryPredictor
from agents.perception.weed_tracker import WeedTracker, TrackedWeed
from agents.perception.yolo_detector import Detection


# ==================== NoiseFilter Tests ====================

class TestNoiseFilter:
    def setup_method(self):
        self.nf = AdvancedNoiseFilter(filter_strength=0.3, smoothing_window=5)

    def test_first_position_returns_raw(self):
        result = self.nf.filter_position(1, 100.0, 200.0)
        np.testing.assert_array_almost_equal(result, [100.0, 200.0])

    def test_filtering_reduces_jitter(self):
        """Small movements should be smoothed out."""
        positions = [(100, 200), (101, 201), (99, 199), (100, 200), (101, 200)]
        results = []
        for x, y in positions:
            results.append(self.nf.filter_position(1, x, y))

        # Filtered positions should have less variance than raw
        raw_var = np.var(positions, axis=0)
        filtered_var = np.var(results, axis=0)
        assert np.all(filtered_var <= raw_var + 1.0)  # tolerance

    def test_outlier_rejection(self):
        """Large jumps should be rejected."""
        # Establish baseline
        for i in range(5):
            self.nf.filter_position(1, 100.0 + i, 200.0)

        # Inject outlier (100 pixel jump)
        result = self.nf.filter_position(1, 300.0, 200.0)

        # Should not jump to 300
        assert result[0] < 200.0

    def test_remove_weed(self):
        self.nf.filter_position(1, 100, 200)
        self.nf.remove_weed(1)
        assert 1 not in self.nf.position_history

    def test_filter_stats(self):
        self.nf.filter_position(1, 100, 200)
        stats = self.nf.get_filter_stats(1)
        assert stats is not None
        assert stats["positions_tracked"] == 1
        assert self.nf.get_filter_stats(999) is None


# ==================== TrajectoryPredictor Tests ====================

class TestTrajectoryPredictor:
    def setup_method(self):
        self.tp = WeedTrajectoryPredictor(max_history_length=20)

    def test_single_position_no_prediction(self):
        self.tp.update_weed_position(1, 100, 200, 0.0)
        result = self.tp.predict_complete_trajectory(1, 5.0)
        assert result is None

    def test_moving_weed_prediction(self):
        """A weed moving right should predict continuation to the right."""
        for i in range(10):
            self.tp.update_weed_position(1, 100 + i * 20, 200, i * 0.2)

        result = self.tp.predict_complete_trajectory(1, 2.0)
        assert result is not None
        assert result["is_moving"] is True
        assert result["confidence"] > 0.1

        # Predicted points should move rightward
        points = result["trajectory_points"]
        assert len(points) > 1
        assert points[-1][0] > points[0][0]

    def test_static_weed(self):
        """A stationary weed should return low confidence."""
        for i in range(5):
            self.tp.update_weed_position(1, 100, 200, i * 0.2)

        result = self.tp.predict_complete_trajectory(1, 2.0)
        # Static weed: either None or is_moving=False
        if result is not None:
            assert result["is_moving"] is False

    def test_cleanup_old(self):
        self.tp.update_weed_position(1, 100, 200, 0.0)
        removed = self.tp.cleanup_old_trajectories(100.0, max_age_seconds=15.0)
        assert 1 in removed
        assert 1 not in self.tp.weed_trajectories

    def test_set_yolo_delay(self):
        self.tp.set_yolo_delay(0.5)
        assert self.tp.yolo_processing_delay == 0.5

        self.tp.set_yolo_delay(0.01)  # Below min
        assert self.tp.yolo_processing_delay == self.tp.min_yolo_delay

    def test_movement_info(self):
        for i in range(5):
            self.tp.update_weed_position(1, 100 + i * 10, 200, i * 0.2)

        info = self.tp.get_movement_info(1)
        assert info is not None
        assert info["has_movement"] is True
        assert info["speed"] > 0

        assert self.tp.get_movement_info(999) is None


# ==================== WeedTracker Tests ====================

class TestWeedTracker:
    def setup_method(self):
        self.tracker = WeedTracker(match_threshold=50.0, max_age=10.0)

    def _make_detection(self, x, y, conf=0.8):
        return Detection(pixel_x=x, pixel_y=y, confidence=conf, box=(x-10, y-10, x+10, y+10))

    def test_new_detections_get_ids(self):
        dets = [self._make_detection(100, 200), self._make_detection(300, 400)]
        result = self.tracker.update(dets, 0.0)
        assert len(result) == 2
        assert result[0].weed_id != result[1].weed_id

    def test_matching_across_frames(self):
        # Frame 1
        self.tracker.update([self._make_detection(100, 200)], 0.0)
        first_id = self.tracker.tracked_weeds[0].weed_id

        # Frame 2: weed moved slightly
        self.tracker.update([self._make_detection(110, 205)], 0.1)
        assert self.tracker.tracked_weeds[0].weed_id == first_id
        assert self.tracker.count == 1

    def test_new_weed_added(self):
        self.tracker.update([self._make_detection(100, 200)], 0.0)
        # Frame 2: original + new weed far away
        self.tracker.update([
            self._make_detection(110, 205),
            self._make_detection(500, 500),
        ], 0.1)
        assert self.tracker.count == 2

    def test_cleanup_removes_old(self):
        self.tracker.update([self._make_detection(100, 200)], 0.0)
        removed = self.tracker.cleanup(20.0)  # 20s later
        assert len(removed) == 1
        assert self.tracker.count == 0

    def test_visible_count(self):
        self.tracker.update([self._make_detection(100, 200)], 0.0)
        assert self.tracker.visible_count == 1

        # Next frame: no detections
        self.tracker.update([], 0.1)
        assert self.tracker.visible_count == 0
        assert self.tracker.count == 1  # Still tracked, just not visible


# ==================== YoloDetector Tests ====================

class TestYoloDetector:
    def test_detect_with_model(self):
        """Integration test: run YOLO on a synthetic image."""
        from agents.perception.yolo_detector import YoloDetector

        detector = YoloDetector(model_path="yolo11nweed.pt")

        # Create a dummy green image (640x480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (34, 139, 34)  # Green

        params = {
            "yolo_confidence": 0.4,
            "yolo_iou": 0.4,
            "max_area_fraction": 0.18,
            "min_area_fraction": 0.0008,
            "max_aspect_ratio": 4.0,
            "min_aspect_ratio": 0.25,
        }

        detections = detector.detect(frame, params)
        # On a blank green image we expect 0 or very few detections
        assert isinstance(detections, list)
        assert detector.last_inference_ms > 0
        assert detector.yolo_delay > 0


# ==================== PerceptionAgent Integration Test ====================

class TestPerceptionAgent:
    @pytest.mark.asyncio
    async def test_process_frame(self):
        """Test the full pipeline: frame -> detect -> track -> filter -> events."""
        from core.event_bus import EventBus
        from agents.perception.agent import PerceptionAgent

        event_bus = EventBus()
        received_events = []
        event_bus.subscribe("weed_detection", lambda e: received_events.append(e))

        agent = PerceptionAgent()
        agent.set_event_bus(event_bus)
        agent.initialize(model_path="yolo11nweed.pt")

        # Process a blank frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        await agent.process_frame(frame, time.time())

        # Should publish a WeedDetectionEvent (even with 0 weeds)
        assert len(received_events) == 1
        assert received_events[0].event_type == "weed_detection"

    @pytest.mark.asyncio
    async def test_status(self):
        from agents.perception.agent import PerceptionAgent

        agent = PerceptionAgent()
        agent.initialize(model_path="yolo11nweed.pt")

        status = agent.get_status()
        assert status["name"] == "Perception"
        assert "tracked_weeds" in status
        assert "frames_processed" in status
