import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock

from close_person_analyzer import ClosePersonAnalyzer


def make_mock_model(objects_dict):
    """Return a mock model whose get_detected_objects returns objects_dict."""
    model = MagicMock()
    model.get_detected_objects = MagicMock(return_value=objects_dict)
    return model


def make_frame(h=100, w=100):
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_depth_map(h=100, w=100, fill=0.5):
    return np.full((h, w), fill, dtype=np.float32)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_values(self):
        analyzer = ClosePersonAnalyzer()
        assert analyzer.object_name == "person"
        assert analyzer.depth_limit == 0.011

    def test_custom_values(self):
        analyzer = ClosePersonAnalyzer(object_name="car", depth_limit=0.05)
        assert analyzer.object_name == "car"
        assert analyzer.depth_limit == 0.05


# ---------------------------------------------------------------------------
# No detections
# ---------------------------------------------------------------------------

class TestNoDetections:
    def test_returns_green_border_when_no_persons(self):
        analyzer = ClosePersonAnalyzer()
        frame = make_frame()
        depth_map = make_depth_map()
        model = make_mock_model({})          # empty — no objects detected
        detections = MagicMock()

        _, color, size, message = analyzer.analyze(frame, depth_map, detections, model)

        assert color == (0, 255, 0)
        assert size == 20
        assert message == ""

    def test_returns_green_border_when_object_class_not_present(self):
        analyzer = ClosePersonAnalyzer(object_name="person")
        model = make_mock_model({"car": [[10, 10, 50, 50]]})
        _, color, _, message = analyzer.analyze(make_frame(), make_depth_map(), MagicMock(), model)

        assert color == (0, 255, 0)
        assert message == ""


# ---------------------------------------------------------------------------
# STOP alert (too close)
# ---------------------------------------------------------------------------

class TestStopAlert:
    def test_red_border_and_stop_message_when_very_close(self):
        """
        depth_limit = 0.011
        depth_map filled with 0.0 → median = 0.0
        median_depth = 1.0 / (0.0 + 1e-10) = 1e10  → far exceeds depth_limit
        BUT we need median_depth < depth_limit for STOP.

        To trigger STOP we need: 1 / (median + eps) < 0.011
        ⟹ median > 1/0.011 ≈ 90.9
        Use fill = 100.0 → median_depth ≈ 0.01 < 0.011  ✓
        """
        analyzer = ClosePersonAnalyzer(depth_limit=0.011)
        depth_map = make_depth_map(fill=100.0)       # median_depth ≈ 0.01 < 0.011
        model = make_mock_model({"person": [[0, 0, 50, 50]]})

        _, color, _, message = analyzer.analyze(make_frame(), depth_map, MagicMock(), model)

        assert color == (0, 0, 255), "Expected red border for STOP"
        assert message == "STOP"

    def test_frame_is_annotated_with_bounding_box(self):
        analyzer = ClosePersonAnalyzer(depth_limit=0.011)
        depth_map = make_depth_map(fill=100.0)
        model = make_mock_model({"person": [[10, 10, 40, 40]]})
        frame = make_frame()

        returned_frame, _, _, _ = analyzer.analyze(frame, depth_map, MagicMock(), model)

        # Check that at least one pixel changed (bounding box was drawn)
        assert not np.all(returned_frame == 0)


# ---------------------------------------------------------------------------
# CAUTION alert (caution zone)
# ---------------------------------------------------------------------------

class TestCautionAlert:
    def test_orange_border_and_caution_message(self):
        """
        depth_limit = 0.011, caution zone: depth_limit <= median_depth < depth_limit * 1.2
        We need: 0.011 <= 1/(median+eps) < 0.0132

        Choose median so that median_depth ≈ 0.0115 (between 0.011 and 0.0132).
        1/0.0115 ≈ 86.96  → use fill ≈ 87.0
        """
        analyzer = ClosePersonAnalyzer(depth_limit=0.011)
        depth_map = make_depth_map(fill=87.0)   # median_depth ≈ 0.01149
        model = make_mock_model({"person": [[0, 0, 50, 50]]})

        _, color, _, message = analyzer.analyze(make_frame(), depth_map, MagicMock(), model)

        assert color == (0, 140, 255), "Expected orange border for CAUTION"
        assert message == "CAUTION"


# ---------------------------------------------------------------------------
# SAFE (person detected but far away)
# ---------------------------------------------------------------------------

class TestSafeScenario:
    def test_green_border_when_person_is_far(self):
        """
        Need median_depth >= depth_limit * 1.2 = 0.0132
        1/(median+eps) >= 0.0132  ⟹ median <= 75.75
        Use fill = 0.5 → median_depth = 1/0.5 = 2.0 >> 0.0132  ✓
        """
        analyzer = ClosePersonAnalyzer(depth_limit=0.011)
        depth_map = make_depth_map(fill=0.5)
        model = make_mock_model({"person": [[0, 0, 50, 50]]})

        _, color, _, message = analyzer.analyze(make_frame(), depth_map, MagicMock(), model)

        assert color == (0, 255, 0)
        assert message == ""


# ---------------------------------------------------------------------------
# Multiple persons — worst case wins
# ---------------------------------------------------------------------------

class TestMultiplePersons:
    def test_stop_takes_priority_over_safe_person(self):
        """When one person is safe and another triggers STOP, border should be STOP."""
        analyzer = ClosePersonAnalyzer(depth_limit=0.011)
        h, w = 100, 200
        frame = make_frame(h, w)

        # Left half: deep values (safe person at columns 0-99)
        depth_map = np.zeros((h, w), dtype=np.float32)
        depth_map[:, :100] = 0.5    # far → safe
        depth_map[:, 100:] = 100.0  # close → STOP

        model = make_mock_model({
            "person": [
                [0, 0, 100, 100],   # safe region
                [100, 0, 200, 100], # stop region
            ]
        })

        _, color, _, message = analyzer.analyze(frame, depth_map, MagicMock(), model)

        # The last evaluated detection wins; if ordering matters, at minimum STOP is reached
        # (Testing that STOP is triggered at some point in the loop)
        assert message in ("STOP", "CAUTION", "")   # behavior depends on loop order
        # More precisely: the stop region should trigger at least once
        # We test the last bbox processed triggers STOP (right-hand person)
        assert color == (0, 0, 255)
        assert message == "STOP"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_depth_map_does_not_raise(self):
        """All-zero depth map → eps prevents division by zero."""
        analyzer = ClosePersonAnalyzer(depth_limit=0.011)
        depth_map = np.zeros((100, 100), dtype=np.float32)
        model = make_mock_model({"person": [[10, 10, 50, 50]]})

        # Should not raise ZeroDivisionError
        result = analyzer.analyze(make_frame(), depth_map, MagicMock(), model)
        assert len(result) == 4

    def test_empty_bounding_box_region_does_not_raise(self):
        """Degenerate bbox where xmin==xmax or ymin==ymax yields empty array."""
        analyzer = ClosePersonAnalyzer(depth_limit=0.011)
        depth_map = make_depth_map()
        model = make_mock_model({"person": [[10, 10, 10, 10]]})  # zero-area bbox

        result = analyzer.analyze(make_frame(), depth_map, MagicMock(), model)
        assert len(result) == 4

    def test_returns_modified_frame(self):
        """The returned frame should be the same object that was passed in."""
        analyzer = ClosePersonAnalyzer()
        frame = make_frame()
        model = make_mock_model({})
        returned_frame, *_ = analyzer.analyze(frame, make_depth_map(), MagicMock(), model)
        assert returned_frame is frame
