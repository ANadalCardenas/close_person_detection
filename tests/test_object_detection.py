import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import torch
import pytest
from unittest.mock import MagicMock, patch

from object_detection import ObjectDetection


def make_mock_model(names_map):
    """Return a mock YOLO inner model with a given names mapping."""
    model = MagicMock()
    model.names = names_map
    return model


def make_detections(*rows):
    """
    Build a torch.Tensor with shape [N, 6] from a list of
    (xmin, ymin, xmax, ymax, conf, cls_id) tuples.
    """
    return torch.tensor(rows, dtype=torch.float32)


# ---------------------------------------------------------------------------
# get_detected_objects (static method — no heavy model needed)
# ---------------------------------------------------------------------------

class TestGetDetectedObjects:
    def test_empty_detections_returns_empty_dict(self):
        model = make_mock_model({})
        detections = make_detections()   # shape [0, 6]
        result = ObjectDetection.get_detected_objects(model, detections)
        assert result == {}

    def test_single_detection(self):
        model = make_mock_model({0: "person"})
        detections = make_detections([10.0, 20.0, 50.0, 80.0, 0.9, 0.0])

        result = ObjectDetection.get_detected_objects(model, detections)

        assert "person" in result
        assert len(result["person"]) == 1
        assert result["person"][0] == pytest.approx([10.0, 20.0, 50.0, 80.0])

    def test_multiple_detections_same_class(self):
        model = make_mock_model({0: "person"})
        detections = make_detections(
            [0.0,  0.0,  40.0, 40.0, 0.95, 0.0],
            [50.0, 50.0, 90.0, 90.0, 0.80, 0.0],
        )

        result = ObjectDetection.get_detected_objects(model, detections)

        assert "person" in result
        assert len(result["person"]) == 2

    def test_multiple_detections_different_classes(self):
        model = make_mock_model({0: "person", 2: "car"})
        detections = make_detections(
            [0.0,  0.0,  40.0, 40.0, 0.95, 0.0],   # person
            [50.0, 50.0, 90.0, 90.0, 0.80, 2.0],   # car
        )

        result = ObjectDetection.get_detected_objects(model, detections)

        assert "person" in result
        assert "car" in result
        assert len(result["person"]) == 1
        assert len(result["car"]) == 1

    def test_bounding_box_coordinates_are_floats(self):
        model = make_mock_model({0: "person"})
        detections = make_detections([10.0, 20.0, 50.0, 80.0, 0.9, 0.0])

        result = ObjectDetection.get_detected_objects(model, detections)

        for coord in result["person"][0]:
            assert isinstance(coord, float)

    def test_confidence_and_class_id_excluded_from_bbox(self):
        """Each bbox list should have exactly 4 elements (no conf or cls_id)."""
        model = make_mock_model({0: "person"})
        detections = make_detections([10.0, 20.0, 50.0, 80.0, 0.99, 0.0])

        result = ObjectDetection.get_detected_objects(model, detections)

        assert len(result["person"][0]) == 4

    def test_class_grouping_correct_with_three_classes(self):
        model = make_mock_model({0: "person", 1: "bicycle", 2: "car"})
        detections = make_detections(
            [0,  0,  10, 10, 0.9, 0.0],
            [10, 10, 20, 20, 0.8, 1.0],
            [20, 20, 30, 30, 0.7, 2.0],
            [30, 30, 40, 40, 0.6, 0.0],  # second person
        )

        result = ObjectDetection.get_detected_objects(model, detections)

        assert len(result["person"]) == 2
        assert len(result["bicycle"]) == 1
        assert len(result["car"]) == 1


# ---------------------------------------------------------------------------
# ObjectDetection.__init__ (mocked torch.hub.load to avoid network access)
# ---------------------------------------------------------------------------

class TestObjectDetectionInit:
    @patch("object_detection.torch.hub.load")
    def test_default_device_is_cpu_when_no_cuda(self, mock_hub_load):
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_hub_load.return_value = mock_model

        with patch("object_detection.torch.cuda.is_available", return_value=False):
            od = ObjectDetection(model_name="yolov5s")

        assert od.device.type == "cpu"

    @patch("object_detection.torch.hub.load")
    def test_explicit_device_is_respected(self, mock_hub_load):
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_hub_load.return_value = mock_model

        device = torch.device("cpu")
        od = ObjectDetection(device=device)

        assert od.device == device

    @patch("object_detection.torch.hub.load")
    def test_model_loaded_from_ultralytics(self, mock_hub_load):
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_hub_load.return_value = mock_model

        ObjectDetection(model_name="yolov5s")

        mock_hub_load.assert_called_once_with("ultralytics/yolov5", "yolov5s")


# ---------------------------------------------------------------------------
# detect_objects (mocked model)
# ---------------------------------------------------------------------------

class TestDetectObjects:
    @patch("object_detection.torch.hub.load")
    def test_detect_objects_returns_xyxy_tensor(self, mock_hub_load):
        expected = torch.tensor([[10.0, 20.0, 50.0, 80.0, 0.9, 0.0]])

        mock_inner = MagicMock()
        mock_inner.to = MagicMock(return_value=mock_inner)
        mock_results = MagicMock()
        mock_results.xyxy = [expected]
        mock_inner.return_value = mock_results
        mock_hub_load.return_value = mock_inner

        od = ObjectDetection()
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = od.detect_objects(frame)

        assert torch.equal(result, expected)
