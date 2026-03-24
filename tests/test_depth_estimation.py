import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from depth_estimation import DepthEstimator


def make_bgr_frame(h=120, w=160):
    """Create a random BGR frame as OpenCV would produce."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def make_pipeline_mock(depth_array):
    """
    Return a mock HF pipeline callable that returns {'depth': depth_array}.
    """
    mock_pipe = MagicMock(return_value={"depth": depth_array})
    return mock_pipe


# ---------------------------------------------------------------------------
# DepthEstimator.__init__
# ---------------------------------------------------------------------------

class TestInit:
    @patch("depth_estimation.pipeline")
    def test_pipeline_created_with_correct_task(self, mock_pipeline):
        DepthEstimator(model_name="some/model", device=None)
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args
        assert call_kwargs[1]["task"] == "depth-estimation" or call_kwargs[0][0] == "depth-estimation"

    @patch("depth_estimation.pipeline")
    def test_default_model_name_used(self, mock_pipeline):
        DepthEstimator()
        call_args = mock_pipeline.call_args
        # model_name appears as keyword arg
        assert "depth-anything/Depth-Anything-V2-Small-hf" in str(call_args)


# ---------------------------------------------------------------------------
# estimate_depth — output shapes and types
# ---------------------------------------------------------------------------

class TestEstimateDepth:
    def _make_estimator(self, depth_output):
        """Create a DepthEstimator whose internal pipeline is mocked."""
        with patch("depth_estimation.pipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value = make_pipeline_mock(depth_output)
            estimator = DepthEstimator(model_name="test/model", device=None)
        return estimator

    # -- numpy array depth output --

    def test_output_shapes_match_input_frame(self):
        h, w = 120, 160
        frame = make_bgr_frame(h, w)
        small_depth = np.random.rand(60, 80).astype(np.float32)   # smaller than frame

        estimator = self._make_estimator(small_depth)
        depth_resized, depth_color = estimator.estimate_depth(frame)

        assert depth_resized.shape == (h, w)
        assert depth_color.shape == (h, w, 3)

    def test_depth_resized_is_float(self):
        frame = make_bgr_frame()
        depth = np.ones((30, 40), dtype=np.float64)

        estimator = self._make_estimator(depth)
        depth_resized, _ = estimator.estimate_depth(frame)

        assert depth_resized.dtype in (np.float32, np.float64)

    def test_depth_color_is_uint8_bgr(self):
        frame = make_bgr_frame()
        depth = np.random.rand(60, 80).astype(np.float32)

        estimator = self._make_estimator(depth)
        _, depth_color = estimator.estimate_depth(frame)

        assert depth_color.dtype == np.uint8
        assert depth_color.ndim == 3
        assert depth_color.shape[2] == 3

    def test_same_size_depth_does_not_change_shape(self):
        h, w = 100, 150
        frame = make_bgr_frame(h, w)
        depth = np.random.rand(h, w).astype(np.float32)

        estimator = self._make_estimator(depth)
        depth_resized, depth_color = estimator.estimate_depth(frame)

        assert depth_resized.shape == (h, w)

    # -- list-wrapped output (some pipeline versions) --

    def test_handles_list_wrapped_output(self):
        h, w = 80, 120
        frame = make_bgr_frame(h, w)
        depth = np.random.rand(40, 60).astype(np.float32)

        with patch("depth_estimation.pipeline") as mock_pipeline_cls:
            # Pipeline returns a list containing the dict
            mock_pipeline_cls.return_value = MagicMock(return_value=[{"depth": depth}])
            estimator = DepthEstimator(model_name="test/model", device=None)

        depth_resized, depth_color = estimator.estimate_depth(frame)

        assert depth_resized.shape == (h, w)

    # -- PIL Image depth output --

    def test_handles_pil_image_depth_output(self):
        h, w = 80, 100
        frame = make_bgr_frame(h, w)
        pil_depth = Image.fromarray(np.random.randint(0, 256, (40, 50), dtype=np.uint8))

        estimator = self._make_estimator(pil_depth)
        depth_resized, depth_color = estimator.estimate_depth(frame)

        assert depth_resized.shape == (h, w)

    # -- Missing 'depth' key raises RuntimeError --

    def test_missing_depth_key_raises_runtime_error(self):
        frame = make_bgr_frame()

        with patch("depth_estimation.pipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value = MagicMock(return_value={"predicted_depth": None})
            estimator = DepthEstimator(model_name="test/model", device=None)

        with pytest.raises(RuntimeError, match="depth"):
            estimator.estimate_depth(frame)

    # -- Colour map range --

    def test_depth_color_values_in_0_255_range(self):
        frame = make_bgr_frame()
        depth = np.random.rand(60, 80).astype(np.float32) * 1000   # large values

        estimator = self._make_estimator(depth)
        _, depth_color = estimator.estimate_depth(frame)

        assert depth_color.min() >= 0
        assert depth_color.max() <= 255

    def test_uniform_depth_map_produces_valid_output(self):
        """A completely flat depth map should not cause errors."""
        frame = make_bgr_frame()
        depth = np.ones((60, 80), dtype=np.float32) * 5.0

        estimator = self._make_estimator(depth)
        depth_resized, depth_color = estimator.estimate_depth(frame)

        assert depth_resized is not None
        assert depth_color is not None
