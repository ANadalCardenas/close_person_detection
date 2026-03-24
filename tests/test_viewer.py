import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
import cv2
import pytest
from unittest.mock import patch, MagicMock

from viewer import Viewer


def make_frame(h=200, w=300, channels=3):
    return np.zeros((h, w, channels), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Viewer instantiation (mock cv2 calls that require a display)
# ---------------------------------------------------------------------------

@pytest.fixture
def viewer():
    with patch("viewer.cv2.namedWindow"), patch("viewer.cv2.setMouseCallback"):
        v = Viewer()
    return v


# ---------------------------------------------------------------------------
# add_border (static method — no display required)
# ---------------------------------------------------------------------------

class TestAddBorder:
    def test_border_increases_frame_size(self):
        frame = make_frame(100, 200)
        result = Viewer.add_border(frame, color=(0, 255, 0), size=10)

        assert result.shape[0] == 100 + 10 * 2   # top + bottom
        assert result.shape[1] == 200 + 10 * 2   # left + right

    def test_border_color_is_applied(self):
        frame = make_frame(50, 50)
        color = (0, 0, 255)   # red in BGR
        result = Viewer.add_border(frame, color=color, size=5)

        # Top-left pixel should be the border color
        np.testing.assert_array_equal(result[0, 0], color)

    def test_no_message_does_not_add_text(self):
        frame = make_frame(80, 80)
        without_text = Viewer.add_border(frame, (0, 255, 0), 10, message="")
        with_text = Viewer.add_border(frame, (0, 255, 0), 10, message="STOP")

        # Frame with text should differ somewhere from frame without text
        assert not np.array_equal(without_text, with_text)

    def test_with_stop_message_returns_same_shape_as_without(self):
        frame = make_frame(100, 100)
        size = 20
        r1 = Viewer.add_border(frame, (0, 0, 255), size, "STOP")
        r2 = Viewer.add_border(frame, (0, 0, 255), size)

        assert r1.shape == r2.shape

    def test_border_size_zero_returns_same_dimensions(self):
        frame = make_frame(60, 80)
        result = Viewer.add_border(frame, (0, 255, 0), 0)

        assert result.shape == frame.shape

    def test_caution_message_rendered(self):
        frame = make_frame(100, 200)
        result = Viewer.add_border(frame, (0, 140, 255), 30, "CAUTION")
        # Frame must be larger (border was added)
        assert result.shape[0] > frame.shape[0]


# ---------------------------------------------------------------------------
# combine_frames (static method)
# ---------------------------------------------------------------------------

class TestCombineFrames:
    def test_same_shape_frames_are_stacked_horizontally(self):
        a = np.ones((100, 200, 3), dtype=np.uint8) * 10
        b = np.ones((100, 200, 3), dtype=np.uint8) * 20

        result = Viewer.combine_frames(a, b)

        assert result.shape == (100, 400, 3)

    def test_combined_frame_contains_both_images(self):
        a = np.full((100, 100, 3), 10, dtype=np.uint8)
        b = np.full((100, 100, 3), 200, dtype=np.uint8)

        result = Viewer.combine_frames(a, b)

        np.testing.assert_array_equal(result[:, :100], a)
        np.testing.assert_array_equal(result[:, 100:], b)

    def test_different_size_depth_frame_is_resized(self):
        frame = make_frame(100, 200)
        depth = make_frame(50, 100)   # different size

        result = Viewer.combine_frames(frame, depth)

        # depth is resized to match frame; result width = frame.w + frame.w
        assert result.shape == (100, 400, 3)

    def test_result_dtype_unchanged(self):
        a = np.zeros((80, 80, 3), dtype=np.uint8)
        b = np.zeros((80, 80, 3), dtype=np.uint8)

        result = Viewer.combine_frames(a, b)

        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# _draw_button
# ---------------------------------------------------------------------------

class TestDrawButton:
    def test_button_region_is_set_after_draw(self, viewer):
        frame = make_frame(300, 400)
        viewer._draw_button(frame)

        assert viewer.button_region is not None
        assert len(viewer.button_region) == 4

    def test_button_is_centered_horizontally(self, viewer):
        frame = make_frame(300, 400)
        viewer._draw_button(frame)
        x1, y1, x2, y2 = viewer.button_region

        center_x = (x1 + x2) // 2
        assert abs(center_x - 400 // 2) <= 1   # within 1 pixel of frame center

    def test_pause_icon_drawn_when_playing(self, viewer):
        frame = make_frame(300, 400)
        viewer.paused = False
        result = viewer._draw_button(frame)
        # Frame must have been modified (button drawn)
        assert not np.all(result == 0)

    def test_play_icon_drawn_when_paused(self, viewer):
        frame_playing = make_frame(300, 400)
        frame_paused = make_frame(300, 400)

        viewer.paused = False
        playing = viewer._draw_button(frame_playing.copy())

        viewer.paused = True
        paused = viewer._draw_button(frame_paused.copy())

        # Icons should differ
        assert not np.array_equal(playing, paused)

    def test_draw_button_returns_frame(self, viewer):
        frame = make_frame(300, 400)
        result = viewer._draw_button(frame)
        assert result is frame


# ---------------------------------------------------------------------------
# _mouse_event
# ---------------------------------------------------------------------------

class TestMouseEvent:
    def test_click_inside_button_toggles_pause(self, viewer):
        viewer.button_region = (100, 100, 160, 160)
        viewer.paused = False

        with patch("viewer.cv2.imshow"):
            viewer._mouse_event(cv2.EVENT_LBUTTONDOWN, 130, 130, 0, None)

        assert viewer.paused is True

    def test_click_outside_button_does_not_toggle(self, viewer):
        viewer.button_region = (100, 100, 160, 160)
        viewer.paused = False

        viewer._mouse_event(cv2.EVENT_LBUTTONDOWN, 10, 10, 0, None)

        assert viewer.paused is False

    def test_no_button_region_does_not_crash(self, viewer):
        viewer.button_region = None
        viewer.paused = False

        viewer._mouse_event(cv2.EVENT_LBUTTONDOWN, 50, 50, 0, None)

        assert viewer.paused is False

    def test_non_click_event_does_not_toggle(self, viewer):
        viewer.button_region = (100, 100, 160, 160)
        viewer.paused = False

        viewer._mouse_event(cv2.EVENT_MOUSEMOVE, 130, 130, 0, None)

        assert viewer.paused is False

    def test_second_click_toggles_back_to_playing(self, viewer):
        viewer.button_region = (100, 100, 160, 160)
        viewer.paused = False

        with patch("viewer.cv2.imshow"):
            viewer._mouse_event(cv2.EVENT_LBUTTONDOWN, 130, 130, 0, None)
            viewer._mouse_event(cv2.EVENT_LBUTTONDOWN, 130, 130, 0, None)

        assert viewer.paused is False


# ---------------------------------------------------------------------------
# show_frame
# ---------------------------------------------------------------------------

class TestShowFrame:
    def test_show_frame_stores_last_frame(self, viewer):
        frame = make_frame(200, 400)
        with patch("viewer.cv2.imshow"):
            viewer.show_frame(frame)
        assert viewer.last_frame is frame

    def test_show_frame_calls_imshow(self, viewer):
        frame = make_frame(200, 400)
        with patch("viewer.cv2.imshow") as mock_imshow:
            viewer.show_frame(frame)
        mock_imshow.assert_called_once()
