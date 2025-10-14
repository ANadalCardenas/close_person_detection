import cv2
import numpy as np

class Viewer:
    @staticmethod
    def add_border(frame, color, size):
        """Add colored border around the frame."""
        return cv2.copyMakeBorder(
            frame, size, size, size, size,
            borderType=cv2.BORDER_CONSTANT,
            value=color
        )

    @staticmethod
    def combine_frames(frame, depth_color):
        """Combine original and depth visualization side-by-side."""
        if frame.shape != depth_color.shape:
            depth_color = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))
        return np.hstack((frame, depth_color))

    @staticmethod
    def show_frame(window_name, combined_frame):
        cv2.imshow(window_name, combined_frame)
