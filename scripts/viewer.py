import cv2
import numpy as np

class Viewer:
    @staticmethod
    def add_border(frame, color, size, message=""):
        """Add colored border around the frame with optional message."""
        frame_with_border = cv2.copyMakeBorder(
            frame, size, size, size, size,
            borderType=cv2.BORDER_CONSTANT,
            value=color
        )
        if message:
            # Draw message at top center of border
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
            text_x = (frame_with_border.shape[1] - text_size[0]) // 2
            text_y = text_size[1] + 10
            cv2.putText(frame_with_border, message, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        return frame_with_border

    @staticmethod
    def combine_frames(frame, depth_color):
        """Combine original and depth visualization side-by-side."""
        if frame.shape != depth_color.shape:
            depth_color = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))
        return np.hstack((frame, depth_color))

    @staticmethod
    def show_frame(window_name, combined_frame):
        cv2.imshow(window_name, combined_frame)
