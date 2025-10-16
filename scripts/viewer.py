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
            # Calculate position to center the message
            (text_w, text_h), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)
            text_x = (frame_with_border.shape[1] - text_w) // 2
            text_y = size + text_h + 10  # Slight offset from top border

            # Draw message in black
            cv2.putText(frame_with_border, message, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5, cv2.LINE_AA)

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
