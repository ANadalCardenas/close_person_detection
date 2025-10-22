import cv2
import numpy as np

class Viewer:
    def __init__(self):
        self.paused = False
        self.button_region = None
        self.window_name = "Detected Person (Left) | Depth Map (Right)"

        # Register mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_event)

    def _mouse_event(self, event, x, y, flags, param):
        """Toggle pause if user clicks inside button region."""
        if event == cv2.EVENT_LBUTTONDOWN and self.button_region is not None:
            x1, y1, x2, y2 = self.button_region
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.paused = not self.paused
                cv2.imshow(self.window_name, self._draw_button(self.last_frame.copy()))

    @staticmethod
    def add_border(frame, color, size, message=""):
        """Add colored border around the frame with optional message."""
        frame_with_border = cv2.copyMakeBorder(
            frame, size, size, size, size,
            borderType=cv2.BORDER_CONSTANT,
            value=color
        )
        if message:
            # Draw text **inside** the top border
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_thickness = 2
            text_color = (0, 0, 0)  # black

            text_size = cv2.getTextSize(message, font, font_scale, font_thickness)[0]
            text_x = (frame_with_border.shape[1] - text_size[0]) // 2
            text_y = size - (size - text_size[1]) // 2  # vertically centered inside top border

            cv2.putText(
            frame_with_border,
            message,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
)
        return frame_with_border

    def _draw_button(self, frame):
        """Draw play/pause icon button at the bottom center between both frames."""
        h, w, _ = frame.shape
        btn_size = 60
        margin_bottom = 20

        # Button coordinates (centered horizontally)
        x1 = w // 2 - btn_size // 2
        y1 = h - btn_size - margin_bottom
        x2 = x1 + btn_size
        y2 = y1 + btn_size
        self.button_region = (x1, y1, x2, y2)

        
        # Draw white button background
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
        # Draw icon instead of text
        icon_color = (0, 0, 0)  # black
        # Draw icon (▶ or ⏸)
        center_x, center_y = x1 + btn_size // 2, y1 + btn_size // 2
        
        if self.paused:
            # ▶ Play icon (triangle)
            pts = np.array([
                [center_x - 10, center_y - 15],
                [center_x - 10, center_y + 15],
                [center_x + 15, center_y]
            ], np.int32)
            cv2.fillPoly(frame, [pts], icon_color)
        else:
            # ⏸ Pause icon (two bars)
            cv2.rectangle(frame, (center_x - 12, center_y - 15), (center_x - 4, center_y + 15), (0, 0, 255), -1)
            cv2.rectangle(frame, (center_x + 4, center_y - 15), (center_x + 12, center_y + 15), (0, 0, 255), -1)

        return frame
    

    @staticmethod
    def combine_frames(frame, depth_color):
        """Combine original and depth visualization side-by-side."""
        if frame.shape != depth_color.shape:
            depth_color = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))
        return np.hstack((frame, depth_color))

    def show_frame(self, combined_frame):
        """Show combined frame with play/pause button."""
        self.last_frame = combined_frame  # save last shown frame
        frame_with_button = self._draw_button(combined_frame)
        cv2.imshow(self.window_name, frame_with_button)