import cv2
import numpy as np


class Viewer:
    """
    Display and interact with video frames showing detected persons and depth maps.
    Provides pause/play functionality and visual alerts through borders and text.
    """

    def __init__(self):
        """
        Initialize the Viewer window and set up the mouse callback for interactivity.
        """
        self.paused = False
        self.button_region = None
        self.last_frame = None
        self.window_name = "Detected Person (Left) | Depth Map (Right)"

        # Create a named window and register a mouse event callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_event)

    def _mouse_event(self, event, x, y, flags, param):
        """
        Handle mouse click events to toggle the pause/play state.

        Parameters:
            event (int): OpenCV mouse event type.
            x (int): X-coordinate of the mouse event.
            y (int): Y-coordinate of the mouse event.
            flags (int): Event flags (not used).
            param: Optional user data (not used).
        """
        if event == cv2.EVENT_LBUTTONDOWN and self.button_region is not None:
            x1, y1, x2, y2 = self.button_region
            # Check if the click occurred inside the button area
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.paused = not self.paused
                if self.last_frame is not None:
                    cv2.imshow(self.window_name, self._draw_button(self.last_frame.copy()))

    @staticmethod
    def add_border(frame, color, size, message=""):
        """
        Add a colored border around the frame with an optional centered message.

        Parameters:
            frame (np.ndarray): Image to annotate.
            color (tuple): BGR color of the border.
            size (int): Thickness of the border.
            message (str): Optional message displayed on the top border.

        Returns:
            np.ndarray: Annotated frame with border and message.
        """
        # Add border around the frame
        frame_with_border = cv2.copyMakeBorder(
            frame, size, size, size, size,
            borderType=cv2.BORDER_CONSTANT,
            value=color
        )

        if message:
            # Draw text centered within the top border
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_thickness = 2
            text_color = (0, 0, 0)  # black

            text_size = cv2.getTextSize(message, font, font_scale, font_thickness)[0]
            text_x = (frame_with_border.shape[1] - text_size[0]) // 2
            text_y = size - (size - text_size[1]) // 2

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
        """
        Draw a play/pause button at the bottom center of the frame.

        Parameters:
            frame (np.ndarray): Frame to annotate.

        Returns:
            np.ndarray: Frame with button overlay.
        """
        h, w, _ = frame.shape
        btn_size = 60
        margin_bottom = 20

        # Compute button coordinates (centered horizontally)
        x1 = w // 2 - btn_size // 2
        y1 = h - btn_size - margin_bottom
        x2 = x1 + btn_size
        y2 = y1 + btn_size
        self.button_region = (x1, y1, x2, y2)

        # Draw white button background
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # Draw simple pause/play icons using shapes (no emoji)
        center_x, center_y = x1 + btn_size // 2, y1 + btn_size // 2
        icon_color = (0, 0, 0)  # black

        if self.paused:
            # Draw play icon (triangle)
            pts = np.array([
                [center_x - 10, center_y - 15],
                [center_x - 10, center_y + 15],
                [center_x + 15, center_y]
            ], np.int32)
            cv2.fillPoly(frame, [pts], icon_color)
        else:
            # Draw pause icon (two rectangles)
            cv2.rectangle(frame, (center_x - 12, center_y - 15), (center_x - 4, center_y + 15), icon_color, -1)
            cv2.rectangle(frame, (center_x + 4, center_y - 15), (center_x + 12, center_y + 15), icon_color, -1)

        return frame

    @staticmethod
    def combine_frames(frame, depth_color):
        """
        Combine two frames (original and depth map) side by side.

        Parameters:
            frame (np.ndarray): Original frame.
            depth_color (np.ndarray): Depth visualization frame.

        Returns:
            np.ndarray: Combined frame.
        """
        if frame.shape != depth_color.shape:
            depth_color = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))
        return np.hstack((frame, depth_color))

    def show_frame(self, combined_frame):
        """
        Display the combined frame with a play/pause button overlay.

        Parameters:
            combined_frame (np.ndarray): Frame to display.
        """
        self.last_frame = combined_frame
        frame_with_button = self._draw_button(combined_frame)
        cv2.imshow(self.window_name, frame_with_button)
