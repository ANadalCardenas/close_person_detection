import numpy as np
import cv2

class ClosePersonAnalyzer:
    """
    Analyze detected persons in a frame to determine their proximity 
    using a depth map, and return visual cues (colored border and message).
    """

    def __init__(self, object_name="person", depth_limit=0.011):
        """
        Initialize the analyzer with the object name to track 
        and the threshold for proximity alert.
        
        Parameters:
            object_name (str): Type of object to analyze (default: "person").
            depth_limit (float): Threshold for how close an object can be 
                                 before triggering an alert.
        """
        self.object_name = object_name
        self.depth_limit = depth_limit

    def analyze(self, frame, depth_map, detections, model):
        """
        Draw bounding boxes for detected objects and determine 
        alert color and message based on proximity.
        
        Parameters:
            frame (np.ndarray): The RGB video frame.
            depth_map (np.ndarray): Depth map corresponding to the frame.
            detections: Object detection results from the model.
            model: The detection model providing object data.
        
        Returns:
            frame (np.ndarray): The annotated frame.
            border_color (tuple): RGB color for the alert border.
            border_size (int): Thickness of the alert border.
            border_message (str): Optional alert message ("STOP" or "CAUTION").
        """
        # Get dictionary of detected objects from the model
        objects_dict = model.get_detected_objects(model.model, detections)

        # Default border: green (safe)
        border_color = (0, 255, 0)
        border_size = 20
        border_message = ""

        # Check if the target object (e.g., "person") was detected
        if self.object_name in objects_dict:
            for bbox in objects_dict[self.object_name]:
                xmin, ymin, xmax, ymax = map(int, bbox)

                # Extract the depth values for the detected region
                object_depth = depth_map[ymin:ymax, xmin:xmax]

                # Calculate the median depth using inverse depth values.
                # Add a small epsilon to prevent division by zero.
                eps = 1e-10
                median_depth = 1.0 / (np.median(object_depth) + eps)

                # Draw the bounding box around the detected object
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Determine proximity and set color/message accordingly
                if median_depth < self.depth_limit:
                    border_color = (0, 0, 255)   # Red: too close
                    border_message = "STOP"
                elif median_depth < self.depth_limit * 1.2:
                    border_color = (0, 140, 255)  # Orange: caution
                    border_message = "CAUTION"

        return frame, border_color, border_size, border_message
