import numpy as np
import cv2

class ClosePersonAnalyzer:
    def __init__(self, object_name="person", depth_limit=0.011):
        self.object_name = object_name
        self.depth_limit = depth_limit

    def analyze(self, frame, depth_map, detections, model):
        """Draw boxes and determine alert color and optional message."""
        objects_dict = model.get_detected_objects(model.model, detections)
        border_color = (0, 255, 0)  # green
        border_size = 20
        border_message = ""

        if self.object_name in objects_dict:
            for bbox in objects_dict[self.object_name]:
                xmin, ymin, xmax, ymax = map(int, bbox)
                object_depth = depth_map[ymin:ymax, xmin:xmax]
                # Calculate the inverse of depth because the Depth Everything model uses the inverse of depth.
                # Create an epsilon to avoid dividing by zero.
                eps = float(1e-10)
                median_depth = float(1)/float((np.median(object_depth) + eps))

                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Change border color and set message if too close
                if median_depth < self.depth_limit:
                    border_color = (0, 0, 255)   # red
                    border_message = "STOP"
                elif median_depth < self.depth_limit * 1.2:
                    border_color = (0, 140, 255)  # orange
                    border_message = "CAUTION"

        return frame, border_color, border_size, border_message
