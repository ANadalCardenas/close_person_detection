import numpy as np
import cv2

class ClosePersonAnalyzer:
    def __init__(self, object_name="person", depth_limit=90):
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
                median_depth = np.max(object_depth)
                label = f"{median_depth:.1f} m"

                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Draw label above bounding box
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (xmin, ymin - th - baseline), (xmin + tw, ymin), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (xmin, ymin - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Change border color and set message if too close
                if median_depth > self.depth_limit:
                    border_color = (0, 0, 255)   # red
                    border_message = "STOP"
                elif median_depth > self.depth_limit * 1.2:
                    border_color = (0, 140, 255)  # orange
                    border_message = "CAUTION"

        return frame, border_color, border_size, border_message
