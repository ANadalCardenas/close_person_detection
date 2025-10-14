import numpy as np
import cv2

class ClosePersonAnalyzer:
    def __init__(self, object_name="person", depth_limit=500):
        self.object_name = object_name
        self.depth_limit = depth_limit

    def analyze(self, frame, depth_map, detections, model):
        """Draw boxes and determine alert color + border thickness."""
        objects_dict = model.get_detected_objects(model.model, detections)
        border_color = (0, 255, 0)  # green
        border_size = 20
        label = ""

        if self.object_name in objects_dict:
            for bbox in objects_dict[self.object_name]:
                xmin, ymin, xmax, ymax = map(int, bbox)
                object_depth = depth_map[ymin:ymax, xmin:xmax]
                median_depth = np.median(object_depth)
                label = f"{median_depth:.1f} m"

                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Draw label
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (xmin, ymin - th - baseline), (xmin + tw, ymin), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (xmin, ymin - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Change border color depending on distance
                if median_depth < self.depth_limit:
                    border_color = (0, 0, 255)
                    border_size = 30
                else:
                    border_color = (255, 165, 0)

        return frame, border_color, border_size
