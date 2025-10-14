import torch

class ObjectDetection:
    def __init__(self, model_name="yolov5s", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading YOLO model on {self.device}...")
        self.model = torch.hub.load("ultralytics/yolov5", model_name).to(self.device)

    def detect_objects(self, frame):
        """Run YOLO inference and return detections."""
        results = self.model(frame)
        return results.xyxy[0]

    @staticmethod
    def get_detected_objects(model, detections):
        """Organize YOLO detections into dict by class name."""
        objects_dict = {}
        for *box, conf, cls_id in detections.tolist():
            class_name = model.names[int(cls_id)]
            box_coords = [float(x) for x in box]
            objects_dict.setdefault(class_name, []).append(box_coords)
        return objects_dict
