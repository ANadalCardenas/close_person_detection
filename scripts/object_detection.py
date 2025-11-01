import torch


class ObjectDetection:
    """
    Object detection using YOLOv5 from the Ultralytics repository.
    """

    def __init__(self, model_name="yolov5s", device=None):
        """
        Initialize the YOLOv5 model for object detection.

        Parameters:
            model_name (str): YOLOv5 model variant to load (e.g., "yolov5s", "yolov5m").
            device (torch.device or None): Computation device.
                                           If None, automatically selects CUDA if available.
        """
        # Automatically select GPU if available, otherwise use CPU
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading YOLOv5 model '{model_name}' on device: {self.device} ...")

        # Load the YOLOv5 model from the Ultralytics repository
        self.model = torch.hub.load("ultralytics/yolov5", model_name).to(self.device)

    def detect_objects(self, frame):
        """
        Run YOLO inference on a single frame.

        Parameters:
            frame (np.ndarray or list): Input image or batch of images.

        Returns:
            detections (torch.Tensor): Tensor of detections with format
                                       [xmin, ymin, xmax, ymax, confidence, class_id].
        """
        results = self.model(frame)
        return results.xyxy[0]

    @staticmethod
    def get_detected_objects(model, detections):
        """
        Organize YOLO detections into a dictionary grouped by class name.

        Parameters:
            model: The YOLO model (used to retrieve class names).
            detections (torch.Tensor): Tensor of detections from YOLO output.

        Returns:
            objects_dict (dict): Dictionary mapping class names to lists of bounding boxes.
                                 Example:
                                 {
                                     "person": [[xmin, ymin, xmax, ymax], ...],
                                     "car": [[xmin, ymin, xmax, ymax], ...]
                                 }
        """
        objects_dict = {}

        # Iterate over each detection and group by class name
        for *box, conf, cls_id in detections.tolist():
            class_name = model.names[int(cls_id)]
            box_coords = [float(x) for x in box]
            objects_dict.setdefault(class_name, []).append(box_coords)

        return objects_dict
