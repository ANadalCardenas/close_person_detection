import cv2
import torch

from object_detection import ObjectDetection
from depth_estimation import DepthEstimator
from close_person_estimation import ClosePersonAnalyzer
from viewer import Viewer

VIDEO_PATH = "/workspace/close_person_detection/media/video.mp4"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize modules
    yolo = ObjectDetection(device=device)
    midas = DepthEstimator(device=device)
    analyzer = ClosePersonAnalyzer(object_name="person", depth_limit=500)
    viewer = Viewer()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    print("Processing video... Press 'q' to quit.")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Depth estimation
            depth_pred, depth_vis_color = midas.estimate_depth(frame)

            # Object detection
            detections = yolo.detect_objects(frame)

            # Combine depth + detection info
            frame, border_color, border_size, border_message = analyzer.analyze(frame, depth_pred, detections, yolo)
            frame = viewer.add_border(frame, border_color, border_size, border_message)


            # Display
            frame = viewer.add_border(frame, border_color, border_size)
            combined = viewer.combine_frames(frame, depth_vis_color)
            viewer.show_frame("Detected Person (Left) | Depth Map (Right)", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")


if __name__ == "__main__":
    main()
