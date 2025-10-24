import cv2
import torch

from object_detection import ObjectDetection
from depth_estimation import DepthEstimator
from close_person_estimation import ClosePersonAnalyzer
from viewer import Viewer

VIDEO_PATH = "/workspace/close_person_detection/media/video.mp4"
# The objects that appear so close are at lass than 0.015 "units"
DEPTH_LIMIT = 0.015

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize modules
    yolo = ObjectDetection(device=device)
    depth_estimator = DepthEstimator(device=device)
    analyzer = ClosePersonAnalyzer(object_name="person", depth_limit=DEPTH_LIMIT)
    viewer = Viewer()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    print("Processing video... Press 'q' to quit.")

    with torch.no_grad():
        while True:
            if viewer.paused:
                # Wait while paused
                if cv2.waitKey(30) & 0xFF == ord('p'):
                    viewer.paused = not viewer.paused
                continue

            ret, frame = cap.read()
            if not ret:
                break

            # Depth estimation
            depth_pred, depth_vis_color = depth_estimator.estimate_depth(frame)

            # Object detection
            detections = yolo.detect_objects(frame)

            # Combine depth + detection info
            frame, border_color, border_size, border_message = analyzer.analyze(frame, depth_pred, detections, yolo)
            frame = viewer.add_border(frame, border_color, border_size, border_message)


            # Display
            frame = viewer.add_border(frame, border_color, border_size)
            combined = viewer.combine_frames(frame, depth_vis_color)
            viewer.show_frame(combined)

            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):  # toggle with keyboard
                viewer.paused = not viewer.paused

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")


if __name__ == "__main__":
    main()
