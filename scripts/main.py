import cv2
import torch

from object_detection import ObjectDetection
from depth_estimation import DepthEstimator
from close_person_analyzer import ClosePersonAnalyzer
from viewer import Viewer


# Path to input video
VIDEO_PATH = "/workspace/close_person_detection/media/video.mp4"

# Objects that appear closer than this depth threshold are considered too close
DEPTH_LIMIT = 0.011


def main():
    """
    Main function to perform real-time close-person detection using:
      - YOLOv5 for object detection
      - Depth Anything V2 for depth estimation
      - A visual analyzer for proximity alerts
      - An interactive viewer for playback and visualization
    """
    # Select computation device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize core components
    object_detection = ObjectDetection(device=device)
    depth_estimator = DepthEstimator(device=device)
    analyzer = ClosePersonAnalyzer(object_name="person", depth_limit=DEPTH_LIMIT)
    viewer = Viewer()

    # Open video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    print("Processing video... Press 'q' to quit or 'p' to pause/resume.")

    with torch.no_grad():  # Disable gradient computation for efficiency
        while True:
            # Handle pause mode (via GUI button or 'p' key)
            if viewer.paused:
                if cv2.waitKey(500) & 0xFF == ord("p"):
                    viewer.paused = not viewer.paused
                continue

            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # Estimate depth map for the current frame
            depth_pred, depth_vis_color = depth_estimator.estimate_depth(frame)

            # Perform object detection using YOLO
            detections = object_detection.detect_objects(frame)

            # Analyze detected persons for proximity alerts
            frame, border_color, border_size, border_message = analyzer.analyze(
                frame, depth_pred, detections, object_detection
            )

            # Add visual alert border with message (STOP / CAUTION)
            frame = viewer.add_border(frame, border_color, border_size, border_message)

            # Combine detection and depth frames side by side
            combined = viewer.combine_frames(frame, depth_vis_color)

            # Display combined frame with control button
            viewer.show_frame(combined)

            # Keyboard controls
            key = cv2.waitKey(500) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                viewer.paused = not viewer.paused

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")


if __name__ == "__main__":
    main()
