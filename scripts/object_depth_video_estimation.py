import cv2
import torch
import numpy as np

# ==============================================================
# Configuration
# ==============================================================
VIDEO_PATH = "/workspace/depth_estimation/media/video.mp4"  # Input video path
MIDAS_MODEL_TYPE = "DPT_Hybrid"  # Options: "DPT_Hybrid", "DPT_Large", "MiDaS_small"
OBJECT = "person"  # Object class to detect and measure depth for


# ==============================================================
# Helper Function
# ==============================================================
def get_detected_objects(model, detections):
    """
    Organize YOLO detections into a dictionary by class name.

    Args:
        model: YOLOv5 model (contains class names).
        detections: torch tensor of detections (xyxy, conf, class_id).

    Returns:
        dict: {class_name: [[x_min, y_min, x_max, y_max], ...]}
    """
    objects_dict = {}

    for *box, conf, cls_id in detections.tolist():
        class_name = model.names[int(cls_id)]
        box_coords = [float(x) for x in box]  # [xmin, ymin, xmax, ymax]

        if class_name not in objects_dict:
            objects_dict[class_name] = []
        objects_dict[class_name].append(box_coords)

    return objects_dict


# ==============================================================
# Main Function
# ==============================================================
def main():
    # ------------------------------
    # Device configuration
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------
    # Load MiDaS model and transforms
    # ------------------------------
    midas = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE)
    midas.to(device).eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform  # MiDaS expects RGB + specific normalization

    # ------------------------------
    # Load YOLOv5 model
    # ------------------------------
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    # ------------------------------
    # Open video file
    # ------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    print("Processing video... Press 'q' to quit.")

    # ------------------------------
    # Frame-by-frame processing
    # ------------------------------
    with torch.no_grad():  # Disable gradients for faster inference
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Convert BGR (OpenCV) → RGB (MiDaS expects RGB)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply MiDaS transform and add batch dimension
            input_batch = transform(img_rgb).to(device)

            # Predict depth map
            depth_pred = midas(input_batch)

            # Resize depth map to original frame size
            depth_pred = torch.nn.functional.interpolate(
                depth_pred.unsqueeze(1),  # Add channel dimension
                size=img_rgb.shape[:2],   # Match original HxW
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()

            # Normalize depth for visualization
            depth_vis = cv2.normalize(depth_pred, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis_color = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_MAGMA)

            # ------------------------------
            # Detect objects with YOLO
            # ------------------------------
            results = model(frame)
            detections = results.xyxy[0]

            objects_dict = get_detected_objects(model, detections)

            # ------------------------------
            # Draw detections for the target object
            # ------------------------------
            if OBJECT in objects_dict:
                for bbox in objects_dict[OBJECT]:
                    xmin, ymin, xmax, ymax = map(int, bbox)

                    # Extract depth region for the detected object
                    object_depth = depth_pred[ymin:ymax, xmin:xmax]
                    median_depth = np.median(object_depth)

                    # Draw bounding box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    # Label text (median depth)
                    label = f"{median_depth:.1f} m"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    text_x = xmin
                    text_y = max(ymin - 10, text_height + 10)

                    # Draw label background
                    cv2.rectangle(
                        frame,
                        (text_x, text_y - text_height - baseline),
                        (text_x + text_width, text_y + baseline),
                        (0, 255, 0),
                        cv2.FILLED
                    )

                # Draw label text
                cv2.putText(
                    frame, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
                )
                border_size = 20
                frame = cv2.copyMakeBorder(
                    frame,
                    top=border_size,
                    bottom=border_size,
                    left=border_size,
                    right=border_size,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 255)  # Red color
                )    
            else:
                # If there are not objects close, the border will be green
                border_size = 20
                frame = cv2.copyMakeBorder(
                    frame,
                    top=border_size,
                    bottom=border_size,
                    left=border_size,
                    right=border_size,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, 255, 0)  # Green color ()
                )    

            # ------------------------------
            # Combine original + depth frames
            # ------------------------------
            # Ensure both have the same height
            if frame.shape != depth_vis_color.shape:
                depth_vis_color = cv2.resize(depth_vis_color, (frame.shape[1], frame.shape[0]))
            combined = np.hstack((frame, depth_vis_color))
            cv2.imshow("Detected Person (Left) | Depth Map (Right)", combined)

            # Exit with 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ------------------------------
    # Cleanup
    # ------------------------------
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")


# ==============================================================
# Entry Point
# ==============================================================
if __name__ == "__main__":
    main()
