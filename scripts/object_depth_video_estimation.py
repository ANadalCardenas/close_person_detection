import cv2
import torch
import numpy as np

# ------------------------------
# Configuration
# ------------------------------
VIDEO_PATH = "/workspace/depth_estimation/media/video.mp4"  # Input video path
MIDAS_MODEL_TYPE = "DPT_Hybrid"                             # MiDaS model: "DPT_Hybrid", "DPT_Large", "MiDaS_small"
OBJECT = "person"


def get_detected_objects(model, detections):
    objects_dict ={}
    # Iterate over detections
    for *box, conf, cls_id in detections.tolist():
        class_name = model.names[int(cls_id)]
        box_coords = [float(x) for x in box]  # [xmin, ymin, xmax, ymax
        # Add box to the corresponding class in the dictionary
        if class_name not in objects_dict:
            objects_dict[class_name] = []
        objects_dict[class_name].append(box_coords)
    return objects_dict
    
def main():    
    # ------------------------------
    # Device (GPU if available)
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------
    # Load MiDaS model and transforms
    # ------------------------------
    midas = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE)
    midas.to(device).eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform  # Important: use default_transform correctly

    # ------------------------------
    # Load Yolov5 model to detect OBJECT
    # ------------------------------
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # ------------------------------
    # Open video
    # ------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    print("Processing video... Press 'q' to quit.")

    # ------------------------------
    # Process video frame by frame
    # ------------------------------
    with torch.no_grad():  # Disable gradients for inference
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            # ------------------------------
            # Makes a video in which all objects will appear with different colour depending on their depth
            # ------------------------------
            # Convert BGR (OpenCV) to RGB (MiDaS expects RGB)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply MiDaS transform and add batch dimension
            input_batch = transform(img_rgb).to(device)  # Shape: [1, 3, H, W]


            # Predict depth
            depth_pred = midas(input_batch)

            # Resize depth map to original frame size
            depth_pred = torch.nn.functional.interpolate(
                depth_pred.unsqueeze(1),             # Add channel dimension for interpolation
                size=img_rgb.shape[:2],              # Resize to original frame height and width
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()                # Remove extra dims and move to CPU

            # Normalize depth for visualization
            depth_vis = cv2.normalize(depth_pred, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis_color = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_MAGMA)    

            # ------------------------------
            # Makes a video in which the detected objects appear with their estimated depths.
            # ------------------------------
            results = model(frame)
            detections = results.xyxy[0]
            # Gets the information of detected objects
            objects_dict = get_detected_objects(model, detections)

            if OBJECT in objects_dict.keys():
                for bbox in objects_dict[OBJECT]:
                    # Convert BGR (OpenCV) to RGB (MiDaS expects RGB)
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Apply MiDaS transform and add batch dimension
                    input_batch = transform(img_rgb).to(device)  # Shape: [1, 3, H, W]
                    # Predict depth
                    depth_pred = midas(input_batch)

                    # Resize depth map to original frame size
                    depth_pred = torch.nn.functional.interpolate(
                        depth_pred.unsqueeze(1),             # Add channel dimension for interpolation
                        size=img_rgb.shape[:2],              # Resize to original frame height and width
                        mode="bicubic",
                        align_corners=False
                    ).squeeze().cpu().numpy()                # Remove extra dims and move to CPU

                    xmin, ymin, xmax, ymax = map(int, bbox)
                    object_depth = depth_pred[ymin:ymax, xmin:xmax]
                    median_depth = np.median(object_depth)

                    # Normalize depth for visualization
                    depth_vis = cv2.normalize(depth_pred, None, 0, 255, cv2.NORM_MINMAX)

                    # Draw bounding box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    # Prepare label
                    label = f"{median_depth:.1f} m"
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    text_x = xmin
                    text_y = max(ymin - 10, text_height + 10)

                    # Draw background rectangle and text
                    cv2.rectangle(frame,
                                  (text_x, text_y - text_height - baseline),
                                  (text_x + text_width, text_y + baseline),
                                  (0, 255, 0),
                                  cv2.FILLED)
                    cv2.putText(frame, label, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    # Combine the two frames side by side ---
                    combined = np.hstack((frame, depth_vis_color))
                    cv2.imshow("Depth Detected Person (Left)| Depth Map (Right)", combined)
            else:
                # Combine the two frames side by side ---
                combined = np.hstack((frame, depth_vis_color))
                cv2.imshow("Depth Detected Person (Left)| Depth Map (Right)", combined)

            # Press 'q' to exit early
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")

if __name__ == "__main__":
    main()