
import cv2
import torch
import numpy as np

# ------------------------------
# Configuration
# ------------------------------
VIDEO_PATH = "/workspace/depth_estimation/media/video.mp4"  # Input video path
MIDAS_MODEL_TYPE = "DPT_Hybrid"                             # MiDaS model: "DPT_Hybrid", "DPT_Large", "MiDaS_small"

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
    # Initialize webcam
    # ------------------------------
    cap = cv2.VideoCapture(0)
    # ------------------------------
    # Process video webcamframe by frame
    # ------------------------------
    with torch.no_grad():  # Disable gradients for inference
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video
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
            depth_vis_color = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_PLASMA)
            # Show the depth video
            cv2.imshow("Depth Map", depth_vis_color)
            # Press 'q' to exit early
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
