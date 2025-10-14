import torch
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, model_type="DPT_Hybrid", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading MiDaS ({model_type}) on {self.device}...")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device).eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.default_transform

    def estimate_depth(self, frame):
        """Return predicted depth and colored visualization."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            depth_pred = self.model(input_batch)
            depth_pred = torch.nn.functional.interpolate(
                depth_pred.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()

        depth_vis = cv2.normalize(depth_pred, None, 0, 255, cv2.NORM_MINMAX)
        depth_color = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_MAGMA)
        return depth_pred, depth_color
