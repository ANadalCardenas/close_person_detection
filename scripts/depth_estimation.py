import torch
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

class DepthEstimator:
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf", device=None):
        """
        Uses the HuggingFace pipeline for depth-estimation (Depth Anything V2).
        model_name: HF model id (small/base/large variants may exist)
        device: torch.device or None -> automatically set to GPU if available.
        """
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        # pipeline device argument takes -1 for CPU, 0..n for CUDA devices
        hf_device = 0 if torch.cuda.is_available() else -1
        print(f"Loading Depth Anything V2 model '{model_name}' on device: {self.device} ...")
        # create pipeline
        self.pipe = pipeline(task="depth-estimation", model=model_name, device=hf_device)
        


    def estimate_depth(self, frame):
        """
        Input: frame (BGR numpy array as from OpenCV)
        Returns:
          - depth_pred: numpy array (float) same HxW as frame (relative or metric depending on model)
          - depth_color: colorized uint8 visualization for display (BGR)
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Run pipeline: returns dict with key "depth" (as numpy array or torch tensor)
        result = self.pipe(pil_img)
        # result may be a list or dict depending on transformers version; handle both:
        if isinstance(result, list):
            result = result[0]

        depth = result.get("depth", None)
        if depth is None:
            raise RuntimeError("Depth pipeline did not return 'depth'. Check transformers version and model.")

        # Ensure numpy array
        if not isinstance(depth, np.ndarray):
            # sometimes it's a PIL image or torch tensor
            try:
                depth = np.array(depth)
            except Exception:
                try:
                    depth = depth.detach().cpu().numpy()
                except Exception as e:
                    raise RuntimeError(f"Cannot convert depth to numpy array: {e}")

        # depth might be smaller resolution — resize to original frame size
        depth_resized = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        # create a visualization: normalize to 0..255 and color map
        depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

        return depth_resized, depth_color
