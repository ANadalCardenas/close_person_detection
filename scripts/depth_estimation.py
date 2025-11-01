import torch
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline


class DepthEstimator:
    """
    Estimate depth maps from RGB images using the Hugging Face 
    'Depth Anything V2' model.
    """

    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf", device=None):
        """
        Initialize the depth estimation pipeline.

        Parameters:
            model_name (str): Hugging Face model identifier. 
                              Small, base, and large variants may be available.
            device (torch.device or None): Device to load the model on.
                                           Automatically selects GPU if available.
        """
        print(f"Loading Depth Anything V2 model '{model_name}' on device: {device} ...")

        # Create the depth estimation pipeline
        self.pipe = pipeline(task="depth-estimation", model=model_name, device=device)

    def estimate_depth(self, frame):
        """
        Estimate depth for an input frame and return both raw and colorized results.

        Parameters:
            frame (np.ndarray): Input BGR image (e.g., from OpenCV).

        Returns:
            depth_resized (np.ndarray): Depth prediction as a float32 array (same HxW as input).
            depth_color (np.ndarray): Colorized depth map (uint8, BGR) for visualization.
        """
        # Convert BGR (OpenCV) image to RGB (expected by PIL/transformers)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Run the depth estimation pipeline
        result = self.pipe(pil_img)

        # Handle both possible output structures: dict or list of dicts
        if isinstance(result, list):
            result = result[0]

        # Extract depth map
        depth = result.get("depth", None)
        if depth is None:
            raise RuntimeError("Depth pipeline did not return 'depth'. Check transformers version and model output.")

        # Ensure the depth output is a NumPy array
        if not isinstance(depth, np.ndarray):
            try:
                depth = np.array(depth)
            except Exception:
                try:
                    depth = depth.detach().cpu().numpy()
                except Exception as e:
                    raise RuntimeError(f"Cannot convert depth to numpy array: {e}")

        # Resize depth map to match the original frame resolution
        depth_resized = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Create a normalized and colorized visualization for display
        depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

        return depth_resized, depth_color
