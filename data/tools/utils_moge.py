import cv2
import numpy as np
import torch

from thirdparty.MoGe.moge.model.v2 import MoGeModel as MoGeModelV2


class MogePipeline:
    """
    Inference pipeline for MoGeModelV2 to estimate the horizontal Field of View (FoV).
    """

    def __init__(
        self, 
        model_name: str = "Ruicheng/moge-2-vitl", 
        device: torch.device = torch.device("cuda")
    ):
        """
        Initializes the pipeline and loads the MoGe model.

        Args:
            model_name (str): Path or name of the pre-trained MoGe model.
            device (torch.device): Device to load the model onto (e.g., 'cuda').
        """
        self.device = device
        self.model = MoGeModelV2.from_pretrained(model_name).to(device)

    def infer(self, input_image: np.ndarray) -> float:
        """
        Performs inference to estimate the horizontal FoV from an image.

        Args:
            input_image (np.ndarray): The input image (H, W, 3) in BGR format.

        Returns:
            float: The estimated horizontal FoV in degrees.
        """

        input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_tensor = torch.tensor(
            input_image_rgb / 255.0, 
            dtype=torch.float32, 
            device=self.device
        ).permute(2, 0, 1)

        # Run model inference (resolution_level=1 with minimal token usage)
        output = self.model.infer(input_tensor, resolution_level=1)

        intrinsics = output['intrinsics'].cpu().numpy()
        
        # Calculate horizontal FoV: FoV_x = 2 * arctan(cx / fx)
        fov_x_rad = 2 * np.arctan(intrinsics[0, 2] / intrinsics[0, 0])
        fov_x_deg = np.rad2deg(fov_x_rad)
        
        return fov_x_deg