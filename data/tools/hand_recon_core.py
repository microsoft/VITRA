import numpy as np
import torch
import copy

from .utils_moge import MogePipeline
from .utils_hawor import HaworPipeline
from libs.models.mano_wrapper import MANO

class Config:
    """
    Configuration class for file paths.
    Paths are initialized with default values but can be overridden by arguments.
    """
    def __init__(self, args=None):
        # --- Paths (Overridden by CLI arguments) ---
        self.HAWOR_MODEL_PATH = getattr(args, 'hawor_model_path', './weights/hawor/hawor/checkpoints/hawor.ckpt')
        self.DETECTOR_PATH = getattr(args, 'detector_path', './weights/hawor/external/detector.pt')
        self.MOGE_MODEL_PATH = getattr(args, 'moge_model_path', 'Ruicheng/moge-2-vitl')
        self.MANO_PATH = getattr(args, 'mano_path', './weights/mano')


class HandReconstructor:
    """
    Core pipeline for 3D hand reconstruction, combining camera estimation (MoGe) 
    and motion/pose estimation (HaWoR) with the MANO model.
    """
    def __init__(
            self, 
            config: Config,
            device: torch.device = torch.device("cuda")
    ):
        """
        Initializes the reconstruction pipeline components.

        Args:
            hawor_model_path (str): Path to the HaWoR model checkpoint.
            detector_path (str): Path to the hand detector weights.
            moge_model_name (str): Name of the MoGe model for FoV estimation.
            mano_path (str): Path to the MANO model weights.
            device (torch.device): Device to load models onto.
        """
        self.device = device
        self.hawor_pipeline = HaworPipeline(
            model_path=config.HAWOR_MODEL_PATH, detector_path=config.DETECTOR_PATH, device=device
        )
        self.moge_pipeline = MogePipeline(
            model_name=config.MOGE_MODEL_PATH, device=device
        )
        self.mano = MANO(model_path=config.MANO_PATH).to(device)

    def recon(self, images: list) -> dict:
        """
        Performs the complete 3D hand reconstruction process.

        Steps:
        1. Estimate camera FoV using MoGe (median over all frames).
        2. Calculate the camera focal length.
        3. Estimate hand pose/shape/translation using HaWoR.
        4. Re-calculate the global translation by aligning the wrist joint 
           of the MANO mesh with the HaWoR predicted translation.

        Args:
            images (list): List of input image frames (numpy array format).

        Returns:
            dict: Reconstruction results with re-calculated translations and FoV.
        """

        N = len(images)
        if N == 0:
            return {'left': {}, 'right': {}, 'fov_x': None}
        
        H, W = images[0].shape[:2]

        # --- 1. FoV Estimation (MoGe) ---
        all_fov_x = []
        for i in range(N):
            img = images[i]
            fov_x = self.moge_pipeline.infer(img)
            all_fov_x.append(fov_x)
        
        # Use median FoV across all frames
        fov_x = np.median(np.array(all_fov_x))
        img_focal = 0.5 * W / np.tan(0.5 * fov_x * np.pi / 180)

        # --- 2. Hand Pose and Translation Estimation (HaWoR) ---
        recon_results = self.hawor_pipeline.recon(images, img_focal, single_image=(N==1))

        recon_results_new_transl = {'left': {}, 'right': {}, 'fov_x': fov_x}
        # --- 3. Re-calculate Global Translation (MANO Alignment) ---
        for img_idx in range(N):
            for hand_type in ['left', 'right']:
                if hand_type == 'left':
                    if not img_idx in recon_results['left']:
                        continue
                    result = recon_results['left'][img_idx]
                else:
                    if not img_idx in recon_results['right']:
                        continue
                    result = recon_results['right'][img_idx]

                # Convert results to tensors
                betas = torch.from_numpy(result['beta']).unsqueeze(0).to(self.device)
                hand_pose = torch.from_numpy(result['hand_pose']).unsqueeze(0).to(self.device)
                transl = torch.from_numpy(result['transl']).unsqueeze(0).to(self.device)  

                # Forward pass through MANO model
                model_output = self.mano(betas = betas, hand_pose = hand_pose)    
                verts_m = model_output.vertices[0]
                joints_m = model_output.joints[0]

                # Flip x-axis for left hand consistency
                if hand_type == 'left':
                    verts_m[:,0] = -1*verts_m[:,0]
                    joints_m[:,0] = -1*joints_m[:,0]
                
                wrist = joints_m[0]

                # Calculate new translation
                transl_new = wrist + transl

                # Store results with the new translation
                result_new_transl = copy.deepcopy(result)
                result_new_transl['transl'] = transl_new[0].cpu().numpy()
                recon_results_new_transl[hand_type][img_idx] = result_new_transl
        
        return recon_results_new_transl