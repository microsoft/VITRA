"""
robot_dataset.py

Dataset implementation for robot manipulation data (e.g., XHand teleoperation).
Handles loading, processing, and normalization of robot state and action trajectories.
"""

import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from vitra.datasets.augment_utils import apply_color_augmentation
from vitra.datasets.dataset_utils import (
    ActionFeature,
    StateFeature,
    calculate_fov,
    compute_new_intrinsics_crop,
    compute_new_intrinsics_resize,
)
from vitra.datasets.human_dataset import pad_action
from vitra.utils.data_utils import GaussianNormalizer, read_dataset_statistics


# Mapping from XHand joint indices to human hand model indices
# Format: (xhand_idx, human_idx, sign_multiplier)
XHAND_HUMAN_MAPPING = [
    (10, 8, 1),
    (11, 14, 1),
    (12, 17, 1),
    (13, 23, 1),
    (16, 26, 1),
    (17, 32, 1),
    (14, 35, 1),
    (15, 41, 1),
    (7, 43, -1),
    (6, 44, 1),
    (8, 46, -1),
    (9, 7, 1),
]


class RoboDatasetCore(object):
    """
    Core dataset class for robot manipulation data.
    
    Handles loading and processing of robot teleoperation trajectories,
    including state/action sequences, images, and camera parameters.
    """
    
    def __init__(
        self,
        root_dir: str,
        statistics_path: Optional[str] = None,
        action_past_window_size: int = 0,
        action_future_window_size: int = 16,
        image_past_window_size: int = 0,
        image_future_window_size: int = 0,
        load_images: bool = True
    ):
        """
        Initialize robot dataset core.
        
        Args:
            root_dir: Root directory containing robot data
            statistics_path: Path to normalization statistics JSON file
            action_past_window_size: Number of past action frames to include
            action_future_window_size: Number of future action frames to predict
            image_past_window_size: Number of past image frames to include
            image_future_window_size: Number of future image frames to include
            load_images: Whether to load image data
        """
        self.root = root_dir
        self.action_past_window_size = action_past_window_size
        self.action_future_window_size = action_future_window_size
        self.image_past_window_size = image_past_window_size
        self.image_future_window_size = image_future_window_size
        self.load_images = load_images
        
        # Load normalization statistics if provided
        if statistics_path is not None:
            self.data_statistics = read_dataset_statistics(statistics_path)
        else:
            self.data_statistics = None
        
        # TODO: Implement data loading
        # - Load instruction JSON
        # - Build sample index
        
        raise NotImplementedError("Data loading logic needs to be implemented")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def set_global_data_statistics(self, global_data_statistics: dict):
        """
        Set global normalization statistics and initialize normalizer.
        
        Args:
            global_data_statistics: Dictionary containing mean/std for state/action
        """
        self.global_data_statistics = global_data_statistics
        if not hasattr(self, 'gaussian_normalizer'):
            self.gaussian_normalizer = GaussianNormalizer(self.global_data_statistics)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - image_list: List of RGB images
                - action_list: Action sequence array
                - current_state: Current robot state
                - action_mask: Mask indicating valid actions
                - current_state_mask: Mask indicating valid state dimensions
                - fov: Field of view (horizontal, vertical)
                - intrinsics: Camera intrinsic matrix
                - instruction: Text instruction
        """
        # TODO: Implement data loading for single sample
        # - Load image(s)
        # - Load state and action sequences
        # - Apply augmentation if needed
        # - Compute FOV from intrinsics
        raise NotImplementedError("Sample loading logic needs to be implemented")

    def transform_trajectory(
        self,
        sample_dict: dict = None,
        normalization: bool = True,
    ):
        """Pad action and state dimensions to match a unified size."""

        action_np = sample_dict["action_list"]
        state_np = sample_dict["current_state"]
       
        action_dim = action_np.shape[1]
        state_dim = state_np.shape[0]
        if normalization:
            # Normalize left and right hand actions and states separately
            action_np = self.gaussian_normalizer.normalize_action(action_np)
            state_np = self.gaussian_normalizer.normalize_state(state_np)

        # ===== Pad to unified dimensions =====
        unified_action_dim = ActionFeature.ALL_FEATURES[1]   # 192
        unified_state_dim = StateFeature.ALL_FEATURES[1]     # 212

        unified_action, unified_action_mask = pad_action(
            action_np,
            sample_dict["action_mask"],
            action_dim,
            unified_action_dim
        )

        unified_state, unified_state_mask = pad_state_robot(
            state_np,
            sample_dict["current_state_mask"],
            state_dim,
            unified_state_dim
        )

        human_state, human_state_mask, human_action, human_action_mask = transfer_xhand_to_human(
            unified_state, unified_state_mask,
            unified_action, unified_action_mask
        )

        sample_dict["action_list"] = human_action
        sample_dict["action_mask"] = human_action_mask
        sample_dict["current_state"] = human_state
        sample_dict["current_state_mask"] = human_state_mask

        return sample_dict

def pad_state_robot(
    state: torch.Tensor,
    state_mask: torch.Tensor,
    state_dim: int,
    unified_state_dim: int
):
    """
    Expand state mask, mask invalid state dims, and pad current_state to a standard size.

    Args:
        current_state (Tensor): original state tensor, shape [state_dim]
        current_state_mask (Tensor): per-hand state mask, shape [state_dim//2] or [state_dim]
        state_dim (int): original state dimension
        unified_state_dim (int): target padded state dimension

    Returns:
        Tuple[Tensor, Tensor]: 
            padded current_state [unified_state_dim],
            padded current_state_mask [unified_state_dim]
    """

    current_state = torch.tensor(state, dtype=torch.float32)
    current_state_mask = torch.tensor(state_mask, dtype=torch.bool)
    
    # Expand state mask from per-hand to per-dim
    expanded_state_mask = current_state_mask.repeat_interleave(state_dim // 2)

    # Mask out invalid state dimensions
    current_state_masked = current_state * expanded_state_mask.to(current_state.dtype)

    # Initialize output tensors
    padded_state = torch.zeros(unified_state_dim, dtype=current_state.dtype)
    padded_mask = torch.zeros(unified_state_dim, dtype=torch.bool)

    # Fill first half of state_dim (left hand)
    padded_state[:state_dim//2] = current_state_masked[:state_dim//2].clone()
    padded_mask[:state_dim//2] = expanded_state_mask[:state_dim//2].clone()

    # Fill second half of state_dim (right hand)
    padded_state[state_dim//2:state_dim] = current_state_masked[state_dim//2:state_dim].clone()
    padded_mask[state_dim//2:state_dim] = expanded_state_mask[state_dim//2:state_dim].clone()

    return padded_state, padded_mask

def transfer_xhand_to_human(
    unified_state: torch.Tensor,
    unified_state_mask: torch.Tensor,
    unified_action: Optional[torch.Tensor],
    unified_action_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Transfer XHand robot format to human hand model format.
    
    Maps robot joint angles and end-effector poses to the human hand
    representation used by the model.
    
    Args:
        unified_state: Robot state tensor, shape (212,) - required
        unified_state_mask: State validity mask, shape (212,) - required
        unified_action: Robot action sequence, shape (T, 192) - can be None
        unified_action_mask: Action validity mask, shape (T, 192) - required
        
    Returns:
        Tuple of (human_state, human_state_mask, human_action, human_action_mask)
        human_action will be None if unified_action is None
    """
    
    # Initialize output tensors for state (always required)
    human_state = torch.zeros(unified_state.shape[0], dtype=torch.float32)
    human_state_mask = torch.zeros(unified_state_mask.shape[0], dtype=torch.bool)
    
    # Transfer left hand end-effector 6-DoF pose (translation + rotation)
    human_state[0:6] = unified_state[0:6]
    human_state_mask[0:6] = unified_state_mask[0:6]

    # Transfer right hand end-effector 6-DoF pose
    human_state[51:57] = unified_state[18:24]
    human_state_mask[51:57] = unified_state_mask[18:24]

    for src, dst, sign in XHAND_HUMAN_MAPPING:
        human_state[dst] = sign * unified_state[src]
        human_state[dst+51] = sign * unified_state[src+18]

    # Set state mask strategy: if hand is active, set entire hand dimensions to active
    human_state_mask[0:51] = unified_state_mask[0]  # scalar bool for left hand
    human_state_mask[51:102] = unified_state_mask[18]  # scalar bool for right hand

    # Initialize action mask (always required, even if action is None)
    human_action_mask = torch.zeros((unified_action_mask.shape[0], 192), dtype=torch.bool)

    # Set action mask strategy: if hand is active, set entire hand dimensions to active
    # For left hand (columns 0:51)
    left_hand_active = unified_action_mask[:, 0]  # (T,) bool array
    human_action_mask[:, 0:51] = left_hand_active.unsqueeze(1).expand(-1, 51)
    
    # For right hand (columns 51:102)
    right_hand_active = unified_action_mask[:, 18]  # (T,) bool array
    human_action_mask[:, 51:102] = right_hand_active.unsqueeze(1).expand(-1, 51)

    # Handle action data (can be None)
    if unified_action is None:
        return human_state, human_state_mask, None, human_action_mask
    
    # Initialize and process action data
    human_action = torch.zeros((unified_action.shape[0], 192), dtype=torch.float32)
    human_action[:, 0:6] = unified_action[:, 0:6]
    human_action[:, 51:57] = unified_action[:, 18:24]

    for src, dst, sign in XHAND_HUMAN_MAPPING:
        human_action[:, dst] = sign * unified_action[:, src]
        human_action[:, dst+51] = sign * unified_action[:, src+18]

    return human_state, human_state_mask, human_action, human_action_mask


def transfer_human_to_xhand(
    human_action: torch.Tensor,
) -> torch.Tensor:
    """
    Transfer human hand model format back to XHand robot format.
    Maps human hand representation to robot joint angles and end-effector poses.
    This is the inverse operation of transfer_xhand_to_human for actions only.
    Args:
        human_action: Human action sequence, shape (T, 192)
    Returns:
        xhand_action: Robot action sequence, shape (T, 36) - 18 dims per hand
    """
    T = human_action.shape[0]
    # Initialize output tensor: 18 dims per hand (6 EEF + 12 joints)
    xhand_action = torch.zeros((T, 36), dtype=torch.float32)
    # Transfer left hand end-effector 6-DoF pose (translation + rotation)
    xhand_action[:, 0:6] = human_action[:, 0:6]
    # Transfer right hand end-effector 6-DoF pose
    xhand_action[:, 18:24] = human_action[:, 51:57]
    
    # Transfer joint angles using reverse mapping
    for src, dst, sign in XHAND_HUMAN_MAPPING:
        # Left hand: human -> xhand
        xhand_action[:, src] = sign * human_action[:, dst]
        # Right hand: human -> xhand
        xhand_action[:, src+18] = sign * human_action[:, dst+51]
    
    return xhand_action