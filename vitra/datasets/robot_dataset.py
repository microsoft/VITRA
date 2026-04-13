"""
robot_dataset.py

Dataset implementation for robot manipulation data (e.g., XHand teleoperation).
Handles loading, processing, and normalization of robot state and action trajectories.
"""

import json
import os
from functools import lru_cache
from typing import Optional, Tuple

import h5py
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
from vitra.datasets.video_utils import load_video_decord
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
        self.root = root_dir
        self.action_past_window_size = action_past_window_size
        self.action_future_window_size = action_future_window_size
        self.image_past_window_size = image_past_window_size
        self.image_future_window_size = image_future_window_size
        self.load_images = load_images

        # Load normalization statistics if provided
        if statistics_path is not None and os.path.exists(statistics_path):
            self.data_statistics = read_dataset_statistics(statistics_path)
        else:
            self.data_statistics = None

        # Discover all episode h5 files
        self.episode_ids = sorted([
            f[:-3] for f in os.listdir(root_dir)
            if f.endswith('.h5')
        ])

        # Build a flat sample index: list of (episode_idx, frame_idx)
        # so each sample corresponds to one frame in one episode
        self.samples = []
        self.episode_lengths = []
        for ep_idx, ep_id in enumerate(self.episode_ids):
            h5_path = os.path.join(self.root, ep_id + '.h5')
            with h5py.File(h5_path, 'r') as f:
                T = int(f['meta/frame_count'][()])
            self.episode_lengths.append(T)
            for frame_idx in range(T):
                self.samples.append((ep_idx, frame_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def set_global_data_statistics(self, global_data_statistics: dict):
        self.global_data_statistics = global_data_statistics
        if not hasattr(self, 'gaussian_normalizer'):
            self.gaussian_normalizer = GaussianNormalizer(self.global_data_statistics)

    @staticmethod
    @lru_cache(maxsize=64)
    def _load_h5_data(h5_path: str):
        """Load and cache all numerical data from an h5 episode file."""
        data = {}
        with h5py.File(h5_path, 'r') as f:
            # Meta
            data['instruction'] = f['meta/instruction'][()].decode('utf-8') if isinstance(f['meta/instruction'][()], bytes) else str(f['meta/instruction'][()])
            data['frame_count'] = int(f['meta/frame_count'][()])
            data['has_left'] = bool(f['meta/has_left'][()])
            data['has_right'] = bool(f['meta/has_right'][()])

            # Observation
            data['intrinsics'] = f['observation/camera/intrinsics'][()].astype(np.float64)

            # State - EEF poses in camera frame and hand joints
            data['left_hand_mount_pose_in_cam'] = f['state/left_hand_mount_pose_in_cam'][()].astype(np.float32)
            data['right_hand_mount_pose_in_cam'] = f['state/right_hand_mount_pose_in_cam'][()].astype(np.float32)
            data['left_hand_joint'] = f['state/left_hand_joint'][()].astype(np.float32)
            data['right_hand_joint'] = f['state/right_hand_joint'][()].astype(np.float32)

            # Action - target joint positions
            data['action_left_arm_joint'] = f['action/left_arm_joint'][()].astype(np.float32)
            data['action_right_arm_joint'] = f['action/right_arm_joint'][()].astype(np.float32)
            data['action_left_hand_joint'] = f['action/left_hand_joint'][()].astype(np.float32)
            data['action_right_hand_joint'] = f['action/right_hand_joint'][()].astype(np.float32)

            # Masks
            data['mask_left_arm'] = f['mask/left_arm'][()].astype(bool)
            data['mask_right_arm'] = f['mask/right_arm'][()].astype(bool)
            data['mask_left_hand'] = f['mask/left_hand'][()].astype(bool)
            data['mask_right_hand'] = f['mask/right_hand'][()].astype(bool)
        return data

    def __getitem__(self, idx: int) -> dict:
        ep_idx, frame_idx = self.samples[idx]
        ep_id = self.episode_ids[ep_idx]
        h5_path = os.path.join(self.root, ep_id + '.h5')

        data = self._load_h5_data(h5_path)
        T = data['frame_count']

        # ---- Build action window indices ----
        # Window: [frame_idx - past, ..., frame_idx, ..., frame_idx + future]
        # We need future frames for the action chunk
        past = self.action_past_window_size
        future = self.action_future_window_size
        win_indices = np.arange(-past, future + 1) + frame_idx  # (W,)
        oob = (win_indices < 0) | (win_indices >= T)
        win_indices_clipped = np.clip(win_indices, 0, T - 1)
        W = len(win_indices)
        anchor = past  # index of current frame within window

        # ---- Build state at current frame ----
        # State format: [left_eef(6), left_hand_joint(Nh), right_eef(6), right_hand_joint(Nh)]
        left_eef = data['left_hand_mount_pose_in_cam'][frame_idx]   # (6,)
        right_eef = data['right_hand_mount_pose_in_cam'][frame_idx] # (6,)
        left_hand_j = data['left_hand_joint'][frame_idx]            # (Nh,)
        right_hand_j = data['right_hand_joint'][frame_idx]          # (Nh,)

        left_state = np.concatenate([left_eef, left_hand_j])    # (6+Nh,)
        right_state = np.concatenate([right_eef, right_hand_j]) # (6+Nh,)
        current_state = np.concatenate([left_state, right_state]).astype(np.float32)

        # State mask: per-hand validity at current frame
        has_left = data['has_left'] and data['mask_left_hand'][frame_idx]
        has_right = data['has_right'] and data['mask_right_hand'][frame_idx]
        current_state_mask = np.array([has_left, has_right], dtype=np.float32)

        # ---- Build action sequence ----
        # Action format per frame: [left_eef(6), left_hand_joint(Nh), right_eef(6), right_hand_joint(Nh)]
        # EEF action = pose at the NEXT frame (matching human dataset convention
        # where actions represent the target state at t+1).
        # Hand joint action = commanded target from action/ group (already targets t+1).
        next_indices = np.clip(win_indices + 1, 0, T - 1)
        next_oob = (win_indices + 1 < 0) | (win_indices + 1 >= T)
        left_eef_win = data['left_hand_mount_pose_in_cam'][next_indices]   # (W, 6)
        right_eef_win = data['right_hand_mount_pose_in_cam'][next_indices] # (W, 6)
        left_hand_act = data['action_left_hand_joint'][win_indices_clipped]       # (W, Nh)
        right_hand_act = data['action_right_hand_joint'][win_indices_clipped]     # (W, Nh)

        action_list = np.concatenate([
            left_eef_win, left_hand_act,
            right_eef_win, right_hand_act
        ], axis=1).astype(np.float32)  # (W, 36)

        # Action mask: per-frame, per-hand validity
        # Both the current frame AND next frame must be valid for a valid action
        left_valid = data['mask_left_hand'][win_indices_clipped] & data['mask_left_hand'][next_indices] & ~oob & ~next_oob
        right_valid = data['mask_right_hand'][win_indices_clipped] & data['mask_right_hand'][next_indices] & ~oob & ~next_oob
        action_mask = np.stack([left_valid, right_valid], axis=1)  # (W, 2)

        # Zero out invalid actions
        Nh = left_hand_act.shape[1]
        half_dim = 6 + Nh  # dims per hand in action
        for i in range(W):
            if not action_mask[i, 0]:
                action_list[i, :half_dim] = 0.0
            if not action_mask[i, 1]:
                action_list[i, half_dim:] = 0.0

        # ---- Load image ----
        if self.load_images:
            mp4_path = os.path.join(self.root, ep_id + '.mp4')
            imgs, _ = load_video_decord(mp4_path, frame_index=[frame_idx])
            image_list = np.stack(imgs, axis=0)  # (1, H, W, 3)
            image_mask = np.array([True])
        else:
            image_list = None
            image_mask = None

        # ---- Intrinsics and FOV ----
        intrinsics = data['intrinsics'].astype(np.float32)
        if image_list is not None:
            H, W_img = image_list.shape[1], image_list.shape[2]
            intrinsics = compute_new_intrinsics_resize(intrinsics, (H, W_img)).astype(np.float32)
        else:
            H = int(2 * intrinsics[1, 2])
            W_img = int(2 * intrinsics[0, 2])

        fov = calculate_fov(H, W_img, intrinsics)

        # ---- Instruction ----
        raw_instruction = data['instruction']
        if has_left and has_right:
            instruction = f"Left hand: {raw_instruction} Right hand: {raw_instruction}"
        elif has_right:
            instruction = f"Left hand: None. Right hand: {raw_instruction}"
        elif has_left:
            instruction = f"Left hand: {raw_instruction} Right hand: None."
        else:
            instruction = f"Left hand: None. Right hand: None."

        result_dict = dict(
            instruction=instruction,
            action_list=action_list,          # (W, 36) float32
            action_mask=action_mask,           # (W, 2) bool
            current_state=current_state,       # (36,) float32
            current_state_mask=current_state_mask,  # (2,) float32
            fov=fov,                           # (2,) float32
            intrinsics=intrinsics,             # (3, 3) float32
        )

        if image_list is not None:
            result_dict['image_list'] = image_list   # (1, H, W, 3) uint8
        if image_mask is not None:
            result_dict['image_mask'] = image_mask   # (1,) bool

        return result_dict

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