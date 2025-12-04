"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import PIL.Image as Image
from scipy.spatial.transform import Rotation as R

import json
import os
from huggingface_hub import hf_hub_download

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None


        if self.padding_side == "right":  
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)  
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)  
        elif self.padding_side == "left":  
            # Manually pad sequences on the left  
            max_len = max(len(seq) for seq in input_ids)  
            input_ids = [torch.cat((torch.full((max_len - len(seq),), self.pad_token_id, dtype=seq.dtype), seq)) for seq in input_ids]  
            labels = [torch.cat((torch.full((max_len - len(seq),), IGNORE_INDEX, dtype=seq.dtype), seq)) for seq in labels]  
            input_ids = torch.stack(input_ids)  
            labels = torch.stack(labels)  
        else:  
            raise ValueError(f"Invalid padding_side: {self.padding_side}")  
  
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]  
  
        attention_mask = input_ids.ne(self.pad_token_id)  

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions)
        action_masks = [instance["action_masks"] for instance in instances]
        action_masks = torch.stack(action_masks)
        # Add continuous actions
        pixel_values = pixel_values.view(-1, *pixel_values.shape[2:]) 
        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            action_masks=action_masks,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output


@dataclass
class PaddedCollatorForHandPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # "labels" is not used for hand prediction, but we keep it for compatibility
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        fov = [instance["fov"] for instance in instances]
        fov = torch.stack(fov)

        intrinsics = [instance["intrinsics"] for instance in instances]
        intrinsics = torch.stack(intrinsics)

        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        if self.padding_side == "right":  
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
            if all([label is not None for label in labels]):
                labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            else:
                labels = torch.zeros_like(input_ids)
        elif self.padding_side == "left":  
            # Manually pad sequences on the left  
            max_len = max(len(seq) for seq in input_ids)  
            input_ids = [torch.cat((torch.full((max_len - len(seq),), self.pad_token_id, dtype=seq.dtype), seq)) for seq in input_ids]  
            input_ids = torch.stack(input_ids)  
            if all([label is not None for label in labels]):
                labels = [torch.cat((torch.full((max_len - len(seq),), IGNORE_INDEX, dtype=seq.dtype), seq)) for seq in labels]  
                labels = torch.stack(labels)
            else:
                labels = torch.zeros_like(input_ids)
        else:  
            raise ValueError(f"Invalid padding_side: {self.padding_side}") 

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions)
        if "action_masks" in instances[0]:
            action_masks = [instance["action_masks"] for instance in instances]
            action_masks = torch.stack(action_masks)
        else:
            action_masks = None
        # Add continuous actions 
        if "current_state_mask" in instances[0]:
            current_state_mask = [instance["current_state_mask"] for instance in instances]
            current_state_mask = torch.stack(current_state_mask)
            current_state = [instance["current_state"] for instance in instances]
            current_state = torch.stack(current_state)
        else:
            current_state = None

        pixel_values = pixel_values.view(-1, *pixel_values.shape[2:]) 

        # "labels" is not used for hand prediction, but we keep it for compatibility
        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            action_masks=action_masks,
            current_state_mask=current_state_mask,
            current_state=current_state,
            fov=fov,
            intrinsics=intrinsics,
        )

        if dataset_names is not None:
            output["dataset_names"] = dataset_names
    
        return output

def read_dataset_statistics(statistics_path: str, default_value=1e-4) -> dict:
    """
    Read dataset statistics from a JSON file.
    Args:
        statistics_path: Path to the JSON file containing dataset statistics.
        default_value: Default value to use if statistics are missing.
    Returns:
        data_statistics: Dictionary with mean and std for state and action of both hands.
    """
    with open(statistics_path, 'r') as file:
        dataset_stats = json.load(file)
        
        # Assert that right hand statistics must exist
        assert 'state_right' in dataset_stats, "Right hand statistics must exist"
        
        # Get right hand statistics
        state_right_mean = np.array(dataset_stats['state_right']['mean'])
        state_right_std = np.array(dataset_stats['state_right']['std'])
        action_right_mean = np.array(dataset_stats['action_right']['mean'])
        action_right_std = np.array(dataset_stats['action_right']['std'])
        
        # For left hand, use right hand dimensions but fill with default_value if not available
        if 'state_left' in dataset_stats:
            state_left_mean = np.array(dataset_stats['state_left']['mean'])
            state_left_std = np.array(dataset_stats['state_left']['std'])
            action_left_mean = np.array(dataset_stats['action_left']['mean'])
            action_left_std = np.array(dataset_stats['action_left']['std'])
        else:
            state_left_mean = np.full_like(state_right_mean, default_value)
            state_left_std = np.full_like(state_right_std, default_value)
            action_left_mean = np.full_like(action_right_mean, default_value)
            action_left_std = np.full_like(action_right_std, default_value)
        
        data_statistics = {
            'state_right_mean': state_right_mean,
            'state_right_std': state_right_std,
            'action_right_mean': action_right_mean,
            'action_right_std': action_right_std,
            'state_left_mean': state_left_mean,
            'state_left_std': state_left_std,
            'action_left_mean': action_left_mean,
            'action_left_std': action_left_std,
        }
    return data_statistics

class GaussianNormalizer:
    """
    A class for normalizing and denormalizing state and action arrays.
    Assumes state/action numpy arrays are concatenated as [left, right].
    Accepts pre-loaded data_statistics dictionary.
    """
    def __init__(self, data_statistics: dict):
        """
        Args:
            data_statistics (dict): pre-loaded statistics dictionary with keys:
                'state_left_mean', 'state_left_std', 'state_right_mean', 'state_right_std',
                'action_left_mean', 'action_left_std', 'action_right_mean', 'action_right_std'
                All values are numpy arrays.
        """
        # Concatenate left and right statistics for vectorized operations
        self.state_mean = np.concatenate([data_statistics['state_left_mean'], data_statistics['state_right_mean']])
        self.state_std = np.concatenate([data_statistics['state_left_std'], data_statistics['state_right_std']])
        self.action_mean = np.concatenate([data_statistics['action_left_mean'], data_statistics['action_right_mean']])
        self.action_std = np.concatenate([data_statistics['action_left_std'], data_statistics['action_right_std']])

    # -----------------------------
    # State normalization
    # -----------------------------
    def normalize_state(self, state: np.ndarray, epsilon=1e-7) -> np.ndarray:
        return (state - self.state_mean) / (self.state_std + epsilon)

    def unnormalize_state(self, norm_state: np.ndarray, epsilon=1e-7) -> np.ndarray:
        return norm_state * (self.state_std + epsilon) + self.state_mean
    # -----------------------------
    # Action normalization
    # -----------------------------
    def normalize_action(self, action: np.ndarray, epsilon=1e-7) -> np.ndarray:
        return (action - self.action_mean) / (self.action_std + epsilon)

    def unnormalize_action(self, norm_action: np.ndarray, epsilon=1e-7) -> np.ndarray:
        return norm_action * (self.action_std + epsilon) + self.action_mean

def gaussian_normalize(data, mean, std, epsilon=1e-7):
    """
    General normalization function for data.
    
    Args:
        data: numpy array or torch tensor to normalize
        mean: mean value(s) for normalization (scalar or array)
        std: standard deviation value(s) for normalization (scalar or array)
        epsilon: small value to avoid division by zero
    
    Returns:
        normalized data in the same format as input
    """
    mean = np.asarray(mean)
    std = np.asarray(std)
    return (data - mean) / (std + epsilon)

def resize_short_side_to_target(image, target=224):
    """
    Resize the image so that its short side matches the target size,
    preserving the aspect ratio and EXIF orientation.
    Args:
        image: PIL Image to resize
        target: Desired size of the short side
    Returns:
        Resized PIL Image with correct orientation
    """
    
    w, h = image.size

    if w < h:
        new_w = target
        new_h = int(h * target / w)
    else:
        new_h = target
        new_w = int(w * target / h)

    # Use Image.Resampling.LANCZOS for newer Pillow versions, fallback to Image.LANCZOS
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
    
    image_resized = image.resize((new_w, new_h), resample_filter)
    return image_resized

def load_normalizer(configs):
    stats_path = configs.get("statistics_path", None)
   # Check if stats_path is a Hugging Face repo (format: "username/repo-name" or "username/repo-name:filename")
    if stats_path and "/" in stats_path and not os.path.exists(stats_path):
        if ":" in stats_path:
            # Format: "username/repo-name:statistics.json"
            repo_id, filename = stats_path.split(":", 1)
            print(f"Loading statistics from Hugging Face Hub: {repo_id}/{filename}")
        else:
            # Format: "username/repo-name" - default to "statistics/statistics.json"
            repo_id = stats_path
            filename = "statistics/dataset_statistics.json"
            print(f"Loading statistics from Hugging Face Hub: {repo_id}/{filename}")
        stats_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
        
    # Load dataset statistics
    data_statistics = read_dataset_statistics(stats_path)
    gaussian_normalizer = GaussianNormalizer(data_statistics)
    return gaussian_normalizer

def recon_abs_actions(action, t_tgt_in_cam, R_tgt_in_cam, hand_pose, abs_joint = True):
    # accumulate pred action to hand_pose, 
    # accumulate pred action to t_tgt_in_cam, 
    # accumulate pred action to R_tgt_in_cam
    pred_hand_pose_rel_euler = action[-45:].reshape(15, 3)
    pre_t_rel = action[:3]
    pre_R_rel_euler = action[3:6]
    pred_hand_pose_rel = R.from_euler('xyz', pred_hand_pose_rel_euler).as_matrix() # degrees = False
    if abs_joint == True:
        pred_hand_pose = pred_hand_pose_rel # hand_pose_t+1 [15, 3, 3]
    else:
        pred_hand_pose = np.matmul(pred_hand_pose_rel, hand_pose) # hand_pose_t+1 [15, 3, 3]
    pre_t = pre_t_rel + t_tgt_in_cam 
    pred_R_rel = R.from_euler('xyz', pre_R_rel_euler).as_matrix() # degrees = False
    pred_R_abs = np.dot(pred_R_rel, R_tgt_in_cam) # R_abs from t to t+1 [3, 3]
    return pre_t, pred_R_abs, pred_hand_pose

def recon_traj(state, rel_action, abs_joint=True, rel_mode='step'):
    """
    Reconstruct the hand trajectory for either left or right hand.
    
    Args:
        state: Current hand state (translation + rotation + hand pose)
        rel_action: Relative action sequence [T, 51]
        abs_joint: Whether to use absolute joint angles
        rel_mode: 'anchor' or 'step' mode for accumulation
        
    Returns:
        traj_list: Stacked absolute actions [T+1, 51] (including initial state at step 0)
    """
    t_tgt_in_cam = state[:3]
    R_tgt_in_cam = state[3:6]
    hand_pose = state[6:]
    
    # Convert to matrix format
    hand_pose = hand_pose.reshape(15, 3)
    hand_pose = R.from_euler('xyz', hand_pose).as_matrix()
    R_tgt_in_cam = R.from_euler('xyz', R_tgt_in_cam).as_matrix()
    
    traj_list = []
    
    # Add initial state (step 0)
    initial_action = np.concatenate([
        state[:3],  # translation
        state[3:6],  # rotation (euler)
        state[6:]  # hand pose (euler)
    ])
    traj_list.append(initial_action)
    
    # Process future steps
    for t in range(rel_action.shape[0]):
        pre_t, pred_R_abs, pred_hand_pose = recon_abs_actions(
            rel_action[t], t_tgt_in_cam, R_tgt_in_cam, hand_pose, abs_joint=abs_joint
        )
        
        # Update state for next iteration (if not in anchor mode)
        if rel_mode != 'anchor':
            t_tgt_in_cam = pre_t
            R_tgt_in_cam = pred_R_abs
            hand_pose = pred_hand_pose
        
        # Concatenate action components
        action = np.concatenate([
            pre_t,
            R.from_matrix(pred_R_abs).as_euler('xyz', degrees=False),
            R.from_matrix(pred_hand_pose).as_euler('xyz', degrees=False).reshape(-1)
        ])
        traj_list.append(action)
    
    return np.stack(traj_list)  # Shape: [T+1, 51]

def recover_state_from_actions(pred_actions):
    for pred_idx, pred_actions in enumerate(pred_actions_all):
        render_img_list = []
        all_action_list_tmp = []
        all_action_list_tmp_left = []

        rel_action_list = pred_actions[:, :102]  # [T, 102]
        
        action_list_tmp = None
        action_list_tmp_left = None
        
        # Process right hand
        if use_right:
            action_list_tmp = _process_hand_actions(
                state=current_state_input,
                rel_action=rel_action_list[:, 51:102],
                abs_joint=abs_joint,
                rel_mode=rel_mode
            )
            
            # Compute intrinsics if needed
            intrinsics_new = compute_new_intrinsics_rect(intrinsics, (new_h, new_w))
        
        # Process left hand
        if use_left:
            action_list_tmp_left = _process_hand_actions(
                state=current_state_left_input,
                rel_action=rel_action_list[:, :51],
                abs_joint=abs_joint,
                rel_mode=rel_mode
            )
        
        if use_chunk_length is None:
            use_chunk_length = rel_action_list.shape[0]