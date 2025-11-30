import json
import numpy as np
from PIL import Image
import torch
from vitra.models import VITRA_Paligemma, load_model
from vitra.utils.data_utils import resize_short_side_to_target
from vitra.datasets.human_dataset import pad_state_human, pad_action
from vitra.utils.data_utils import load_normalizer, recon_traj
from scipy.spatial.transform import Rotation as R
from vitra.datasets.dataset_utils import (
    compute_new_intrinsics_resize, 
    calculate_fov,
    ActionFeature,
    StateFeature,
)
import os
import sys
import argparse
from pathlib import Path

repo_root = Path(__file__).parent.parent  # VITRA/
sys.path.insert(0, str(repo_root))

from visualization.visualize_core import HandVisualizer, normalize_camera_intrinsics, save_to_video, Renderer, process_single_hand_labels
from visualization.visualize_core import Config as HandConfig

def get_state(hand_path):
    """
    Load and extract right hand state from saved hand data file.
    
    Args:
        hand_path (str): Path to the .npy file containing hand data
        
    Returns:
        tuple: (state_t0, beta, fov_x, None) where:
            - state_t0 (np.ndarray): Right hand state [51] containing translation (3), 
                                     global rotation (3 euler angles), and hand pose (45 euler angles)
            - beta (np.ndarray): MANO shape parameters [10]
            - fov_x (float): Horizontal field of view in degrees
            - None: Placeholder for optional text annotations
    """
    hand_data = np.load(hand_path, allow_pickle=True).item()
    hand_pose_t0 = hand_data['right'][0]['hand_pose']
    hand_pose_t0_euler = R.from_matrix(hand_pose_t0).as_euler('xyz', degrees=False) # [15, 3]
    hand_pose_t0_euler = hand_pose_t0_euler.reshape(-1)  # [45]
    global_orient_mat_t0 = hand_data['right'][0]['global_orient']
    R_t0_euler = R.from_matrix(global_orient_mat_t0).as_euler('xyz', degrees=False)  # [3]
    transl_t0 = hand_data['right'][0]['transl']  # [3]
    state_t0 = np.concatenate([transl_t0, R_t0_euler, hand_pose_t0_euler])  # [3+3+45=51]
    fov_x = hand_data['fov_x']
    print('fov_x:', fov_x)
    return state_t0, hand_data['right'][0]['beta'], fov_x, None

def get_state_left(hand_path):
    """
    Load and extract left hand state from saved hand data file.
    
    Args:
        hand_path (str): Path to the .npy file containing hand data
        
    Returns:
        tuple: (state_t0, beta, fov_x, None) where:
            - state_t0 (np.ndarray): Left hand state [51] containing translation (3), 
                                     global rotation (3 euler angles), and hand pose (45 euler angles)
            - beta (np.ndarray): MANO shape parameters [10]
            - fov_x (float): Horizontal field of view in degrees
            - None: Placeholder for optional text annotations
    """
    hand_data = np.load(hand_path, allow_pickle=True).item()
    hand_pose_t0 = hand_data['left'][0]['hand_pose']
    hand_pose_t0_euler = R.from_matrix(hand_pose_t0).as_euler('xyz', degrees=False) # [15, 3]
    hand_pose_t0_euler = hand_pose_t0_euler.reshape(-1)  # [45]
    global_orient_mat_t0 = hand_data['left'][0]['global_orient']
    R_t0_euler = R.from_matrix(global_orient_mat_t0).as_euler('xyz', degrees=False)  # [3]
    transl_t0 = hand_data['left'][0]['transl']  # [3]
    state_t0 = np.concatenate([transl_t0, R_t0_euler, hand_pose_t0_euler])  # [3+3+45=51]
    fov_x = hand_data['fov_x']

    return state_t0, hand_data['left'][0]['beta'], fov_x, None

def euler_traj_to_rotmat_traj(euler_traj, T):
    """
    Convert Euler angle trajectory to rotation matrix trajectory.
    
    Converts a sequence of hand poses represented as Euler angles into 
    rotation matrices suitable for MANO model input.
    
    Args:
        euler_traj (np.ndarray): Hand pose trajectory as Euler angles.
                                 Shape: [T, 45] where T is number of timesteps
                                 and 45 = 15 joints * 3 Euler angles per joint
        T (int): Number of timesteps in the trajectory
        
    Returns:
        np.ndarray: Rotation matrix trajectory. Shape: [T, 15, 3, 3]
                    where each [3, 3] block is a rotation matrix for one joint
    """
    hand_pose = euler_traj.reshape(-1, 3)  # [T*15, 3]
    pose_matrices = R.from_euler('xyz', hand_pose).as_matrix()  # [T*15, 3, 3]
    pose_matrices = pose_matrices.reshape(T, 15, 3, 3)  # [T, 15, 3, 3]

    return pose_matrices

def main():
    """
    Main execution function for hand action prediction and visualization.
    
    This function performs the following steps:
    1. Parses command-line arguments for visualization settings
    2. Loads VLA model and data normalizer
    3. Loads input image and hand state from files
    4. Prepares state and action masks based on which hands to predict
    5. Runs model inference to predict future hand actions
    6. Reconstructs absolute hand trajectories from relative actions
    7. Visualizes predicted hand motions overlaid on input image
    8. Saves visualization as video file
    
    Command-line arguments:
        --config: Path to model configuration JSON file
        --model_path: Path to model checkpoint (optional, overrides config)
        --statistics_path: Path to normalization statistics JSON (optional, overrides config)
        --image_path: Path to input image file
        --hand_path: Path to hand state .npy file
        --video_path: Path to save output visualization video
        --mano_model_path: Path to MANO model files
        --use_left: Whether to predict left hand motion
        --use_right: Whether to predict right hand motion
        --instruction: Text instruction for the model
    """
    parser = argparse.ArgumentParser(description="Hand VLA inference and visualization.")
    
    # Model Configuration
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model configuration JSON file.'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to model checkpoint (overrides config if provided).'
    )
    parser.add_argument(
        '--statistics_path',
        type=str,
        default='VITRA-VLA-3B/statistics/dataset_statistics.json',
        help='Path to normalization statistics JSON (overrides config if provided).'
    )
    
    # Input Files
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Path to input image file.'
    )
    parser.add_argument(
        '--hand_path',
        type=str,
        required=True,
        help='Path to hand state .npy file.'
    )
    
    # Output and Visualization
    parser.add_argument(
        '--video_path',
        type=str,
        default='./example_human_inf.mp4',
        help='path to save the output visualization videos.'
    )
    parser.add_argument(
        '--mano_model_path',
        type=str,
        default='./weights/mano',
        help='Path to the MANO model files.'
    )
    
    # Prediction Settings
    parser.add_argument(
        '--use_left',
        action='store_true',
        help='Enable left hand prediction.'
    )
    parser.add_argument(
        '--use_right',
        action='store_true',
        help='Enable right hand prediction.'
    )
    parser.add_argument(
        '--instruction',
        type=str,
        default="Left: Put the trash into the garbage. Right: None.",
        help='Text instruction for hand motion prediction.'
    )

    # === Environment Configuration ===
    # Disable tokenizers parallelism to avoid deadlocks in multi-process data loading
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parser.parse_args()
    
    # Validate that at least one hand is selected
    if not args.use_left and not args.use_right:
        raise ValueError("At least one of --use_left or --use_right must be specified.")
    
    # Load configs
    configs = json.load(open(args.config))
    
    # Override config with command-line arguments if provided
    if args.model_path is not None:
        configs['model_load_path'] = args.model_path
    if args.statistics_path is not None:
        configs['statistics_path'] = args.statistics_path

    # Load model and normalizer
    model = load_model(configs).cuda()
    model.eval()
    normalizer = load_normalizer(configs)

    # Load image
    image = Image.open(args.image_path)
    ori_w, ori_h = image.size
    image = resize_short_side_to_target(image, target=224)
    w, h = image.size

    use_right = args.use_right
    use_left = args.use_left

    # Initialize state
    hand_path = args.hand_path
    current_state = None
    current_state_left = None
    if use_right:
        current_state, beta, fov_x, _ = get_state(hand_path)
    if use_left:
        current_state_left, beta_left, fov_x, _ = get_state_left(hand_path)

    fov_x = fov_x * np.pi /180
    f_ori = ori_w / np.tan(fov_x / 2) /2
    fov_y = 2 * np.arctan(ori_h / (2 * f_ori))

    f = w / np.tan(fov_x / 2) /2
    intrinsics = np.array([
        [f, 0, w/2],
        [0, f, h/2],
        [0, 0, 1]
    ])

    # Concatenate left and right hand states, filling with zeros if one is None
    if current_state_left is None and current_state is None:
        raise ValueError("Both current_state_left and current_state are None")
    
    state_left = current_state_left if use_left else np.zeros_like(current_state)
    beta_left = beta_left if use_left else np.zeros_like(beta)
    state_right = current_state if use_right else np.zeros_like(current_state_left)
    beta_right = beta if use_right else np.zeros_like(beta_left)
    
    state = np.concatenate([state_left, beta_left, state_right, beta_right], axis=0)
    state_mask = np.array([use_left, use_right], dtype=bool)
    action_mask = np.tile(np.array([[use_left, use_right]], dtype=bool), (model.chunk_size, 1)) 

    fov = torch.tensor([fov_x, fov_y], dtype=torch.float32).unsqueeze(0)     # input your camera FOV here

    image = np.array(image)
    # Save the (resized) image to disk next to the output video path
    save_dir = Path(args.video_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = "/home/t-qixiuli/repo/VITRA/examples/1_resized.png"
    Image.fromarray(image).save(str(save_path))
    print(f"Saved resized image to: {save_path}")
    # Use instruction from command-line argument
    instruction = args.instruction

    # Normalize state
    norm_state = normalizer.normalize_state(state.copy())

    unified_action_dim = ActionFeature.ALL_FEATURES[1]   # 192
    unified_state_dim = StateFeature.ALL_FEATURES[1]     # 212

    unified_state, unified_state_mask = pad_state_human(
        state = norm_state,
        state_mask = state_mask,
        action_dim = normalizer.action_mean.shape[0],
        state_dim = normalizer.state_mean.shape[0],
        unified_state_dim = unified_state_dim,
    )
    _, unified_action_mask = pad_action(
        actions=None,
        action_mask=action_mask.copy(),
        action_dim=normalizer.action_mean.shape[0],
        unified_action_dim=unified_action_dim
    )

    # Model inference
    norm_action = model.predict_action(
        image = image,
        instruction = instruction,
        current_state = unified_state.unsqueeze(0),
        current_state_mask = unified_state_mask.unsqueeze(0),
        action_mask_torch = unified_action_mask.unsqueeze(0),
        num_ddim_steps = 10,
        cfg_scale = 5.0,
        fov = fov,
        sample_times = 1,
    )
    norm_action = norm_action[0, :, :102]
    # Denormalize predicted action
    unnorm_action = normalizer.unnormalize_action(norm_action)
    traj_right = np.zeros((len(action_mask)+1, 51), dtype=np.float32)
    traj_left = np.zeros((len(action_mask)+1, 51), dtype=np.float32)
    if use_left:
        traj_left = recon_traj(
            state=state_left,
            rel_action=unnorm_action[:, 0:51],
        )
    if use_right:
        traj_right = recon_traj(
            state=state_right,
            rel_action=unnorm_action[:, 51:102],
        )


    hand_config = HandConfig(args)
    hand_config.FPS = 3

    visualizer = HandVisualizer(hand_config, render_gradual_traj=False)
    T = len(traj_right)

    fx_exo = intrinsics[0, 0]
    fy_exo = intrinsics[1, 1]
    renderer = Renderer(w, h, (fx_exo, fy_exo), 'cuda')


    all_rendered_frames = []

    traj_mask = np.tile(np.array([[use_left, use_right]], dtype=bool), (T, 1)) 
    left_hand_mask = traj_mask[:, 0]
    right_hand_mask = traj_mask[:, 1]
    
    hand_mask = (left_hand_mask, right_hand_mask)

    left_hand_labels = {
        'transl_worldspace': traj_left[:, 0:3],
        'global_orient_worldspace': R.from_euler('xyz', traj_left[:, 3:6]).as_matrix(),
        'hand_pose': euler_traj_to_rotmat_traj(traj_left[:, 6:51], T),
        'beta': beta_left,
    }
    right_hand_labels = {
        'transl_worldspace': traj_right[:, 0:3],
        'global_orient_worldspace': R.from_euler('xyz', traj_right[:, 3:6]).as_matrix(),
        'hand_pose': euler_traj_to_rotmat_traj(traj_right[:, 6:51], T),
        'beta': beta_right,
    }
    verts_left_worldspace, _ = process_single_hand_labels(left_hand_labels, left_hand_mask, visualizer.mano, is_left=True)
    verts_right_worldspace, _ = process_single_hand_labels(right_hand_labels, right_hand_mask, visualizer.mano, is_left=False)

    hand_traj_wordspace = (verts_left_worldspace, verts_right_worldspace)
    
    R_w2c = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()   # Identity rotation matrices for T frames， world to camera
    t_w2c = np.zeros((T, 3, 1), dtype=np.float32)          # Zero translation vectors for T frames, world to camera

    extrinsics = (R_w2c, t_w2c)



    image_bgr = image[..., ::-1]  # Convert RGB to BGR by reversing the last axis
    resize_video_frames = [image_bgr] * T
    save_frames = visualizer._render_hand_trajectory(
        resize_video_frames,
        hand_traj_wordspace,
        hand_mask,
        extrinsics,
        renderer,
        mode='first'
    )
    all_rendered_frames.append(save_frames)
    save_to_video(save_frames, f'{args.video_path}', fps=hand_config.FPS)
    
    
if __name__ == "__main__":
    main()