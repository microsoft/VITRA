import os
import sys
import cv2
import math
import json
import torch
import argparse
import numpy as np
from PIL import Image
from PIL import ImageOps
from pathlib import Path
import multiprocessing as mp
from vitra.models import VITRA_Paligemma, load_model
from vitra.utils.data_utils import resize_short_side_to_target, load_normalizer, recon_traj
from vitra.utils.config_utils import load_config
from vitra.datasets.human_dataset import pad_state_human, pad_action
from scipy.spatial.transform import Rotation as R
from vitra.datasets.dataset_utils import (
    compute_new_intrinsics_resize, 
    calculate_fov,
    ActionFeature,
    StateFeature,
)

repo_root = Path(__file__).parent.parent  # VITRA/
sys.path.insert(0, str(repo_root))

from visualization.visualize_core import HandVisualizer, normalize_camera_intrinsics, save_to_video, Renderer, process_single_hand_labels
from visualization.visualize_core import Config as HandConfig

def main():
    """
    Main execution function for hand action prediction and visualization.
    
    This function uses a multi-process architecture to separate hand reconstruction
    and VLA inference into independent processes, preventing CUDA conflicts.
    
    Workflow:
    1. Parse command-line arguments and load model configurations
    2. Initialize persistent services:
       - HandReconstructionService: Runs HAWOR + MOGE models in separate process
       - VLAInferenceService: Runs VLA model in separate process
    3. Load or reconstruct hand state:
       - Uses precomputed .npy file if available (same stem as image)
       - Otherwise runs hand reconstruction service
    4. Prepare input data:
       - Load and resize image
       - Extract hand state (translation, rotation, pose) for left/right hands
       - Create state and action masks based on which hands to predict
    5. Run VLA inference to predict future hand actions (multiple samples for diversity)
    6. Reconstruct absolute hand trajectories from relative actions
    7. Visualize predicted hand motions using MANO hand model
    8. Generate grid layout video showing all samples and save to file
    9. Cleanup: Shutdown persistent services and free GPU memory

    """
    parser = argparse.ArgumentParser(description="Hand VLA inference and visualization.")
    
    # Model Configuration
    parser.add_argument('--config_path', type=str, required=True, help='Path to model configuration JSON file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint (overrides config)')
    parser.add_argument('--statistics_path', type=str, default=None, help='Path to normalization statistics JSON (overrides config)')
    
    # Input/Output
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image file')
    parser.add_argument('--hand_path', type=str, default=None, help='Path to hand state .npy file (optional, will run reconstruction if not provided)')
    parser.add_argument('--video_path', type=str, default='./example_human_inf.mp4', help='Path to save output visualization video')
    
    # Hand Reconstruction Models
    parser.add_argument('--hawor_model_path', type=str, default='./weights/hawor/checkpoints/hawor.ckpt', help='Path to HAWOR model weights')
    parser.add_argument('--detector_path', type=str, default='./weights/hawor/external/detector.pt', help='Path to hand detector model')
    parser.add_argument('--moge_model_name', type=str, default='Ruicheng/moge-2-vitl', help='MOGE model name from Hugging Face')
    parser.add_argument('--mano_path', type=str, default='/home/t-qixiuli/repo/VITRA/weights/mano', help='Path to MANO model files')
    # parser.add_argument('--output_path', type=str, default='./recon_results.npy', help='Path to save reconstruction results')
    
    # Prediction Settings
    parser.add_argument('--use_left', action='store_true', help='Enable left hand prediction')
    parser.add_argument('--use_right', action='store_true', help='Enable right hand prediction')
    parser.add_argument('--instruction', type=str, default="Left: Put the trash into the garbage. Right: None.", help='Text instruction for hand motion')
    parser.add_argument('--sample_times', type=int, default=4, help='Number of action samples to generate for diversity')
    parser.add_argument('--fps', type=int, default=8, help='Frames per second for output video')
    
    # Advanced Options
    parser.add_argument('--save_state_local', action='store_true', help='Save hand state locally as .npy file')

    # === Environment Configuration ===
    # Disable tokenizers parallelism to avoid deadlocks in multi-process data loading
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parser.parse_args()
    
    # Validate that at least one hand is selected
    if not args.use_left and not args.use_right:
        raise ValueError("At least one of --use_left or --use_right must be specified.")
    
    # Load configs
    configs = load_config(args.config_path)
    
    # Override config with command-line arguments if provided
    if args.model_path is not None:
        configs['model_load_path'] = args.model_path
    if args.statistics_path is not None:
        configs['statistics_path'] = args.statistics_path

    # Check if a precomputed hand reconstruction .npy (same stem as image) exists.
    image_path_obj = Path(args.image_path)
    npy_path = image_path_obj.with_suffix('.npy')

    # Initialize services
    print("Initializing services...")
    if npy_path.exists():
        # If precomputed .npy exists, load hand state from it
        # Hand State File Format
        # The hand state is stored as a .npy file containing a Python dictionary with the following structure:
        # Format: .npy
        # Content: dictionary with MANO-based hand pose parameters and camera FOV.
        # Structure:
        # {
        #     'left': {
        #         0: {
        #             'hand_pose': np.ndarray,      # [15, 3, 3] rotation matrices for MANO joints
        #             'global_orient': np.ndarray,  # [3, 3] global rotation matrix
        #             'transl': np.ndarray,         # [3] root translation in camera coordinates
        #             'beta': np.ndarray            # [10] MANO shape parameters
        #         }
        #     },
        #     'right': {                           # Same structure as 'left'
        #         0: {
        #             ...
        #         }
        #     },
        #     'fov_x': float                       # Horizontal field of view (in degrees)
        # }

        print(f"Found precomputed hand state results: {npy_path}. Using the state instead of running hand recon.")
        hand_data = np.load(npy_path, allow_pickle=True).item()

        hand_recon_service = None
    else:
        print(f"No precomputed hand state .npy found at {npy_path}. Starting hand reconstruction service.")
        
        # Start hand reconstruction service
        hand_recon_service = HandReconstructionService(args)
        hand_data = None


    # Start VLA service (normalizer and model are loaded inside the service)
    vla_service = VLAInferenceService(configs)
    
    # Visualization setup
    hand_config = HandConfig(args)
    hand_config.FPS = args.fps
    visualizer = HandVisualizer(hand_config, render_gradual_traj=False)

    try:
        if hand_data is None:
            # Run hand reconstruction using service
            print("Running hand reconstruction...")
            hand_data = hand_recon_service.reconstruct(args.image_path)
            if args.save_state_local:
                # Save hand state locally as .npy
                np.save(npy_path, hand_data, allow_pickle=True)
                print(f"Saved reconstructed hand state to {npy_path}")

        # Load and process image
        image = Image.open(args.image_path)
        ori_w, ori_h = image.size

        # Handle EXIF orientation if present (fixes image rotation issues)
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass  # If EXIF handling fails, continue with original image

        image_resized = resize_short_side_to_target(image, target=224)
        w, h = image_resized.size

        use_right = args.use_right
        use_left = args.use_left

        # Initialize state
        current_state_left = None
        current_state_right = None
        
        if use_right:
            current_state_right, beta_right, fov_x, _ = get_state(hand_data, hand_side='right')
        if use_left:
            current_state_left, beta_left, fov_x, _ = get_state(hand_data, hand_side='left')
        
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
        if current_state_left is None and current_state_right is None:
            raise ValueError("Both current_state_left and current_state_right are None")
        
        state_left = current_state_left if use_left else np.zeros_like(current_state_right)
        beta_left = beta_left if use_left else np.zeros_like(beta_right)
        state_right = current_state_right if use_right else np.zeros_like(current_state_left)
        beta_right = beta_right if use_right else np.zeros_like(beta_left)
        
        state = np.concatenate([state_left, beta_left, state_right, beta_right], axis=0)
        state_mask = np.array([use_left, use_right], dtype=bool)
        
        # Note: chunk_size needs to be determined from config
        chunk_size = configs.get('fwd_pred_next_n', 16)  # Default to 16 if not in config
        action_mask = np.tile(np.array([[use_left, use_right]], dtype=bool), (chunk_size, 1)) 

        fov = np.array([fov_x, fov_y], dtype=np.float32)
        # Convert resized image to numpy for VLA inference
        image_resized_np = np.array(image_resized)

        # Use instruction from command-line argument
        instruction = args.instruction

        # Model inference using service (includes normalization, padding, and unnormalization)
        print(f"Running VLA inference...")
        sample_times = args.sample_times
        unnorm_action = vla_service.predict(
            image=image_resized_np,
            instruction=instruction,
            state=state,
            state_mask=state_mask,
            action_mask=action_mask,
            fov=fov,
            num_ddim_steps=10,
            cfg_scale=5.0,
            sample_times=sample_times,
        )
        
        fx_exo = intrinsics[0, 0]
        fy_exo = intrinsics[1, 1]
        renderer = Renderer(w, h, (fx_exo, fy_exo), 'cuda')

        T = len(action_mask) + 1
        traj_right_list = np.zeros((sample_times, T, 51), dtype=np.float32)
        traj_left_list = np.zeros((sample_times, T, 51), dtype=np.float32)

        traj_mask = np.tile(np.array([[use_left, use_right]], dtype=bool), (T, 1)) 
        left_hand_mask = traj_mask[:, 0]
        right_hand_mask = traj_mask[:, 1]

        # Masks indicating which hands are used in the trajectory
        hand_mask = (left_hand_mask, right_hand_mask)

        all_rendered_frames = []
        
        # Reconstruct trajectories and visualize for each sample
        for i in range(sample_times):
            traj_right = traj_right_list[i]
            traj_left = traj_left_list[i]
            
            if use_left:
                traj_left = recon_traj(
                    state=state_left,
                    rel_action=unnorm_action[i, :, 0:51],
                )
            if use_right:
                traj_right = recon_traj(
                    state=state_right,
                    rel_action=unnorm_action[i, :, 51:102],
                )
        
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
            
            R_w2c = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()
            t_w2c = np.zeros((T, 3, 1), dtype=np.float32)

            extrinsics = (R_w2c, t_w2c)

            # Use resized image for visualization (convert RGB to BGR)
            image_bgr = image_resized_np[..., ::-1]
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
        
        # Concatenate all samples spatially into a single video
        # all_rendered_frames: list of sample_times frame lists
        # Each frame list has T frames
        num_frames = len(all_rendered_frames[0])
        
        # Determine grid layout (e.g., 2x2 for 4 samples)
        grid_cols = math.ceil(math.sqrt(sample_times))
        grid_rows = math.ceil(sample_times / grid_cols)
        
        # Combine different samples in one video
        combined_frames = []
        for frame_idx in range(num_frames):
            # Collect all sample frames at this time step
            sample_frames = [all_rendered_frames[i][frame_idx] for i in range(sample_times)]
            
            # Pad with black frames if needed to fill the grid
            while len(sample_frames) < grid_rows * grid_cols:
                black_frame = np.zeros_like(sample_frames[0])
                sample_frames.append(black_frame)
            
            # Arrange frames in grid
            rows = []
            for row_idx in range(grid_rows):
                row_frames = sample_frames[row_idx * grid_cols:(row_idx + 1) * grid_cols]
                row_concat = np.concatenate(row_frames, axis=1)  # Concatenate horizontally
                rows.append(row_concat)
            
            # Concatenate rows vertically
            combined_frame = np.concatenate(rows, axis=0)
            combined_frames.append(combined_frame)

        # Save combined video
        save_to_video(combined_frames, f'{args.video_path}', fps=hand_config.FPS)
        print(f"Combined video with {sample_times} samples saved to {args.video_path}")
    
    finally:
        # Cleanup persistent services
        print("Shutting down services...")
        if hand_recon_service is not None:
            hand_recon_service.shutdown()
        vla_service.shutdown()
        print("All services shut down successfully")
    

def get_state(hand_data, hand_side='right'):
    """
    Load and extract hand state from hand data.
    
    Args:
        hand_data (dict): Dictionary containing hand data
        hand_side (str): Which hand to extract, either 'left' or 'right'. Default is 'right'.
        
    Returns:
        tuple: (state_t0, beta, fov_x, None) where:
            - state_t0 (np.ndarray): Hand state [51] containing translation (3), 
                                     global rotation (3 euler angles), and hand pose (45 euler angles)
            - beta (np.ndarray): MANO shape parameters [10]
            - fov_x (float): Horizontal field of view in degrees
            - None: Placeholder for optional text annotations
    """
    if hand_side not in ['left', 'right']:
        raise ValueError(f"hand_side must be 'left' or 'right', got '{hand_side}'")
    
    hand_pose_t0 = hand_data[hand_side][0]['hand_pose']
    hand_pose_t0_euler = R.from_matrix(hand_pose_t0).as_euler('xyz', degrees=False) # [15, 3]
    hand_pose_t0_euler = hand_pose_t0_euler.reshape(-1)  # [45]
    global_orient_mat_t0 = hand_data[hand_side][0]['global_orient']
    R_t0_euler = R.from_matrix(global_orient_mat_t0).as_euler('xyz', degrees=False)  # [3]
    transl_t0 = hand_data[hand_side][0]['transl']  # [3]
    state_t0 = np.concatenate([transl_t0, R_t0_euler, hand_pose_t0_euler])  # [3+3+45=51]
    fov_x = hand_data['fov_x']

    return state_t0, hand_data[hand_side][0]['beta'], fov_x, None

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


def _hand_reconstruction_worker(args_dict, task_queue, result_queue):
    """
    Persistent worker for hand reconstruction that runs in a separate process.
    Keeps model loaded and processes multiple requests until shutdown signal.
    """
    from data.tools.hand_recon_core import Config, HandReconstructor
    
    hand_reconstructor = None
    
    try:
        # Reconstruct args object
        class ArgsObj:
            pass
        args_obj = ArgsObj()
        for key, value in args_dict.items():
            setattr(args_obj, key, value)
        
        # Initialize hand reconstructor once
        print("[HandRecon Process] Initializing hand reconstructor...")
        config = Config(args_obj)
        hand_reconstructor = HandReconstructor(config=config, device='cuda')
        print("[HandRecon Process] Hand reconstructor ready")
        
        # Signal ready
        result_queue.put({'type': 'ready'})
        
        # Process tasks in loop
        while True:
            task = task_queue.get()
            
            if task['type'] == 'shutdown':
                print("[HandRecon Process] Received shutdown signal")
                break
            
            elif task['type'] == 'reconstruct':
                try:
                    image_path = task['image_path']
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError(f"Failed to load image from {image_path}")
                    
                    image_list = [image]
                    recon_results = hand_reconstructor.recon(image_list)
                    
                    result_queue.put({
                        'type': 'result',
                        'success': True,
                        'data': recon_results
                    })
                    
                except Exception as e:
                    import traceback
                    result_queue.put({
                        'type': 'result',
                        'success': False,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
        
    except Exception as e:
        import traceback
        result_queue.put({
            'type': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
    
    finally:
        # Cleanup on shutdown
        if hand_reconstructor is not None:
            del hand_reconstructor
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[HandRecon Process] Cleaned up and exiting")


def _vla_inference_worker(configs_dict, task_queue, result_queue):
    """
    Persistent worker for VLA model inference that runs in a separate process.
    Keeps model loaded and processes multiple requests until shutdown signal.
    """
    from vitra.models import load_model
    from vitra.utils.data_utils import load_normalizer
    from vitra.datasets.human_dataset import pad_state_human, pad_action
    from vitra.datasets.dataset_utils import ActionFeature, StateFeature
    
    model = None
    normalizer = None
    
    try:
        # Load model and normalizer once
        print("[VLA Process] Loading VLA model...")
        model = load_model(configs_dict).cuda()
        model.eval()
        normalizer = load_normalizer(configs_dict)
        print(f"[VLA Process] VLA model ready.")
        
        # Signal ready
        result_queue.put({'type': 'ready'})
        
        # Process tasks in loop
        while True:
            task = task_queue.get()
            
            if task['type'] == 'shutdown':
                print("[VLA Process] Received shutdown signal")
                break
            
            elif task['type'] == 'predict':
                try:
                    image = task['image']
                    instruction = task['instruction']
                    state = task['state']
                    state_mask = task['state_mask']
                    action_mask = task['action_mask']
                    fov = task['fov']
                    num_ddim_steps = task.get('num_ddim_steps', 10)
                    cfg_scale = task.get('cfg_scale', 5.0)
                    sample_times = task.get('sample_times', 1)
                    
                    # Normalize state
                    norm_state = normalizer.normalize_state(state.copy())
                    
                    # Pad state and action
                    unified_action_dim = ActionFeature.ALL_FEATURES[1]  # 192
                    unified_state_dim = StateFeature.ALL_FEATURES[1]    # 212
                    
                    unified_state, unified_state_mask = pad_state_human(
                        state=norm_state,
                        state_mask=state_mask,
                        action_dim=normalizer.action_mean.shape[0],
                        state_dim=normalizer.state_mean.shape[0],
                        unified_state_dim=unified_state_dim,
                    )
                    _, unified_action_mask = pad_action(
                        actions=None,
                        action_mask=action_mask.copy(),
                        action_dim=normalizer.action_mean.shape[0],
                        unified_action_dim=unified_action_dim
                    )
                    
                    # Convert to torch and move to GPU
                    fov = torch.from_numpy(fov).unsqueeze(0)
                    unified_state = unified_state.unsqueeze(0)
                    unified_state_mask = unified_state_mask.unsqueeze(0)
                    unified_action_mask = unified_action_mask.unsqueeze(0)
                    
                    # Run inference
                    norm_action = model.predict_action(
                        image=image,
                        instruction=instruction,
                        current_state=unified_state,
                        current_state_mask=unified_state_mask,
                        action_mask_torch=unified_action_mask,
                        num_ddim_steps=num_ddim_steps,
                        cfg_scale=cfg_scale,
                        fov=fov,
                        sample_times=sample_times,
                    )
                    
                    # Extract and denormalize action
                    norm_action = norm_action[:, :, :102]
                    unnorm_action = normalizer.unnormalize_action(norm_action)
                    
                    # Convert to numpy for inter-process communication
                    if isinstance(unnorm_action, torch.Tensor):
                        unnorm_action_np = unnorm_action.cpu().numpy()
                    else:
                        unnorm_action_np = np.array(unnorm_action)
                    
                    result_queue.put({
                        'type': 'result',
                        'success': True,
                        'data': unnorm_action_np
                    })
                    
                except Exception as e:
                    import traceback
                    result_queue.put({
                        'type': 'result',
                        'success': False,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
        
    except Exception as e:
        import traceback
        result_queue.put({
            'type': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
    
    finally:
        # Cleanup on shutdown
        if model is not None:
            del model
        if normalizer is not None:
            del normalizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("[VLA Process] Cleaned up and exiting")


class HandReconstructionService:
    """Service wrapper for persistent hand reconstruction process"""
    
    def __init__(self, args):
        self.ctx = mp.get_context('spawn')
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        
        # Convert args to dict for pickling
        args_dict = {
            'hawor_model_path': args.hawor_model_path,
            'detector_path': args.detector_path,
            'moge_model_name': args.moge_model_name,
            'mano_path': args.mano_path,
        }
        
        # Start persistent process
        self.process = self.ctx.Process(
            target=_hand_reconstruction_worker,
            args=(args_dict, self.task_queue, self.result_queue)
        )
        self.process.start()
        
        # Wait for ready signal
        ready_msg = self.result_queue.get()
        if ready_msg['type'] == 'ready':
            print("Hand reconstruction service initialized")
        elif ready_msg['type'] == 'error':
            raise RuntimeError(f"Failed to initialize hand reconstruction: {ready_msg['error']}")
    
    def reconstruct(self, image_path):
        """Request hand reconstruction for an image"""
        self.task_queue.put({
            'type': 'reconstruct',
            'image_path': image_path
        })
        
        result = self.result_queue.get()
        if result['type'] == 'result' and result['success']:
            return result['data']
        else:
            raise RuntimeError(f"Hand reconstruction failed: {result.get('error', 'Unknown error')}")
    
    def shutdown(self):
        """Shutdown the persistent process"""
        self.task_queue.put({'type': 'shutdown'})
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()


class VLAInferenceService:
    """Service wrapper for persistent VLA inference process"""
    
    def __init__(self, configs):
        self.ctx = mp.get_context('spawn')
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        
        # Start persistent process
        self.process = self.ctx.Process(
            target=_vla_inference_worker,
            args=(configs, self.task_queue, self.result_queue)
        )
        self.process.start()
        
        # Wait for ready signal
        ready_msg = self.result_queue.get()
        if ready_msg['type'] == 'ready':
            print("VLA inference service initialized")
        elif ready_msg['type'] == 'error':
            raise RuntimeError(f"Failed to initialize VLA model: {ready_msg['error']}")
    
    def predict(self, image, instruction, state, state_mask, action_mask, 
                fov, num_ddim_steps=10, cfg_scale=5.0, sample_times=1):
        """Request action prediction with state normalization and padding"""

        self.task_queue.put({
            'type': 'predict',
            'image': image,
            'instruction': instruction,
            'state': state,
            'state_mask': state_mask,
            'action_mask': action_mask,
            'fov': fov,
            'num_ddim_steps': num_ddim_steps,
            'cfg_scale': cfg_scale,
            'sample_times': sample_times,
        })
        
        result = self.result_queue.get()
        if result['type'] == 'result' and result['success']:
            # Return unnormalized action as numpy array
            return result['data']
        else:
            raise RuntimeError(f"VLA inference failed: {result.get('error', 'Unknown error')}")
    
    def shutdown(self):
        """Shutdown the persistent process"""
        self.task_queue.put({'type': 'shutdown'})
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' to avoid CUDA fork issues
    mp.set_start_method('spawn', force=True)
    main()