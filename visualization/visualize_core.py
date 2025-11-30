import os
import cv2
import numpy as np
import torch
import matplotlib as mpl

from .video_utils import (
    read_video_frames,
    resize_frames_to_long_side,
    save_to_video,
    add_overlay_text
)
from libs.models.mano_wrapper import MANO
from .render_utils import Renderer

class Config:
    """
    Configuration class for file paths, parameters, and visual settings.
    Paths are initialized with default values but can be overridden by arguments.
    """
    def __init__(self, args=None):
        # --- Paths (Overridden by CLI arguments) ---
        self.VIDEO_ROOT = getattr(args, 'video_root', 'data/examples/videos')
        self.LABEL_ROOT = getattr(args, 'label_root', 'data/examples/annotations')
        self.SAVE_PATH = getattr(args, 'save_path', 'data/examples/visualize')
        self.MANO_MODEL_PATH = getattr(args, 'mano_model_path', './weights/mano')

        # --- Fixed Parameters ---
        self.RENDER_SIZE_LONG_SIDE = 480
        self.FPS = 15

        # --- Fixed Colors and CMAPs ---
        self.LEFT_CMAP = "inferno"
        self.RIGHT_CMAP = "inferno"

        # Base colors for the hands
        self.LEFT_COLOR = np.array([0.6594, 0.6259, 0.7451])
        self.RIGHT_COLOR = np.array([0.4078, 0.4980, 0.7451])


class HandVisualizer:
    """
    Main class for loading data, configuring the renderer, and visualizing
    the hand episode, including mesh and trajectory.
    """
    def __init__(self, config: Config, render_gradual_traj: bool = False):
        self.config = config
        self.render_gradual_traj = render_gradual_traj
        self.all_modes = ['cam', 'first']
        if self.render_gradual_traj:
            self.all_modes = ['cam', 'full', 'first']

        # Initialize MANO and faces (use right hand model for both hands by default)
        self.mano = MANO(model_path=self.config.MANO_MODEL_PATH).cuda()
        faces_right = torch.from_numpy(self.mano.faces).float().cuda()
        # MANO faces are defined for right hand, left hand faces need vertex order flip
        self.faces_left = faces_right[:, [0, 2, 1]]
        self.faces_right = faces_right

    def _render_hand_trajectory(self, video_frames, hand_traj_wordspace, hand_mask, extrinsics, renderer: Renderer, mode: str):
        """
        Renders hand mesh for one frame or hand trajectory across multiple frames,
        depending on the mode ('cam', 'first', 'full').
        """
        verts_left_worldspace, verts_right_worldspace = hand_traj_wordspace
        left_hand_mask, right_hand_mask = hand_mask
        R_w2c, t_w2c = extrinsics

        num_total_frames = len(video_frames)
        all_save_frames = []

        # Determine rendering parameters based on mode
        if mode == 'cam':
            # Renders only the current frame's mesh
            num_loop_frames = num_total_frames
            # Single color for all frames
            left_colors = self.config.LEFT_COLOR[np.newaxis, :].repeat(num_total_frames, axis=0)
            right_colors = self.config.RIGHT_COLOR[np.newaxis, :].repeat(num_total_frames, axis=0)
        elif mode == 'first':
            # Renders the full trajectory onto the first frame
            num_loop_frames = 1
            left_colors = self.config.LEFT_COLOR[np.newaxis, :].repeat(num_total_frames, axis=0)
            right_colors = self.config.RIGHT_COLOR[np.newaxis, :].repeat(num_total_frames, axis=0)
        elif mode == 'full':
            # Renders a gradual trajectory for each frame
            num_loop_frames = num_total_frames
            # Generate color sequence for trajectory
            left_colors, right_colors = generate_hand_colors(num_total_frames, self.config.LEFT_CMAP, self.config.RIGHT_CMAP)
        else:
            raise ValueError(f'Unknown rendering mode: {mode}')

        for current_frame_idx in range(num_loop_frames):

            if not mode == 'first':
                print(f'Processing frame {current_frame_idx + 1}/{num_loop_frames}', end='\r')
                # Start with the base video frame (copied and normalized)
                curr_img_overlay = video_frames[current_frame_idx].copy().astype(np.float32) / 255.0

            # Calculate camera space vertices for all frames relative to the *current* camera pose
            R_w2c_cur = R_w2c[current_frame_idx]
            t_w2c_cur = t_w2c[current_frame_idx]

            # R_w2c_cur: (3, 3), t_w2c_cur: (3, 1). Need to broadcast to all frames (T)
            verts_left_camspace = (
                R_w2c_cur @ verts_left_worldspace.transpose(0, 2, 1) + t_w2c_cur
            ).transpose(0, 2, 1)
            verts_right_camspace = (
                R_w2c_cur @ verts_right_worldspace.transpose(0, 2, 1) + t_w2c_cur
            ).transpose(0, 2, 1)

            # Determine the segment of the trajectory to render for the current frame
            if mode == 'cam':
                # Render only the current frame's mesh (from index to index)
                start_traj_idx = current_frame_idx
                end_traj_idx = current_frame_idx + 1
                transparency = [1.0]
            elif mode == 'first':
                # Render full trajectory on frame 0 (index 0 to T)
                start_traj_idx = 0
                end_traj_idx = num_total_frames
                transparency = [1.0] * (end_traj_idx - start_traj_idx)
                # The loop only runs once for mode='first'
                if current_frame_idx > 0: continue
            elif mode == 'full':
                # Render gradual trajectory (from index to T)
                start_traj_idx = current_frame_idx
                end_traj_idx = num_total_frames
                # Gradual transparency for the trajectory: older points are more transparent
                transparency = np.linspace(0.4, 0.7, end_traj_idx - start_traj_idx)
            else:
                raise ValueError(f'Unknown rendering mode: {mode}')

            # Iterate over the trajectory segment
            for traj_idx, kk in enumerate(range(start_traj_idx, end_traj_idx)):

                if mode == 'first':
                    print(f'Processing frame {traj_idx + 1}/{num_total_frames}', end='\r')
                    curr_img_overlay = video_frames[current_frame_idx].copy().astype(np.float32)/255

                # Get hand data for the trajectory point 'kk'
                left_mask_k = left_hand_mask[kk]
                right_mask_k = right_hand_mask[kk]
                transp_k = transparency[traj_idx] if len(transparency) > traj_idx else 1.0

                left_verts_list, left_color_list, left_face_list = ([], [], [])
                right_verts_list, right_color_list, right_face_list = ([], [], [])

                if left_mask_k != 0:
                    left_verts_list = [torch.from_numpy(verts_left_camspace[kk]).float().cuda()]
                    # Repeat color for all 778 vertices
                    left_color_list = [torch.from_numpy(left_colors[kk]).float().unsqueeze(0).repeat(778, 1).cuda()]
                    left_face_list = [self.faces_left]

                if right_mask_k != 0:
                    right_verts_list = [torch.from_numpy(verts_right_camspace[kk]).float().cuda()]
                    right_color_list = [torch.from_numpy(right_colors[kk]).float().unsqueeze(0).repeat(778, 1).cuda()]
                    right_face_list = [self.faces_right]

                verts_list  = left_verts_list + right_verts_list
                faces_list  = left_face_list + right_face_list
                colors_list = left_color_list + right_color_list

                if verts_list:
                    # Render the mesh onto the current image
                    rend, mask = renderer.render(verts_list, faces_list, colors_list)
                    rend = rend[..., ::-1]  # RGB to BGR

                    color_mesh = rend.astype(np.float32) / 255.0
                    valid_mask = mask[..., None].astype(np.float32)

                    # Alpha blending for the mesh overlay:
                    # new_image = base * (1-mask) + mesh * mask * alpha + base * mask * (1-alpha)
                    curr_img_overlay = (
                        curr_img_overlay[:, :, :3] * (1 - valid_mask) +
                        color_mesh[:, :, :3] * valid_mask * transp_k +
                        curr_img_overlay[:, :, :3] * valid_mask * (1 - transp_k)
                    )
                if mode == 'first':
                    # Finalize image format and color space
                    final_frame = (curr_img_overlay * 255).astype(np.uint8)
                    final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                    all_save_frames.append(final_frame)
            
            if mode == 'cam' or mode == 'full':
                # Finalize image format and color space
                final_frame = (curr_img_overlay * 255).astype(np.uint8)
                final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                all_save_frames.append(final_frame)

        print(f'Finished rendering with mode: {mode}')
        return all_save_frames

    def process_episode(self, episode_name: str):
            """Loads data and orchestrates the visualization process for a single episode."""
            print(f'\nProcessing episode: {episode_name}')

            # 1. Load Paths and Check Existence
            dataset_name = episode_name.split('_')[0]
            ep_name = episode_name.split('_')[-2] + '_' + episode_name.split('_')[-1]
            video_name = episode_name.replace(f'{dataset_name}_', '').replace(f'_{ep_name}', '')
            video_path = os.path.join(self.config.VIDEO_ROOT, f'{video_name}.mp4')
            label_path = os.path.join(self.config.LABEL_ROOT, episode_name + '.npy')

            if not os.path.exists(label_path):
                print(f'Episode file {label_path} does not exist, skipping...')
                return

            # 2. Load Episode Data
            cap = cv2.VideoCapture(video_path)
            episode_info = np.load(label_path, allow_pickle=True).item()

            start_frame, end_frame = get_frame_interval(episode_info)
            R_w2c, t_w2c, normalized_intrinsics = get_camera_info(episode_info)
            caption_left, caption_right, hand_type = get_caption_info(episode_info)
            (verts_left_worldspace, left_hand_mask), (verts_right_worldspace, right_hand_mask) = \
                get_hand_labels(episode_info, self.mano)

            # 3. Read and Resize Video Frames
            video_frames = read_video_frames(cap, start_frame=start_frame, end_frame=end_frame, interval=1)
            resize_video_frames = resize_frames_to_long_side(video_frames, self.config.RENDER_SIZE_LONG_SIDE)
            H, W, _ = resize_video_frames[0].shape

            # 4. Initialize Renderer
            # Denormalize intrinsics based on the new frame size (W, H)
            intrinsics_denorm = normalized_intrinsics.copy()
            intrinsics_denorm[0] *= W
            intrinsics_denorm[1] *= H
            fx_exo = intrinsics_denorm[0, 0]
            fy_exo = intrinsics_denorm[1, 1]

            renderer = Renderer(W, H, (fx_exo, fy_exo), 'cuda')

            # 5. Render Hands for All Modes
            all_rendered_frames = []
            hand_traj_wordspace = (verts_left_worldspace, verts_right_worldspace)
            hand_mask = (left_hand_mask, right_hand_mask)
            extrinsics = (R_w2c, t_w2c)

            for mode in self.all_modes:
                save_frames = self._render_hand_trajectory(
                    resize_video_frames,
                    hand_traj_wordspace,
                    hand_mask,
                    extrinsics,
                    renderer,
                    mode=mode
                )
                all_rendered_frames.append(save_frames)

            # 6. Concatenate Frames and Add Captions
            final_save_frames = []
            num_frames = len(all_rendered_frames[0])

            # Select primary caption and extract opposite intervals for text lookup
            caption_primary = caption_right if hand_type == 'right' else caption_left
            caption_opposite = caption_left if hand_type == 'right' else caption_right
            opposite_intervals = [interval for _, interval in caption_opposite]

            for frame_idx in range(num_frames):
                # Concatenate frames from different modes side by side
                curr_img_overlay = np.concatenate(
                    [all_rendered_frames[mode_idx][frame_idx] for mode_idx in range(len(self.all_modes))],
                    axis=1
                )

                # Get caption for the primary hand (assumes primary caption only has one interval: [0])
                overlay_text_primary = caption_primary[0][0]

                # Get caption for the opposite hand based on the current frame index
                opposite_idx = find_caption_index(frame_idx, opposite_intervals)
                overlay_text_opposite = caption_opposite[opposite_idx][0] if opposite_idx is not None else 'None.'

                # Format and add the full overlay text
                overlay_text_full = generate_overlay_text(
                    overlay_text_primary, 
                    overlay_text_opposite, 
                    hand_type
                )
                add_overlay_text(curr_img_overlay, overlay_text_full)

                final_save_frames.append(curr_img_overlay)

            # 7. Save Final Video
            os.makedirs(self.config.SAVE_PATH, exist_ok=True)
            save_to_video(final_save_frames, f'{self.config.SAVE_PATH}/{episode_name}.mp4', fps=self.config.FPS)
            print(f'\nSuccessfully saved episode to {self.config.SAVE_PATH}/{episode_name}.mp4')

def find_caption_index(frame_index: int, intervals: list[tuple[int, int]]) -> int | None:
    """Finds the interval index for a given frame index."""
    for idx, (start, end) in enumerate(intervals):
        if start <= frame_index <= end:
            return idx
    return None

def generate_hand_colors(T: int, left_cmap: str, right_cmap: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates RGB color sequences for left and right hands over T frames.
    Returns colors in shape (T, 3), normalized 0-1, based on the specified colormaps.
    """
    t_norm = np.linspace(0, 0.95, T)
    left_colors = mpl.colormaps.get_cmap(left_cmap)(t_norm)[:, :3]
    right_colors = mpl.colormaps.get_cmap(right_cmap)(t_norm)[:, :3]
    return left_colors, right_colors

def get_frame_interval(episode_info: dict) -> tuple[int, int]:
    """Extracts start (inclusive) and end (exclusive) frame indices from episode info."""
    video_decode_frames = episode_info['video_decode_frame']
    start_frame = video_decode_frames[0]
    end_frame = video_decode_frames[-1] + 1
    return start_frame, end_frame

def normalize_camera_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    """
    Normalizes intrinsics based on the assumption that the principal point
    is at the image center (image size is 2*cx, 2*cy).
    """
    # Create a deep copy to avoid modifying the original array
    normalized_intrinsics = intrinsics.copy()
    normalized_intrinsics[0] /= normalized_intrinsics[0, 2] * 2
    normalized_intrinsics[1] /= normalized_intrinsics[1, 2] * 2
    return normalized_intrinsics

def get_camera_info(episode_info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts and normalizes camera intrinsics and extrinsics (world-to-cam).
    """
    extrinsics = episode_info['extrinsics']  # world2cam, shape (T, 4, 4)
    R_w2c = extrinsics[:, :3, :3].copy()
    t_w2c = extrinsics[:, :3, 3:].copy()  # shape (T, 3, 1)

    intrinsics = episode_info['intrinsics'].copy()
    normalized_intrinsics = normalize_camera_intrinsics(intrinsics)

    return R_w2c, t_w2c, normalized_intrinsics

def get_caption_info(episode_info: dict) -> tuple[list, list, str]:
    """
    Extracts and formats caption information for left and right hands.
    Adds a large interval if captions are empty to cover all frames.
    """
    hand_type = episode_info['anno_type']

    caption_right = episode_info['text'].get('right', [])
    caption_left = episode_info['text'].get('left', [])

    # Ensure captions are not empty to simplify downstream logic
    if not caption_right:
        caption_right = [['None.', (0, 10000)]] # large interval to cover all frames
    if not caption_left:
        caption_left = [['None.', (0, 10000)]]

    return caption_left, caption_right, hand_type

def get_hand_labels(episode_info: dict, mano: MANO):
    """
    Processes hand labels (pose, shape, translation, orientation) through the MANO model
    to obtain hand vertices in world space.
    """
    left_labels = episode_info['left']
    right_labels = episode_info['right']

    # --- Left Hand Processing ---
    left_hand_mask = left_labels['kept_frames']
    verts_left, _ = process_single_hand_labels(left_labels, left_hand_mask, mano, is_left=True)

    # --- Right Hand Processing ---
    right_hand_mask = right_labels['kept_frames']
    verts_right, _ = process_single_hand_labels(right_labels, right_hand_mask, mano)
    
    return (verts_left, left_hand_mask), (verts_right, right_hand_mask)

def process_single_hand_labels(hand_labels: dict, hand_mask: np.ndarray, mano: MANO, is_left: bool = False):
    """
    Helper function to compute MANO vertices for a single hand (left or right).
    """
    T = len(hand_mask)
    
    wrist_worldspace = hand_labels['transl_worldspace'].reshape(-1, 1, 3)
    wrist_orientation = hand_labels['global_orient_worldspace']
    beta = hand_labels['beta']
    pose = hand_labels['hand_pose']

    # Set pose to identity for masked frames (no hand present)
    identity = np.eye(3, dtype=pose.dtype)
    identity_block = np.broadcast_to(identity, (pose.shape[1], 3, 3))
    mask_indices = (hand_mask == 0)
    if np.any(mask_indices):
        pose[mask_indices] = identity_block
    # pose[hand_mask == 0] = identity_block

    beta_torch = torch.from_numpy(beta).float().cuda().unsqueeze(0).repeat(T, 1)
    pose_torch = torch.from_numpy(pose).float().cuda()
    
    # Placeholder for global orientation in MANO forward pass (will be applied manually later)
    global_rot_placeholder = torch.eye(3).float().unsqueeze(0).unsqueeze(0).cuda().repeat(T, 1, 1, 1)
    # MANO forward pass
    mano_out = mano(betas=beta_torch, hand_pose=pose_torch, global_orient=global_rot_placeholder)
    
    verts = mano_out.vertices.cpu().numpy()
    joints = mano_out.joints.cpu().numpy()

    # Apply the wrist orientation and translation to get world space coordinates
    # X-axis flip for the left hand.
    if is_left:
        verts[:, :, 0] *= -1
        joints[:, :, 0] *= -1

    # World space transformation: R @ (V - J0) + T
    # (T, 778, 3) = (T, 3, 3) @ (T, 3, 778) + (T, 3, 1) -> (T, 3, 778) -> (T, 778, 3)
    verts_worldspace = (
        wrist_orientation @ 
        (verts - joints[:, 0][:, None]).transpose(0, 2, 1)
    ).transpose(0, 2, 1) + wrist_worldspace

    return verts_worldspace, joints[:, 0]

def generate_overlay_text(overlay_text: str, overlay_text_opposite: str, hand_type: str) -> str:
    """Formats the caption string based on the primary hand type."""
    if hand_type == 'right':
        return f'Left: {overlay_text_opposite} | Right: {overlay_text}'
    else:
        return f'Left: {overlay_text} | Right: {overlay_text_opposite}'