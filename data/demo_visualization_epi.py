# Demo script for visualizing hand VLA episodes.
import sys
import os

import sys
import os

# Adjust system path to include the vitra root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
vitra_root = os.path.dirname(current_dir)
if vitra_root not in sys.path:
    sys.path.insert(0, vitra_root)

import argparse

from visualization.visualize_core import Config, HandVisualizer

# --- Main Execution Function ---
def main():
    """Main execution function, including argument parsing."""
    parser = argparse.ArgumentParser(description="Visualize hand VLA episodes with customizable paths.")
    
    # Path Arguments
    parser.add_argument(
        '--video_root',
        type=str,
        default='data/examples/videos',
        help='Root directory containing the video files.'
    )
    parser.add_argument(
        '--label_root',
        type=str,
        default='data/examples/annotations',
        help='Root directory containing the episode label (.npy) files.'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='data/examples/visualize',
        help='Directory to save the output visualization videos.'
    )
    parser.add_argument(
        '--mano_model_path',
        type=str,
        default='./weights/mano',
        help='Path to the MANO model files.'
    )
    
    # Visualization Arguments
    parser.add_argument(
        '--render_gradual_traj',
        action='store_true',
        help='Set flag to render a gradual trajectory (full mode).'
    )

    args = parser.parse_args()

    # 1. Initialize Visualizer with parsed arguments
    config = Config(args)
    
    # Ensure save path exists
    os.makedirs(config.SAVE_PATH, exist_ok=True)
    
    visualizer = HandVisualizer(config, render_gradual_traj=args.render_gradual_traj)

    # 2. Load Episode Names
    try:
        all_episode_names_npy = sorted(os.listdir(args.label_root))
        all_episode_names = [n.split('.npy')[0] for n in all_episode_names_npy]

    except FileNotFoundError:
        print(f"Error: Episode list directory not found at {args.label_root}. Cannot proceed.")
        return

    # 3. Process All Episodes
    print(f"--- Running Hand Visualizer ---")
    print(f"Video Root: {config.VIDEO_ROOT}")
    print(f"Label Root: {config.LABEL_ROOT}")
    print(f"Save Path: {config.SAVE_PATH}")
    print(f"MANO Model Path: {config.MANO_MODEL_PATH}")
    print(f"Rendering Gradual Trajectory: {args.render_gradual_traj}")
    print(f"-------------------------------")

    for episode_name in all_episode_names:
        visualizer.process_episode(episode_name)


if __name__ == '__main__':
    main()