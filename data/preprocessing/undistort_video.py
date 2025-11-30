import os
import numpy as np
import cv2
import argparse
from utils import create_ffmpeg_writer, concatenate_ts_files

class Config:
    """
    Configuration settings for video undistortion and processing.
    Paths and parameters are initialized with defaults but can be overridden 
    by command-line arguments.
    """
    def __init__(self, args=None):
        # --- Paths (Overridden by CLI arguments) ---
        self.VIDEO_ROOT = getattr(args, 'video_root', '/data1/yudeng/ego4d/full_scale')
        self.INTRINSICS_ROOT = getattr(args, 'intrinsics_root', '/data1/yudeng/ego4d/intrinsics_combine')
        self.SAVE_ROOT = getattr(args, 'save_root', 'debug_final')
        
        # --- Processing Parameters (Overridden by CLI arguments) ---
        self.VIDEO_START_IDX = getattr(args, 'video_start', 0)
        self.VIDEO_END_IDX = getattr(args, 'video_end', None)
        self.BATCH_SIZE = getattr(args, 'batch_size', 1000)
        self.CRF = getattr(args, 'crf', 22)

def prepare_undistort_maps(width: int, height: int, intrinsics_info: dict) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """
    Loads intrinsic parameters and prepares the undistortion and rectification
    maps (map1, map2) for an omnidirectional camera.

    Args:
        width: The width of the video frame.
        height: The height of the video frame.
        intrinsics_info: Dictionary containing 'intrinsics_ori' and 'intrinsics_new'.

    Returns:
        A tuple: (map1, map2, remap_flag).
        map1, map2: Undistortion maps (None if not needed).
        remap_flag: Boolean indicating if undistortion/remap is necessary (xi > 0).
    """
    intrinsics_ori = intrinsics_info['intrinsics_ori']
    intrinsics_new = intrinsics_info['intrinsics_new']

    K = intrinsics_ori['K'].astype(np.float32)
    D = intrinsics_ori['D'].astype(np.float32)
    xi = np.array(intrinsics_ori['xi']).astype(np.float32)

    new_K = intrinsics_new['K'].astype(np.float32)

    # Determine whether to use remap or not based on xi (remap_flag is True if xi > 0)
    remap_flag = (xi > 0)
    
    if remap_flag:
        # Initialize undistortion and rectification maps using omnidir model
        map1, map2 = cv2.omnidir.initUndistortRectifyMap(
            K, D, xi, np.eye(3), new_K, (width, height),
            cv2.CV_16SC2, cv2.omnidir.RECTIFY_PERSPECTIVE
        )
    else:
        map1, map2 = None, None

    return map1, map2, remap_flag

def process_single_video(
    video_name: str, 
    video_root: str, 
    intrinsics_root: str, 
    save_root: str, 
    batch_size: int = 1000, 
    crf: int = 22
):
    """
    Processes a single omnidirectional video, performs undistortion using 
    provided intrinsics, and saves the result in batches using FFmpeg.

    Args:
        video_name: Name of the video (without extension).
        video_root: Root directory of the input videos.
        intrinsics_root: Root directory of the intrinsics files (.npy).
        save_root: Root directory for saving the output videos.
        batch_size: Number of frames to process and save per temporary TS file batch.
        crf: Constant Rate Factor (CRF) for FFmpeg encoding quality.
    """

    print(f'Processing {video_name}')

    video_path = os.path.join(video_root, video_name + '.mp4')
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load intrinsics data
    intrinsics_path = os.path.join(intrinsics_root, f'{video_name}.npy')
    intrinsics_info = np.load(intrinsics_path, allow_pickle=True).item()

    # Prepare undistortion maps
    map1, map2, remap_flag = prepare_undistort_maps(width, height, intrinsics_info)

    # Initialize the first batch ffmpeg writer
    batch_number = 0
    writer = create_ffmpeg_writer(
        os.path.join(save_root, f'{video_name}_b{batch_number:04d}.ts'),
        width, height, fps, crf
    )

    idx = 0

    # Read and process frames
    while True:
        print(f'Processing {video_name} frame {idx} / {video_length}', end='\r')
        ret, frame = cap.read()
        if not ret:
            # End of video stream: close the last writer
            writer.stdin.close()
            writer.wait()
            break

        # Undistort the frame
        if remap_flag:
            undistorted_frame = cv2.remap(
                frame, map1, map2,
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT
            )
        else:
            # If no remap is required, use the original frame
            undistorted_frame = frame
        
        # Convert BGR to RGB before writing to ffmpeg (FFmpeg expects RGB)
        undistorted_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)

        # Write to ffmpeg stdin
        writer.stdin.write(undistorted_frame.tobytes())

        # Check if the current batch is complete (for idx + 1)
        if (idx + 1) % batch_size == 0:
            # Finalize the current batch writer
            writer.stdin.close()
            writer.wait()

            # Start the next batch writer
            batch_number += 1
            writer = create_ffmpeg_writer(
                os.path.join(save_root, f'{video_name}_b{batch_number:04d}.ts'),
                width, height, fps, crf
            )

        idx += 1

    cap.release()

    # Merge all temporary TS chunks into the final MP4 file
    concatenate_ts_files(save_root, video_name, batch_number + 1)

def main():
    """
    Main function to parse arguments, load video list, and run the 
    undistortion process for the specified range of videos.
    """
    parser = argparse.ArgumentParser(description='Undistort videos using omnidirectional camera intrinsics.')
    
    # Arguments corresponding to Config parameters
    parser.add_argument('--video_root', type=str, default='/data1/yudeng/ego4d/full_scale', help='Folder containing input videos')
    parser.add_argument('--intrinsics_root', type=str, default='/data1/yudeng/ego4d/intrinsics_combine', help='Folder containing intrinsics info')
    parser.add_argument('--save_root', type=str, default='debug_final22', help='Folder for saving output videos')
    parser.add_argument('--video_start', type=int, default=0, help='Start video index (inclusive)')
    parser.add_argument('--video_end', type=int, default=None, help='End video index (exclusive)')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of frames to be processed per batch (TS chunk)')
    parser.add_argument('--crf', type=int, default=22, help='CRF for ffmpeg encoding quality')
    
    args = parser.parse_args()
    
    # Initialize configuration from arguments
    config = Config(args)

    # Create the output directory if it doesn't exist
    os.makedirs(config.SAVE_ROOT, exist_ok=True)

    # Get all video names automatically
    try:
        video_names = sorted(os.listdir(config.VIDEO_ROOT))
        video_names = [name.split('.')[0] for name in video_names if name.endswith('.mp4')]
    except FileNotFoundError:
        print(f"Error: Video root directory not found at {config.VIDEO_ROOT}. Cannot proceed.")
        return

    if config.VIDEO_END_IDX is None:
        end_idx = len(video_names)
    else:
        end_idx = config.VIDEO_END_IDX
    
    video_names_to_process = video_names[config.VIDEO_START_IDX:end_idx]
    
    if not video_names_to_process:
        print("No videos found to process in the specified range.")
        return

    # Process videos
    for video_name in video_names_to_process:
        try:
            process_single_video(
                video_name,
                config.VIDEO_ROOT,
                config.INTRINSICS_ROOT,
                config.SAVE_ROOT,
                config.BATCH_SIZE,
                config.CRF
            )
        except Exception as e:
            # Print error and continue to the next video (preserves original exception handling)
            print(f'Error processing {video_name}: {e}')
            continue


if __name__ == '__main__':
    main()