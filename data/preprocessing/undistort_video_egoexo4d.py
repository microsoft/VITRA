import os
import cv2
import json
import argparse
from projectaria_tools.core import calibration
from utils import create_ffmpeg_writer, concatenate_ts_files

class Config:
    """
    Configuration settings for EgoExo4D video undistortion and processing.
    Paths and parameters are initialized with defaults but can be overridden 
    by command-line arguments.
    """
    def __init__(self, args=None):
        # --- Paths (Overridden by CLI arguments) ---
        self.VIDEO_ROOT = getattr(args, 'video_root', '/data2/v-leizhou/egoexo_data')
        self.INTRINSICS_ROOT = getattr(args, 'intrinsics_root', '/data2/v-leizhou/processed_data/aria_calib_json')
        self.SAVE_ROOT = getattr(args, 'save_root', 'debug_final_egoexo')
        
        # --- Processing Parameters (Overridden by CLI arguments) ---
        self.VIDEO_START_IDX = getattr(args, 'video_start', 0)
        self.VIDEO_END_IDX = getattr(args, 'video_end', None)
        self.BATCH_SIZE = getattr(args, 'batch_size', 1000)
        self.CRF = getattr(args, 'crf', 22)

def process_single_video(
    video_name: str, 
    aria_name: str, 
    video_root: str, 
    intrinsics_root: str, 
    save_root: str, 
    batch_size: int = 1000, 
    crf: int = 22
):
    """
    Processes a single EgoExo4D video, performs undistortion using 
    ProjectAriaTools, and saves the result in batches using FFmpeg.

    Args:
        video_name: Name of the video take folder.
        aria_name: Aria camera name used in the frame-aligned video path.
        video_root: Root directory of the input videos.
        intrinsics_root: Root directory of the intrinsics files (.json).
        save_root: Root directory for saving the output videos.
        batch_size: Number of frames to process and save per temporary TS file batch.
        crf: Constant Rate Factor (CRF) for FFmpeg encoding quality.
    """

    print(f'Processing {video_name}')

    # Construct the full video path based on EgoExo4D folder structure
    video_path = os.path.join(
        video_root, 
        'takes', 
        video_name, 
        'frame_aligned_videos', 
        f'{aria_name}_214-1.mp4'
    )
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load ground truth intrinsics info from JSON using ProjectAriaTools
    intrinsics_file_path = os.path.join(intrinsics_root, f'{video_name}.json')
    intrinsics_info = calibration.device_calibration_from_json(intrinsics_file_path).get_camera_calib("camera-rgb")

    # Use a fixed pinhole intrinsics for the output video resolution (1408x1408)
    pinhole = calibration.get_linear_camera_calibration(1408, 1408, 412.5) 

    # Initialize the first batch ffmpeg writer
    batch_number = 0
    writer = create_ffmpeg_writer(
        os.path.join(save_root, f'{video_name}_b{batch_number:04d}.ts'),
        width, height, fps, crf
    )

    idx = 0

    # Read and process frames
    while True:
        # Print progress in-place
        print(f'Processing {video_name} frame {idx} / {video_length}', end='\r') 
        ret, frame = cap.read()
        if not ret:
            # End of video stream: close the last writer
            writer.stdin.close()
            writer.wait()
            break

        # Undistort the frame using ProjectAriaTools' distortion function (original logic)
        undistorted_frame = calibration.distort_by_calibration(frame, pinhole, intrinsics_info)
        
        # Convert BGR to RGB before writing to ffmpeg (FFmpeg expects RGB)
        undistorted_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)

        # Write to ffmpeg stdin
        writer.stdin.write(undistorted_frame.tobytes())

        # Check if the current batch is complete
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
    Main function to parse arguments, load the Aria camera name mapping, 
    load the video list, and run the undistortion process.
    """
    parser = argparse.ArgumentParser(description='Undistort EgoExo4D videos using ProjectAriaTools calibration.')
    
    # Arguments corresponding to Config parameters
    parser.add_argument('--video_root', type=str, default='/data2/v-leizhou/egoexo_data', help='Root folder containing EgoExo4D video takes')
    parser.add_argument('--intrinsics_root', type=str, default='/data2/v-leizhou/processed_data/aria_calib_json', help='Root folder containing Aria calibration JSON files')
    parser.add_argument('--save_root', type=str, default='debug_final_egoexo', help='Root folder for saving output videos')
    parser.add_argument('--video_start', type=int, default=0, help='Start video index (inclusive)')
    parser.add_argument('--video_end', type=int, default=None, help='End video index (exclusive)')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of frames to be processed per batch (TS chunk)')
    parser.add_argument('--crf', type=int, default=22, help='CRF for ffmpeg encoding quality')
    
    args = parser.parse_args()
    
    # Initialize configuration from arguments
    config = Config(args)

    # Create the output directory if it doesn't exist
    os.makedirs(config.SAVE_ROOT, exist_ok=True)

    # Get all video names automatically (assuming subfolders are video names)
    try:
        video_names = sorted(os.listdir(os.path.join(config.VIDEO_ROOT, 'takes')))
        video_names = [name.split('.')[0] for name in video_names]
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
        
    # Load aria camera names from the JSON file (Preserves original hardcoded path)
    try:
        with open("./egoexo4d_aria_name.json", "r", encoding="utf-8") as f:
            aria_names = json.load(f)
    except FileNotFoundError:
        print("Error: The Aria name mapping file './egoexo4d_aria_name.json' was not found. Cannot proceed.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode the Aria name mapping file './egoexo4d_aria_name.json'. Cannot proceed.")
        return

    # Process videos
    for video_name in video_names_to_process:
        try:
            # Get aria name for the current video
            aria_name = aria_names[video_name]

            process_single_video(
                video_name,
                aria_name,
                config.VIDEO_ROOT,
                config.INTRINSICS_ROOT,
                config.SAVE_ROOT,
                config.BATCH_SIZE,
                config.CRF
            )
        except KeyError:
             # Handle missing Aria name for a video
             print(f'Error processing {video_name}: Aria name not found in the map file.')
             continue
        except Exception as e:
            # Catch and report other processing errors, then continue
            print(f'Error processing {video_name}: {e}')
            continue


if __name__ == '__main__':
    main()