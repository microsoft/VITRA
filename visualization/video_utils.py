# Utils for video processing, frame manipulation, and visualization.
import cv2
import numpy as np
from typing import List
import imageio

def rotate_frame(frame: np.ndarray) -> np.ndarray:
    """Rotate a frame 90 degrees clockwise."""
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

def center_crop_image(frame: np.ndarray, crop_percent: float = 1.0) -> np.ndarray:
    """
    Center crop an image to a specified percentage of its original size.

    Args:
        frame (np.ndarray): Input image.
        crop_percent (float): Percentage of original size to crop to (0 < crop_percent <= 1).

    Returns:
        np.ndarray: Cropped image.
    """
    if crop_percent == 1.0:  
        return frame

    # Get original dimensions  
    original_height, original_width = frame.shape[:2]  
    
    # Calculate new dimensions  
    new_width = int(original_width * crop_percent)  
    new_height = int(original_height * crop_percent)  
    
    # Calculate top-left corner of the cropping box  
    start_x = (original_width - new_width) // 2  
    start_y = (original_height - new_height) // 2  
    
    # Perform the crop  
    cropped_frame = frame[start_y:start_y + new_height, start_x:start_x + new_width]  

    return cropped_frame

def read_video_frames(
    cap,
    start_frame: int = None,
    end_frame: int = None,
    interval: int = 1,
    rotate: bool = False,
    crop_percent: float = 1.0
    ) -> List[np.ndarray]:
    """
    Read frames from a video capture object with optional rotation and cropping.

    Args:
        cap: OpenCV VideoCapture object.
        start_frame (int): Starting frame index.
        end_frame (int): Ending frame index.
        interval (int): Frame interval for sampling.
        rotate (bool): Whether to rotate frames 90 degrees clockwise.
        crop_percent (float): Center crop percentage.

    Returns:
        List[np.ndarray]: List of frames.
    """
    frame_count = 0  
    frame_list = []

    # If a start frame is specified, move the video pointer
    if start_frame is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) 
        frame_count += start_frame
    
    # If no end frame is specified, read until the end of the video
    if end_frame is None:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read frames in a loop
    while frame_count < end_frame:  
        ret, frame = cap.read()  
        if not ret:  
            break  

        # Process frames at the specified interval
        if frame_count % interval == 0:

            # Optionally rotate the frame
            if rotate:
                frame = rotate_frame(frame)
            
            # Optionally center crop the frame
            frame = center_crop_image(frame, crop_percent=crop_percent)
            frame_list.append(frame) 

        frame_count += 1  

    return frame_list

def save_to_video(frames: List[np.ndarray], output_path: str, fps: int = 30):
    """
    Save a list of frames to a video file.

    Args:
        frames (List[np.ndarray]): List of frames.
        output_path (str): Output video path.
        fps (int): Frames per second.
    """
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264')

def resize_frames_to_long_side(frames: List[np.ndarray], target_long_side: int) -> List[np.ndarray]:
    """
    Resize frames so the longer side matches the target size, preserving aspect ratio.

    Args:
        frames (List[np.ndarray]): List of frames.
        target_long_side (int): Desired length of the longer side.

    Returns:
        List[np.ndarray]: Resized frames.
    """

    # If target_long_side is None, return original frames without resizing
    if target_long_side is None:
        return frames
    
    resized_frames = []  
    # Loop over each frame
    for frame in frames:  
        height, width = frame.shape[:2]  
          
        # Determine the scaling factor and new dimensions  
        if width > height:  
            scale_factor = target_long_side / width  
        else:  
            scale_factor = target_long_side / height  
          
        new_width = int(width * scale_factor)  
        new_height = int(height * scale_factor)  
          
        # Resize the frame while maintaining the aspect ratio  
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)  
        resized_frames.append(resized_frame)  
      
    return resized_frames  

def sample_frames_evenly(video_frames: List[np.ndarray], num_frames: int) -> List[np.ndarray]:
    """
    Sample frames evenly from a video sequence.

    Args:
        video_frames (List[np.ndarray]): List of frames.
        num_frames (int): Number of frames to sample.

    Returns:
        List[np.ndarray]: Sampled frames.
    """
    total = len(video_frames)
    if num_frames >= total:
        return video_frames.copy() 

    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    return [video_frames[i] for i in indices]

def wrap_text(text: str, max_width: int, font, font_scale: float) -> List[str]:
    """  
    Wraps text to fit within a given width.  
    
    Args:  
        text (str): The text to wrap.  
        max_width (int): The maximum width in pixels.  
        font: The font to use.  
        font_scale (float): The scale of the font.  
    
    Returns:  
        List of lines of wrapped text.  
    """  
    words = text.split(' ')  
    lines = []  
    current_line = ''  
    
    for word in words:  
        test_line = current_line + word + ' '  
        # Measure the width of the line  
        size = cv2.getTextSize(test_line, font, font_scale, 1)[0]  
        if size[0] > max_width:  
            lines.append(current_line.strip())  
            current_line = word + ' '  
        else:  
            current_line = test_line  
    
    if current_line:  
        lines.append(current_line.strip())  
    
    return lines  

def add_overlay_text(frame: np.ndarray, caption: str) -> np.ndarray:
    """
    Add overlay text to a frame.

    Args:
        frame (np.ndarray): Input image.
        caption (str): Text to overlay.

    Returns:
        np.ndarray: Frame with text.
    """

    w = frame.shape[1]  # Width of the frame
    y0, dy = 30, 20  # Starting Y position and line height  
    for i, line in enumerate(wrap_text(caption, w, cv2.FONT_HERSHEY_SIMPLEX, 1.0)):  
        y = y0 + i * dy  
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  
    return frame