import os
import ffmpeg
import tempfile
  
# ------------------------------------------------------
# Concatenate TS batch files into a final MP4
# ------------------------------------------------------
def concatenate_ts_files(folder, video_name, batch_counts):
    """Concatenate batch TS files into final mp4."""
    # Use a unique temporary file to avoid race conditions in multi-worker scenarios
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir=folder) as f:
        inputs_path = f.name
        # Create a file list for ffmpeg
        for i in range(batch_counts):
            f.write(f"file '{video_name}_b{i:04d}.ts'\n")

    try:
        # Merge using ffmpeg concat demuxer
        ffmpeg.input(inputs_path, format='concat', safe=0).output(
            os.path.join(folder, f'{video_name}.mp4'),
            c='copy'
        ).run()
    finally:
        # Cleanup temporary TS files and list file
        for i in range(batch_counts):
            ts_file = os.path.join(folder, f"{video_name}_b{i:04d}.ts")
            if os.path.exists(ts_file):
                os.remove(ts_file)
        if os.path.exists(inputs_path):
            os.remove(inputs_path)

# ------------------------------------------------------
# Create a new ffmpeg writer process
# ------------------------------------------------------
def create_ffmpeg_writer(output_path, width, height, fps, crf):
    """Spawn an ffmpeg async encoding process for writing raw frames."""
    return (
        ffmpeg.output(
            ffmpeg.input(
                'pipe:0',
                format='rawvideo',
                pix_fmt='rgb24',
                s=f'{width}x{height}',
                r=fps,
            ),
            output_path,
            **{
                'preset': 'medium',
                'pix_fmt': 'yuv420p',
                'b:v': '0',
                'c:v': 'libx264',
                'crf': str(crf),
                'r': fps,
            }
        )
        .overwrite_output()
        .run_async(quiet=True, pipe_stdin=True)
    )