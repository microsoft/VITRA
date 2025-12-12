
# Instructions for Preparing Human Hand V-L-A Data

This folder provides essential documentation and scripts for the human hand VLA data used in this project.
**Please note that the metadata we provide may continue to receive updates in the future. Based on manual inspection, the current version achieves roughly 90% annotation accuracy, and we plan to further improve the metadata quality in future updates.**

The contents of this folder are as follows:

## 📑 Table of Contents
- [1. Prerequisites](#1-prerequisites)
- [2. Data Download](#2-data-download)
- [3. Video Preprocessing](#3-video-preprocessing)
- [4. Metadata Structure](#4-metadata-structure)
- [5. Data Visualization](#5-data-visualization)

---
## 1. Prerequisites
Our data preprocessing and visualization rely on several dependencies that need to be prepared in advance. If you have already completed the installation steps in **1.2 Visualization Requirements** of the [`readme.md`](../readme.md), you can skip this section.

### Python Libraries
[PyTorch3D](https://github.com/facebookresearch/pytorch3d?tab=readme-ov-file) is required for visualization. You can install it according to the official guide, or simply run the command below:
```bash
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@stable#egg=pytorch3d  
```
[FFmpeg](https://github.com/FFmpeg/FFmpeg) is also required for video processing:
```bash
sudo apt install ffmpeg
pip install ffmpeg-python
```

Other Python dependencies can be installed using the following command:
```bash
pip install projectaria_tools smplx
pip install --no-build-isolation git+https://github.com/mattloper/chumpy#egg=chumpy
```
### MANO Hand Model

Our reconstructed hand labels are based on the MANO hand model. **We only require the right hand model.** The model parameters can be downloaded from the [official website](https://mano.is.tue.mpg.de/index.html) and organized in the following structure ([mano_mean_params.npz](../weights/mano/mano_mean_params.npz) is already included in our repo):
```
weights/
└── mano/
    ├── MANO_RIGHT.pkl
    └── mano_mean_params.npz
```

---

## 2. Data Download

### Meta Information

We provide the metadata for the human V-L-A episodes we constructed, which can be downloaded from [this link](https://huggingface.co/datasets/VITRA-VLA/VITRA-1M). Each metadata entry contains the segmentation information of the corresponding V-L-A episode, language descriptions, as well as reconstructed camera parameters and 3D hand information. The detailed structure of the metadata can be found at [Metadata Structure](#4-metadata-structure). The total size of all metadata is approximately 100 GB.

After extracting the files, the downloaded metadata will have the following structure:
```
Metadata/
├── {dataset_name1}/
│   ├── episode_frame_index.npz
|   └── episodic_annotations/
│       ├── {dataset_name1}_{video_name1}_ep_{000000}.npy
│       ├── {dataset_name1}_{video_name1}_ep_{000001}.npy
│       ├── {dataset_name1}_{video_name1}_ep_{000002}.npy
│       ├── {dataset_name1}_{video_name2}_ep_{000000}.npy
│       ├── {dataset_name1}_{video_name2}_ep_{000001}.npy
│       └── ...
├── {dataset_name2}/
│   └── ...
```
Here, {dataset_name} indicates which dataset the episode belongs to, {video_name} corresponds to the name of the original raw video, and ep_{000000} is the episode’s index.

### Videos

Our project currently uses videos collected from four sources: [Ego4D](https://ego4d-data.org/#), [Epic-Kitchen](https://epic-kitchens.github.io/2025), [EgoExo4D](https://ego-exo4d-data.org/#intro), and [Something-Something V2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset). Due to license restrictions, we cannot provide our processed video data directly. To access the data, please apply for and download the original videos from the official dataset websites. Note that we only need the _raw video_ files for this project.

The structure of the downloaded raw data for each dataset is as follows:
- **Ego4D**:  
```
Ego4D_root/
└── v2/
    └── full_scale/
        ├── {video_name1}.mp4
        ├── {video_name2}.mp4
        ├── {video_name3}.mp4
        └── ...
```
- **Epic-Kitchen**:  
```
Epic-Kitchen_root/
├── P01/
│   └── videos/
│       ├── {video_name1}.MP4
│       ├── {video_name2}.MP4
│       └── ...
├── P02/
│   └── videos/
│       ├── {video_name3}.MP4
│       ├── {video_name4}.MP4
│       └── ...
└── ...
```
- **EgoExo4D**:  
```
EgoExo4D_root/
└── takes/
    ├── {video_name1}/
    │   └── frame_aligned_videos/
    │       ├── {cam_name1}.mp4
    │       ├── {cam_name2}.mp4
    │       └── ...
    ├── {video_name2}/
    │   └── frame_aligned_videos/
    │       ├── {cam_name1}.mp4
    │       ├── {cam_name2}.mp4
    │       └── ...
    └── ...
```
- **Somethingsomething-v2**:  
```
Somethingsomething-v2_root/
├── {video_name1}.webm
├── {video_name2}.webm
├── {video_name3}.webm
└── ...
```
---

## 3. Video Preprocessing

A large portion of the raw videos in Ego4D and EgoExo4D have fisheye distortion. To standardize the processing, we corrected the fisheye distortion and converted the videos to a pinhole camera model. Our metadata is based on the resulting undistorted videos. To enable reproduction of our data, we provide scripts to perform this undistortion on the original videos. 

### Camera Intrinsics

We provide our estimated intrinsics for raw videos in Ego4D (computed using [DroidCalib](https://github.com/boschresearch/DroidCalib) as described in our paper) and the ground-truth Project Aria intrinsics for EgoExo4D (from the [official repository](https://github.com/EGO4D/ego-exo4d-egopose/tree/main/handpose/data_preparation)). These files can be downloaded via [this link](https://huggingface.co/datasets/VITRA-VLA/VITRA-1M/tree/main/intrinsics) and organized as follows:
```
camera_intrinsics_root/
├── ego4d/
│   ├── {video_name1}.npy
│   ├── {video_name2}.npy
│   └── ...
└── egoexo4d/
    ├── {video_name3}.json
    ├── {video_name4}.json
    └── ...
```
### Video Undistortion
Given the raw videos organized according to the structure described in [Data Download](#2-data-download) and the provided camera intrinsics, the fisheye-distorted videos can be undistorted using the following script:
```bash
cd data/preprocessing

# for Ego4D videos
usage: undistort_video.py [-h] --video_root VIDEO_ROOT --intrinsics_root INTRINSICS_ROOT --save_root SAVE_ROOT [--video_start START_IDX] [--video_END END_IDX] [--batchsize BATCHSIZE] [--crf CRF]

options:
  -h, --help                            show this help message and exit
  --video_root VIDEO_ROOT               Folder containing input videos
  --intrinsics_root INTRINSICS_ROOT     Folder containing intrinsics info
  --save_root SAVE_ROOT                 Folder for saving output videos
  --video_start VIDEO_START             Start video index (inclusive)
  --video_end VIDEO_END                 End video index (exclusive)
  --batch_size BATCH_SIZE               Number of frames to be processed per batch (TS chunk)
  --crf CRF                             CRF for ffmpeg encoding quality
```

An example command is:
```bash
# for Ego4D videos
python undistort_video.py --video_root Ego4D_root/v2/full_scale --intrinsics_root camera_intrinsics_root/ego4d --save_root Ego4D_undistorted_root --video_start 0 --video_end 10
```
which processes 10 Ego4D videos sequentially and saves the undistorted outputs to ``Ego4D_root/v2/undistorted_videos``.

Similarly, for EgoExo4D videos, you can run a command like:
```bash
# for EgoEXO4D videos
python undistort_video_egoexo4d.py --video_root EgoExo4D_root --intrinsics_root camera_intrinsics_root/egoexo4d --save_root EgoExo4D_undistorted_root --video_start 0 --video_end 10
```

Each video is processed in segments according to the specified batch size and then concatenated afterward. Notably, processing the entire dataset is time-consuming and requires substantial storage (around 10 TB). The script provided here is only a basic reference example. **We recommend parallelizing and optimizing it before running it on a compute cluster.**

**The undistortion step is only applied to Ego4D and EgoExo4D videos. Epic-Kitchen and Somethingsomething-v2 do not require undistortion and can be used directly as downloaded from the official sources.**

---

## 4. Metadata Structure
Our metadata for each V-L-A episode can be loaded via:
```python
import numpy as np

# Load meta data dictionary
episode_info = np.load(f'{dataset_name1}_{video_name1}_ep_{000000}.npy', allow_pickle=True).item()

```
The detailed structure of the ``episode_info`` is as follows:
```
episode_info (dict)                                 # Metadata for a single V-L-A episode
├── 'video_clip_id_segment': list[int]              # Deprecated
├── 'extrinsics': np.ndarray                        # (Tx4x4) World2Cam extrinsic matrix
├── 'intrinsics': np.ndarray                        # (3x3) Camera intrinsic matrix
├── 'video_decode_frame': list[int]                 # Frame indices in the original raw video (starting from 0)
├── 'video_name': str                               # Original raw video name
├── 'avg_speed': float                              # Average wrist movement per frame (in meters)
├── 'total_rotvec_degree': float                    # Total camera rotation over the episode (in degrees)
├── 'total_transl_dist': float                      # Total camera translation distance over the episode (in meters)
├── 'anno_type': str                                # Annotation type, specifying the primary hand action considered when segmenting the episode
├── 'text': (dict)                                  # Textual descriptions for the episode
│     ├── 'left': List[(str, (int, int))]           # Each entry contains (description, (start_frame_in_episode, end_frame_in_episode))
│     └── 'right': List[(str, (int, int))]          # Same structure for the right hand
├── 'text_rephrase': (dict)                         # Rephrased textual descriptions from GPT-4
│     ├── 'left': List[(List[str], (int, int))]     # Each entry contains (list of rephrased descriptions, (start_frame_in_episode, end_frame_in_episode))
│     └── 'right': List[(List[str], (int, int))]    # Same as above for the right hand
├── 'left' (dict)                                   # Left hand 3D pose info
│   ├── 'beta': np.ndarray                          # (10) MANO hand shape parameters (based on the MANO_RIGHT model)
│   ├── 'global_orient_camspace': np.ndarray        # (Tx3x3) Hand wrist rotations from MANO's canonical space to camera space
│   ├── 'global_orient_worldspace': np.ndarray      # (Tx3x3) Hand wrist rotations from MANO's canonical space to world space
│   ├── 'hand_pose': np.ndarray                     # (Tx15x3x3) Local hand joints rotations (based on the MANO_RIGHT model)
│   ├── 'transl_camspace': np.ndarray               # (Tx3) Deprecated
│   ├── 'transl_worldspace': np.ndarray             # (Tx3) Hand wrist translation in world space
│   ├── 'kept_frames': list[int]                    # (T) 0–1 mask of valid left-hand reconstruction frames
│   ├── 'joints_camspace': np.ndarray               # (Tx21x3) 3D hand joint positions in camera space
│   ├── 'joints_worldspace': np.ndarray             # (Tx21x3) 3D joint positions in world space
│   ├── 'wrist': np.ndarray                         # Deprecated
│   ├── 'max_translation_movement': float           # Deprecated
│   ├── 'max_wrist_rotation_movement': float        # Deprecated
│   └── 'max_finger_joint_angle_movement': float    # Deprecated
└── 'right' (dict)                                  # Right hand 3D pose info (same structure as 'left')
    ├── 'beta': np.ndarray
    ├── 'global_orient_camspace': np.ndarray
    ├── 'global_orient_worldspace': np.ndarray
    ├── 'hand_pose': np.ndarray
    ├── 'transl_camspace': np.ndarray
    ├── 'transl_worldspace': np.ndarray
    ├── 'kept_frames': list[int]
    ├── 'joints_camspace': np.ndarray
    ├── 'joints_worldspace': np.ndarray
    ├── 'wrist': np.ndarray
    ├── 'max_translation_movement': float
    ├── 'max_wrist_rotation_movement': float
    └── 'max_finger_joint_angle_movement': float
```
To better understand how to use the episode metadata, we provide a visualization script, as described in the next section.

---

## 5. Data Visualization
Our metadata for each episode can be visualized with the following command, which will generate a video in the same format as shown on [our webpage](https://microsoft.github.io/VITRA/).   
We recommend following the undistortion procedure described above and place all undistorted videos in a single ``video_root`` folder, store the corresponding metadata in a ``label_root`` folder, and then run the visualization script.

```bash
usage: data/demo_visualization_epi.py [-h] --video_root VIDEO_ROOT --label_root LABEL_ROOT --save_path SAVE_PATH --mano_model_path MANO_MODEL_PATH [--render_gradual_traj]

options:
  -h, --help                            show this help message and exit
  --video_root VIDEO_ROOT               Root directory containing the video files
  --label_root LABEL_ROOT               Root directory containing the episode label (.npy) files
  --save_path SAVE_PATH                 Directory to save the output visualization videos
  --mano_model_path MANO_MODEL_PATH     Path to the MANO model files
  --render_gradual_traj                 Set flag to render a gradual trajectory (full mode)
```
We provide an example command for running the script, as well as a sample for visualization:
```bash
python data/demo_visualization_epi.py --video_root data/examples/videos --label_root data/examples/annotations --save_path data/examples/visualize --mano_model_path MANO_MODEL_PATH --render_gradual_traj
```
Note that using ``--render_gradual_traj`` renders the hand trajectory from the current frame to the end of the episode for every frame, which can be slow. To speed up visualization, you may omit this option.


For a more detailed understanding of the metadata, please see ``visualization/visualize_core.py``.
