<div align="center">

<h1 align="center"><span
    style="font-family: 'Courier New', Courier, monospace; font-size: 115%;"><span style="font-size: 130%;">V</span>ITRA</span>:<br><span
    style="font-size:2.22rem;">Scalable Vision-Language-Action Model Pretraining <br> for Robotic Manipulation with Real-Life Human Activity Videos
    </span></h1>

<p align="center">
    <a href="https://arxiv.org/abs/2510.21571"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
    <a href='https://microsoft.github.io/VITRA/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
    <a href='https://huggingface.co/VITRA-VLA/VITRA-VLA-3B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
    <a href='https://huggingface.co/datasets/VITRA-VLA/VITRA-1M'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-yellow'></a>
    <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-orange' alt='License'></a>
</p>

<p align="center"><img src="assets/teaser.jpg" width="100%" alt="VITRA Teaser"></p>

<div align="justify">
<span style="font-family: 'Courier New', Courier, monospace; font-size: 115%;"><span style="font-size: 130%;">V</span>ITRA</span> is a novel approach for pretraining Vision-Language-Action (VLA) models for robotic manipulation using large-scale, unscripted, real-world videos of human hand activities. Treating human hand as dexterous robot end-effector, we show that in-the-wild egocentric human videos without any annotations can be transformed into data formats fully aligned with existing robotic V-L-A training data in terms of task granularity and labels. We create a human hand V-L-A dataset containing over 1 million episodes. We further develop a VLA model with a causal action transformer trained on this dataset. It demonstrates strong zero-shot human-hand action prediction in entirely new scenes and serves as a cornerstone for few-shot finetuning and adaptation to real-world robotic manipulation.
<br>
<br>

***For video demonstrations, please refer to our [project page](https://microsoft.github.io/VITRA/).***
</div>

<br>

</div>

---

## 🚩 New & Updates
*   **[2025-12-05]** 🚀 Release the code for performing zero-shot inference using a single image.
*   **[2025-11-30]** 🚀 Our code, pretrained models, and datasets are now open-sourced.
*   **[2025-10-24]** 🚀 **VITRA** paper is released on arXiv.

---
## 🤗 Pretrained Models and Datesets

Our pretrained model and datasets are available on the huggingface hub:
<table>
  <thead>
    <tr>
      <th>Hugging Face Model</th>
      <th>#Params</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/VITRA-VLA/VITRA-VLA-3B" target="_blank"><code>VITRA-VLA-3B</code><a></td>
      <td style="font-size: 0.92em;">3B</td>
      <td style="font-size: 0.92em;">Base VLA model pretrained on Human Hand Data.</td>
    </tr>
  </tbody>
</table>



**Note: Our base VLA model is finetuned from [Paligemma2](https://huggingface.co/google/paligemma2-3b-mix-224). if you do not have access to [Paligemma2](https://huggingface.co/google/paligemma2-3b-mix-224), please request permission on the [official website](https://huggingface.co/google/paligemma2-3b-mix-224).**



<table>
  <thead>
    <tr>
      <th rowspan="2">Hugging Face Dataset</th>
      <th colspan="2" style="text-align: center;">Sub Datasets</th>
    </tr>
    <tr>
      <th>Dataset Name</th>
      <th>Number of Episodes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6"><a href="https://huggingface.co/datasets/VITRA-VLA/VITRA-1M" target="_blank"><code>VITRA-1M</code></a></td>
      <td><code>ego4d_cooking_and_cleaning</code></td>
      <td>454,244</td>
    </tr>
    <tr>
      <td><code>ego4d_other</code></td>
      <td>494,439</td>
    </tr>
    <tr>
      <td><code>epic</code></td>
      <td>154,464</td>
    </tr>
    <tr>
      <td><code>egoexo4d</code></td>
      <td>67,053</td>
    </tr>
    <tr>
      <td><code>ssv2</code></td>
      <td>52,718</td>
    </tr>
    <tr>
      <td><strong>Total</strong></td>
      <td><strong>1,222,918</strong></td>
    </tr>
  </tbody>
</table>

**Note: See [`data/data.md`](data/data.md) for detailed information about our datasets.**

## 📑 Table of Contents
- [1. Installation](#1-installation)
  - [1.1 Training / Inference Requirements](#11-training--inference-requirements)
  - [1.2 Visualization Requirements](#12-visualization-requirements)
- [2. Inference with Human Hand Image](#2-inference-with-human-hand-image)
- [3. Fine-tuning with a Custom Robot Dataset](#3-fine-tuning-with-a-custom-robot-dataset)
  - [3.1 Data Preparation](#31-data-preparation)
  - [3.2 Implementing a Custom RoboDatasetCore](#32-implementing-a-custom-robodatasetcore)
  - [3.3 Computing Dataset Statistics](#33-computing-dataset-statistics)
  - [3.4 Modify Configuration](#34-modify-configuration)
  - [3.5 Run Scripts](#35-run-scripts)
- [4. Deployment in the Real World](#4-deployment-in-the-real-world)
- [5. Human Hand VLA Dataset Utilization](#5-human-hand-vla-dataset-utilization)
- [6. Human Data Pretraining from Scratch](#6-human-data-pretraining-from-scratch)
  - [6.1 Dataset Preparation](#61-dataset-preparation)
  - [6.2 Compute Dataset Statistics](#62-compute-dataset-statistics)
  - [6.3 Modify Configuration](#63-modify-configuration)
  - [6.4 Run Scripts](#64-run-scripts)
- [Citation](#citation)


---

## 1. Installation
### 1.1 Training / Inference Requirements
We recommend using `conda` to manage the environment. We require PyTorch >= 2.3.0 and CUDA >= 12.1 (It may run with lower versions, but we have not tested it). If the environment is used solely for training, it is recommended to use a higher version of PyTorch to achieve improved training speed.


```bash
# Clone the repository
git clone https://github.com/microsoft/VITRA.git
cd VITRA

# Create environment
conda create -n vitra python=3.10 -y
conda activate vitra

# Install dependencies
pip install -e .
```

<details>
<summary>Click to view detailed system requirements</summary>

*   **OS**: Linux (Ubuntu 20.04/22.04 recommended)
*   **Python**: 3.10+
*   **CUDA**: 11.8+
*   **GPU**: Minimum 16GB VRAM for inference, A100/H100 recommended for training.
</details>

### 1.2 Visualization Requirements
If you want to **visualize** the results after inference, run **dataset visualization**, or perform zero-shot human hand action prediction from a single image, please follow the instructions below.

**Submodules Installation**

Please clone the submodules to perform hand pose estimation.
```bash
git submodule update --init --recursive
```

**Libraries Installation**

Please install the following additional modules for visualization using the commands below:

```bash
pip install -e .[visulization] --no-build-isolation
```

<details>
<summary>Click here if you encounter issues when installing Installing <a href="https://github.com/facebookresearch/pytorch3d?tab=readme-ov-file">PyTorch3D</a> </summary>

*   If you encounter issues when installing [PyTorch3D](https://github.com/facebookresearch/pytorch3d?tab=readme-ov-file), please follow the installation instructions provided in the [PyTorch3D](https://github.com/facebookresearch/pytorch3d?tab=readme-ov-file) repository or try installing it separately using:

    ```bash
    pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@stable#egg=pytorch3d
    ```
</details>


If [FFmpeg](https://github.com/FFmpeg/FFmpeg) is not installed on your system, please install it first.
```bash
sudo apt install ffmpeg
```

**MANO Hand Model**

Our reconstructed hand labels are based on the MANO hand model. **We only require the right hand model.** The model parameters can be downloaded from the [official website](https://mano.is.tue.mpg.de/index.html) and organized in the following structure:
```
weights/
└── mano/
    ├── MANO_RIGHT.pkl
    └── mano_mean_params.npz
```
Please download the model weights of [HaWoR](https://github.com/ThunderVVV/HaWoR) for hand pose estimation:

```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./weights/hawor/external/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt -P ./weights/hawor/checkpoints/
```

---

## 2. Inference with Human Hand Image
You can use our pretrained model to perform zero-shot 3D human hand action prediction directly from an **egocentric human hand image (landscape view)** based on instructions. To predict human actions from pre-captured images, please run [`scripts/run_human_inference.sh`](scripts/run_human_inference.sh). Here is a simple example:
```bash
python scripts/inference_human_prediction.py \
    --config VITRA-VLA/VITRA-VLA-3B \
    --image_path ./examples/0002.jpg \
    --sample_times 4 \
    --save_state_local \
    --use_right \
    --video_path ./example_human_inf.mp4 \
    --mano_path ./weights/mano \
    --instruction "Left hand: None. Right hand: Pick up the picture of Michael Jackson." \
```
All example images are captured on mobile phones in **rooms that do not appear anywhere in the V-L-A dataset**. They also include entirely **unseen concepts**, such as photos of celebrities.

Users can capture images with their own devices and directly test the model using the recorded images.
> **Note:**  
> For best inference quality, it is recommended to capture **landscape view** images from a camera height close to that of a **human head** to match natural egocentric viewpoints. Highly unusual or distorted hand poses/positions may cause inference failures.


Below is the **minimum usage example** for predicting human actions. 


```python
import json
import torch
import numpy as np
from PIL import Image
from vitra.models import VITRA_Paligemma, load_model
from vitra.utils.data_utils import resize_short_side_to_target, load_normalizer
from vitra.datasets.human_dataset import pad_state_human, pad_action
from vitra.utils.config_utils import load_config
from vitra.datasets.dataset_utils import (
    ActionFeature,
    StateFeature,
)

# Load configs
configs = load_config('VITRA-VLA/VITRA-VLA-3B')

# Override config if provided
pretrained_path = 'VITRA-VLA/VITRA-VLA-3B'
statistics_path = 'VITRA-VLA/VITRA-VLA-3B'
configs['model_load_path'] = pretrained_path
configs['statistics_path'] = statistics_path

# Load model and normalizer
model = load_model(configs).cuda()
model.eval()

normalizer = load_normalizer(configs)

image_path = "your_image.jpg"
image = Image.open(image_path)
image = resize_short_side_to_target(image, target=224)
fov = torch.tensor([[np.deg2rad(60.0), np.deg2rad(60.0)]])      # input your camera FOV here [fov_x, fov_y]

image = np.array(image)

# Input your prompt here. Only predict the right hand action as an example.
instruction = "Left hand: None. Right hand: Pick up the phone on the table."  

# Initialize state
# State vector structure (total dimension: 122):
#   - state_left [51]:      Left hand state vector
#       * [0:3]    transl:          Translation in camera space (x, y, z in meters)
#       * [3:6]    global_orient:   Global rotation as Euler angles (xyz, in radians)
#       * [6:51]   hand_pose:       45 joint angles as Euler angles (15 joints × 3 axes, in radians)
#   - beta_left [10]:       Left hand MANO shape parameters
#   - state_right [51]:     Right hand state vector (same structure as state_left)
#       * [0:3]    transl:          Translation in camera space (x, y, z in meters)
#       * [3:6]    global_orient:   Global rotation as Euler angles (xyz, in radians)
#       * [6:51]   hand_pose:       45 joint angles as Euler angles (15 joints × 3 axes, in radians)
#   - beta_right [10]:      Right hand MANO shape parameters
state = np.zeros((normalizer.state_mean.shape[0],))             # Input your hand state here
# Only use right hand state as an example.
# state_mask[0] indicates whether to use left hand state, 
# state_mask[1] indicates whether to use right hand state.
state_mask = np.array([False, True], dtype=bool)                # Input your hand state mask here. 


# Input your action_mask here. Shape: (W, 2) where W is chunk_size. 
# action_mask[:, 0] indicates whether to predict left hand actions, 
# action_mask[:, 1] indicates whether to predict right hand actions. 
# All left hand False, all right hand True as an example.
action_mask = np.tile(np.array([[False, True]], dtype=bool), (model.chunk_size, 1))  


# Normalize state
norm_state = normalizer.normalize_state(state)

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
    action_mask=action_mask,
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
    sample_times = 1
)
norm_action = norm_action[0, :,:102]
# Denormalize predicted action
unnorm_action = normalizer.unnormalize_action(norm_action)
print("Predicted Action:", unnorm_action)
```

---

## 3. Fine-tuning with a Custom Robot Dataset


Our VITRA model serve as starting points for robot-specific fine-tuning (e.g., on Xhand or your custom robot).

### 3.1 Data Preparation

We use the **camera-space end-effector (EEF) pose** to represent the wrist pose of the dexterous hand. The teleoperation **command sent to the XHand** is treated as the dexterous-hand **joint action**, and the retrieved **XHand joint angles** are used as the hand joint state. Before fine-tuning with robot data, we strongly recommend aligning both hands’ EEF **translation** and **rotation** coordinate frames to the coordinate system used in the Hand V-L-A dataset.  

Align all coordinates to the **camera coordinate system**:

- **X-axis**: pointing to the right of the screen (positive direction)  
- **Y-axis**: pointing downward on the screen (positive direction)  
- **Z-axis**: pointing away from the camera, perpendicular into the screen (positive direction)  

For the **EEF rotation origin**, pay attention to the mirror relationship between the left and right hands. After aligning the rotation origin, the EEF pose should match the case in the following image. The left side is an example of Human V-L-A dataset; The right side is an aligned example of XHand dataset. 

![coordinate_alignment](assets/coordinate_alignment.png)

In addition, we provide the `transfer_xhand_to_human` function in [`vitra/dataset/robot_dataset.py`](./vitra/dataset/robot_dataset.py), which maps XHand joint angles to the closest degrees of freedom used in the human-hand representation. (Note that this XHand-to-human-hand alignment **does not need to be preprocessed offline**. It will be automatically applied during trajectory transformation inside the `transform_trajectory` function of the `RoboDatasetCore` class.)

If you are using a different dexterous hand model, we recommend implementing a similar function to align its joint configuration with the human-hand DoF.


### 3.2 Implementing a Custom `RoboDatasetCore`

To adapt the dataset loader to your own robot data, you should create a custom implementation of the `RoboDatasetCore` class in [`vitra/dataset/robot_dataset.py`](./vitra/dataset/robot_dataset.py).

The following methods typically need to be overridden to match the **human-data pretraining format**:

```python
def __init__(...):
    # dataset initialization
def __len__(self):
    # return the number of frames
    ...

def __getitem__(self, idx):
    """
    Returns a sample dictionary
    """
    ...
    return {
        "instruction": instruction,
        "image_list": image_list,
        "image_mask": image_mask,
        "action_list": action_list,
        "action_mask": action_mask,
        "current_state": current_state,
        "current_state_mask": current_state_mask,
        "fov": fov,
        "intrinsics": self.intrinsics,
    }
```

The dictionary returned by `__getitem__` contains multiple fields representing the robot state, actions, and observations. Here, we use the **XHand** configuration as an example for illustration:

- **Left Hand**: 6-DoF EEF pose + 12 joint angles = 18-DoF
- **Right Hand**: 6-DoF EEF pose + 12 joint angles = 18-DoF
- **The ordering of dimensions in the state or action**: [`left_eef_trans`, `left_eef_euler_rotation`, `left_hand_joint`, `right_eef_trans`, `right_eef_euler_rotation`, `right_hand_joint`]

</br>

| Key                    | Type                     | Shape             | Description                                                                                               |
|------------------------|--------------------------|-------------------|-----------------------------------------------------------------------------------------------------------|
| **instruction**        | `str`                    | —                 | Natural language description of the task. <br> e.g., `"Left hand: None. Right hand: {Right hand prompt}"` |
| **image_list**         | `np.ndarray (uint8)`     | `(1, H, W, C)`    | RGB image sequence. <br>• `C=3` (channels) <br>• `H, W` = height & width                                  |
| **image_mask**         | `np.ndarray (bool)`      | `(1,)`            | Indicates which frames are valid (`1=valid`, `0=padded`).                                                 |
| **action_list**        | `np.ndarray (float32)`   | `(T, 36)`         | Sequence of robot actions. (an action chunking) <br>• `T` = length of an action chunking                                                         |
| **action_mask**        | `np.ndarray (bool)`      | `(T, 2)`          | Indicates which time steps contain valid left-hand or right-hand actions. `action_mask[:, 0]` corresponds to the left hand, and `action_mask[:, 1]` corresponds to the right hand.                                                      |
| **current_state**      | `np.ndarray (float32)`   | `(36,)`           | Robot state at the current time step.                                                                     |
| **current_state_mask** | `np.ndarray (float32)`   | `(2,)`            | Indicates which hand's state is valid. `current_state_mask[0]` corresponds to the left hand, and `current_state_mask[1]` corresponds to the right hand.                                           |
| **fov**                | `np.ndarray (float32)`   | `(2,)`            | Camera field of view. `[fov_x, fov_y]`                                                                                    |
| **intrinsics**         | `np.ndarray (float32)`   | `(3, 3)`          | Camera intrinsic matrix. Not used during training or model inference.                                     |




### 3.3 Computing Dataset Statistics

Before training, we need to compute the **dataset statistics** for the robot dataset, specifically the mean and standard deviation of the **state** and **action** variables. These statistics will be used to **normalize the actions** using a Gaussian normalization.

### 3.4 Modify Configuration

After preparing the dataset and computing statistics, you can edit the `vitra/configs/robot_finetune.json` file to update paths and other relevant settings. `pretrain_path` should be replaced with the path to the pretrained model or the folder containing it, so that fine-tuning can be performed correctly.


In `scripts/run_robot_finetune.sh`, make sure to input your Hugging Face token and WANDB API key for proper authentication and logging.

### 3.5 Run Scripts

Once the configuration is ready, you can start pretraining by running:

```bash
bash scripts/run_robot_finetune.sh
```

<!-- TODO: Give an example code -->
---

## 4. Deployment in the Real World
Below is an example of running inference with **VITRA** on the XHand platform.

```python
import json
import torch
import numpy as np
from PIL import Image
from vitra.models import VITRA_Paligemma, load_model
from vitra.utils.data_utils import resize_short_side_to_target
from vitra.datasets.human_dataset import pad_state_human, pad_action
from vitra.utils.data_utils import load_normalizer
from vitra.datasets.dataset_utils import (
    ActionFeature,
    StateFeature,
)
from vitra.datasets.robot_dataset import (
    transfer_xhand_to_human,
    transfer_human_to_xhand,
    pad_state_robot, pad_action
)
# Load configs
configs = json.load(open('configs/robot_finetune.json'))
pretrained_path = 'checkpoints/finetuned_model.pt'
statistics_path = 'statistics/RoboData_statistics.json'
configs['model_load_path'] = pretrained_path
configs['statistics_path'] = statistics_path

# Load model and normalizer
model = load_model(configs).cuda()
model.eval()

normalizer = load_normalizer(configs)

image_path = "your_image.jpg"
image = Image.open(image_path)
image = resize_short_side_to_target(image, target=224)
fov = torch.tensor([[np.deg2rad(60.0), np.deg2rad(60.0)]])      # input your camera FOV here [fov_x, fov_y]

image = np.array(image)

# Input your prompt here. Only predict the right hand action as an example.
instruction = "Left hand: None. Right hand: Pour the contents of the bottle into the pot."  

# Initialize state
# State vector structure (total dimension: 36):
#   - Left hand [0:18]:
#       * [0:3]    transl:          Wrist translation in camera space (x, y, z in meters)
#       * [3:6]    global_orient:   Wrist rotation as Euler angles (xyz, in radians)
#       * [6:18]   hand_pose:       XHand joint angles (12 joints, in radians)
#   - Right hand [18:36]:
#       * [18:21]  transl:          Wrist translation in camera space (x, y, z in meters)
#       * [21:24]  global_orient:   Wrist rotation as Euler angles (xyz, in radians)
#       * [24:36]  hand_pose:       XHand joint angles (12 joints, in radians)
state = np.zeros((normalizer.state_mean.shape[0],))             # Input your hand state here
# Only use right hand state as an example.
# state_mask[0] indicates whether to use left hand state, 
# state_mask[1] indicates whether to use right hand state.
state_mask = np.array([False, True], dtype=bool)                # Input your hand state mask here. 


# Input your action_mask here. Shape: (W, 2) where W is chunk_size. 
# action_mask[:, 0] indicates whether to predict left hand actions, 
# action_mask[:, 1] indicates whether to predict right hand actions. 
# All left hand False, all right hand True as an example.
action_mask = np.tile(np.array([[False, True]], dtype=bool), (model.chunk_size, 1))  


# Normalize state
norm_state = normalizer.normalize_state(state)

unified_action_dim = ActionFeature.ALL_FEATURES[1]   # 192
unified_state_dim = StateFeature.ALL_FEATURES[1]     # 212

unified_state, unified_state_mask = pad_state_robot(
    state = norm_state,
    state_mask = state_mask,
    state_dim = normalizer.state_mean.shape[0],
    unified_state_dim = unified_state_dim,
)
_, unified_action_mask = pad_action(
    actions=None,
    action_mask=action_mask,
    action_dim=normalizer.action_mean.shape[0],
    unified_action_dim=unified_action_dim
)
human_state, human_state_mask, _, human_action_mask = transfer_xhand_to_human(
    unified_state, unified_state_mask,
    None, unified_action_mask
)
# Model inference
norm_action = model.predict_action(
    image = image,
    instruction = instruction,
    current_state = human_state.unsqueeze(0),
    current_state_mask = human_state_mask.unsqueeze(0),
    action_mask_torch = human_action_mask.unsqueeze(0),
    num_ddim_steps = 10,
    cfg_scale = 5.0,
    fov = fov,
    sample_times = 1
)
norm_action = norm_action[0, :,:102]
norm_robot_action = transfer_human_to_xhand(norm_action)
# Denormalize predicted action
unnorm_action = normalizer.unnormalize_action(norm_robot_action)
print("Predicted Action:", unnorm_action.shape)
# results in 2x 18-DoF actions of 16 steps with shape [16, 36]

```

---

## 5. Human Hand VLA Dataset Utilization


We release the **Human Hand V-L-A** dataset, which transforms "in-the-wild" videos into robot-aligned `(image, instruction, action)` tuples.  

You can download the dataset annotations from [VITRA-1M](https://huggingface.co/datasets/VITRA-VLA/VITRA-1M).
After downloading, please extract all `.gz` files using the following commands:
```bash
tar -xzvf ego4d_cooking_and_cleaning.tar.gz
tar -xzvf ego4d_other.tar.gz
tar -xzvf egoexo4d.tar.gz
tar -xzvf ssv2.tar.gz
tar -xzvf epic.tar.gz
```
For detailed information on the dataset, its structure, and usage instructions, please refer to the [`data/data.md`](data/data.md) file.


----


## 6. Human Data Pretraining from Scratch

To reproduce our pretraining results on the **Human-VLA** dataset:

### 6.1 Dataset Preparation

First, please follow the instructions in [`data/data.md`](data/data.md) to download the video dataset and the corresponding annotation files. After downloading, perform **[undistortion](data/data.md)** on the videos to correct for fisheye and lens distortions before further preprocessing.

#### Note on EgoExo4D Video Preprocessing

The undistorted EgoExo4D videos contain black borders due to fisheye correction. We apply an additional preprocessing step to remove these borders: first resize all EgoExo4D frames size to 448×448, then center-cropped to 256×256. The change of the intrinsics is already handled in `vitra/datasets/human_dataset.py`, where the computation of camera intrinsics incorporates the crop transformation and adjusts the camera FOV accordingly.


Once prepared, organize the data in the following directory structure:
```
Data_root/
├── Video/
│   ├── Ego4D_root/
│   │   ├── {video_name1}.mp4
│   │   ├── {video_name2}.mp4
│   │   └── ...
│   ├── Epic-Kitchen_root/
│   │   ├── {video_name1}.MP4
│   │   ├── {video_name2}.MP4
│   │   └── ...
│   ├── EgoExo4D_root/
│   │   ├── {video_name1}.mp4
│   │   ├── {video_name2}.mp4
│   │   └── ...
│   └── Somethingsomething-v2_root/
│       ├── {video_name1}.webm
│       ├── {video_name2}.webm
│       └── ...
└── Annotation/
    ├── ego4d_cooking_and_cleaning/
    ├── ego4d_other/
    ├── egoexo4d/
    ├── epic/
    ├── ssv2/
    └── statistics/
```

#### Notes:

- `Video/` contains all raw video files from different sources such as Ego4D, Epic-Kitchen, EgoExo4D, and Something-Something v2.  
- `Annotation/` contains the annotation files aligned with the corresponding videos.  

#### (Optional) Speed-Up Tips for Training

To further accelerate training, we recommend the following optional preprocessing steps:

1. Resize all processed videos so that the shorter side is 224 for faster decoding and reduced memory usage (preserve the aspect ratio during resizing).
2. Split long videos into shorter clips of at most 2000 frames each.  
   Name the clips using the format: `{video_name}_part{part_index}.mp4`, where `part_index` starts from **1**.
3. In `configs/human_pretrain.json`, set:
```json
"clip_len": 2000
```
This significantly improves data-loading speed during training, especially for large-scale datasets.

### 6.2 Compute Dataset Statistics

You can directly use the precomputed statistics provided in the downloaded dataset.
Alternatively, You can compute the dataset statistics (mean and standard deviation of states and actions) by running the following command.  

```bash
python vitra/datasets/calculate_statistics.py --save_folder data_root/statistics
```

### 6.3 Modify Configuration

Edit the `vitra/configs/human_pretrain.json` file to update paths and other relevant settings.

In `scripts/run_human_pretrain.sh`, make sure to input your Hugging Face token and WANDB API key for proper authentication and logging.

### 6.4 Run Scripts

Once the configuration is ready, you can start pretraining by running:
```bash
# Distributed training
bash scripts/run_human_pretrain.sh
```


---

## Citation

If you find our work useful in your research, please cite:

```bibtex
@article{li2025vitra,
  title={Scalable Vision-Language-Action Model Pretraining for Robotic Manipulation with Real-Life Human Activity Videos},
  author={Qixiu Li and Yu Deng and Yaobo Liang and Lin Luo and Lei Zhou and Chengtang Yao and Lingqi Zeng and Zhiyuan Feng and Huizhi Liang and Sicheng Xu and Yizhong Zhang and Xi Chen and Hao Chen and Lily Sun and Dong Chen and Jiaolong Yang and Baining Guo},
  journal={arXiv preprint arXiv:2510.21571},
  year={2025}
}
```
