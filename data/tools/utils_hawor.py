import sys
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
from ultralytics import YOLO

# Dynamically add HaWoR path for local imports
current_file_dir = os.path.dirname(os.path.abspath(__file__))
hawor_path = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'thirdparty', 'HaWoR'))
if hawor_path not in sys.path:
    sys.path.insert(0, hawor_path)

from thirdparty.HaWoR.lib.models.hawor import HAWOR
from thirdparty.HaWoR.lib.pipeline.tools import parse_chunks
from thirdparty.HaWoR.lib.eval_utils.custom_utils import interpolate_bboxes
from thirdparty.HaWoR.hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
from thirdparty.HaWoR.hawor.configs import get_config


def load_hawor(checkpoint_path: str):
    """
    Loads the HAWOR model and its configuration from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.

    Returns:
        tuple: (HAWOR model instance, model configuration object)
    """
    model_cfg_path = Path(checkpoint_path).parent.parent / 'model_config.yaml'
    model_cfg = get_config(str(model_cfg_path), update_cachedir=True)

    # Override config for correct bbox cropping when using ViT backbone
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, \
            f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    model = HAWOR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg

class HaworPipeline:
    """
    Pipeline for hand detection, tracking, and HAWOR motion estimation.
    """

    def __init__(
        self, 
        model_path: str = '', 
        detector_path: str = '', 
        device: torch.device = torch.device("cuda")
    ):
        """
        Initializes the HAWOR model and detector path.

        Args:
            model_path (str): Path to the HAWOR checkpoint.
            detector_path (str): Path to the hand detector (YOLO) weights.
            device (torch.device): Device to load models onto.
        """
        self.device = device
        self.detector_path = detector_path
        
        model, _ = load_hawor(model_path)
        model = model.to(device)
        model.eval()
        self.model = model

    def recon(
            self, 
            images: list, 
            img_focal: float, 
            thresh: float = 0.2, 
            single_image: bool = False
        ) -> dict:
        
        """
        Performs hand detection, tracking, and HAWOR-based 3D reconstruction.

        Args:
            images (list): List of consecutive input image frames (cv2/numpy format).
            img_focal (float): Focal length of the camera in pixels.
            thresh (float): Confidence threshold for hand detection.
            single_image (bool): Flag for single-image processing mode.

        Returns:
            dict: Dictionary of reconstruction results for 'left' and 'right' hands.
        """
        # Load detector and perform detection/tracking
        hand_det_model = YOLO(self.detector_path)
        _, tracks = detect_track(images, hand_det_model, thresh=thresh)
        
        # Perform HAWOR motion estimation
        recon_results = hawor_motion_estimation(
            images, tracks, self.model, img_focal, single_image=single_image
        )
        
        # delete the YOLO detector to avoid accumulation of tracking history
        del hand_det_model
        
        return recon_results

# Adapted from https://github.com/ThunderVVV/HaWoR/blob/main/scripts/scripts_test_video/detect_track_video.py
def detect_track(imgfiles: list, hand_det_model: YOLO, thresh: float = 0.5) -> tuple:
    """
    Detects and tracks hands across a sequence of images using YOLO.

    Args:
        imgfiles (list): List of image frames.
        hand_det_model (YOLO): The initialized YOLO hand detection model.
        thresh (float): Confidence threshold for detection.

    Returns:
        tuple: (list of boxes (unused in original logic), dict of tracks)
    """
    boxes_ = []
    tracks = {}

    for t, img_cv2 in enumerate(tqdm(imgfiles)):

        ### --- Detection ---
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                results = hand_det_model.track(img_cv2, conf=thresh, persist=True, verbose=False)
                
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                handedness = results[0].boxes.cls.cpu().numpy()
                if not results[0].boxes.id is None:
                    track_id = results[0].boxes.id.cpu().numpy()
                else:
                    track_id = [-1] * len(boxes)

                boxes = np.hstack([boxes, confs[:, None]])

                find_right = False
                find_left = False

                for idx, box in enumerate(boxes):
                    if track_id[idx] == -1:
                        if handedness[[idx]] > 0:
                            id = int(10000)
                        else:
                            id = int(5000)
                    else:
                        id = track_id[idx]
                    subj = dict()
                    subj['frame'] = t 
                    subj['det'] = True
                    subj['det_box'] = boxes[[idx]]
                    subj['det_handedness'] = handedness[[idx]]
                    
                    if (not find_right and handedness[[idx]] > 0) or (not find_left and handedness[[idx]]==0):
                        if id in tracks:
                            tracks[id].append(subj)
                        else:
                            tracks[id] = [subj]

                        if handedness[[idx]] > 0:
                            find_right = True
                        elif handedness[[idx]] == 0:
                            find_left = True

    return boxes_, tracks

# Adapted from https://github.com/ThunderVVV/HaWoR/blob/main/scripts/scripts_test_video/hawor_video.py
def hawor_motion_estimation(
    imgfiles: list, 
    tracks: dict, 
    model: HAWOR, 
    img_focal: float, 
    single_image: bool = False
) -> dict:
    """
    Performs HAWOR 3D hand reconstruction on detected and tracked hand regions.

    Args:
        imgfiles (list): List of image frames.
        tracks (dict): Dictionary mapping track ID to a list of detection objects.
        model (HAWOR): The initialized HAWOR model.
        img_focal (float): Camera focal length.
        single_image (bool): Flag for single-image processing mode.

    Returns:
        dict: Reconstructed parameters ('left' and 'right' hand results).
    """

    left_results = {}
    right_results = {}
    
    tid = np.array([tr for tr in tracks])

    left_trk = []
    right_trk = []
    for k, idx in enumerate(tid):
        trk = tracks[idx]

        valid = np.array([t['det'] for t in trk])        
        is_right = np.concatenate([t['det_handedness'] for t in trk])[valid]
        
        if is_right.sum() / len(is_right) < 0.5:
            left_trk.extend(trk)
        else:
            right_trk.extend(trk)
    left_trk = sorted(left_trk, key=lambda x: x['frame'])
    right_trk = sorted(right_trk, key=lambda x: x['frame'])
    final_tracks = {
        0: left_trk,
        1: right_trk
    }
    tid = [0, 1]

    img = imgfiles[0]
    img_center = [img.shape[1] / 2, img.shape[0] / 2]# w/2, h/2  
    H, W = img.shape[:2]

    for idx in tid:
        print(f"tracklet {idx}:")
        trk = final_tracks[idx]

        # interp bboxes
        valid = np.array([t['det'] for t in trk])
        if not single_image:
            if valid.sum() < 2:
                continue
        else:
            if valid.sum() < 1:
                continue
        boxes = np.concatenate([t['det_box'] for t in trk])
        non_zero_indices = np.where(np.any(boxes != 0, axis=1))[0]
        first_non_zero = non_zero_indices[0]
        last_non_zero = non_zero_indices[-1]
        boxes[first_non_zero:last_non_zero+1] = interpolate_bboxes(boxes[first_non_zero:last_non_zero+1])
        valid[first_non_zero:last_non_zero+1] = True


        boxes = boxes[first_non_zero:last_non_zero+1]
        is_right = np.concatenate([t['det_handedness'] for t in trk])[valid]
        frame = np.array([t['frame'] for t in trk])[valid]
        
        if is_right.sum() / len(is_right) < 0.5:
            is_right = np.zeros((len(boxes), 1))
        else:
            is_right = np.ones((len(boxes), 1))

        frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=1)

        if len(frame_chunks) == 0:
            continue

        for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
            print(f"inference from frame {frame_ck[0]} to {frame_ck[-1]}")
            img_ck = [imgfiles[i] for i in frame_ck]
            if is_right[0] > 0:
                do_flip = False
            else:
                do_flip = True
            
            results = model.inference(img_ck, boxes_ck, img_focal=img_focal, img_center=img_center, do_flip=do_flip)

            data_out = {
                "init_root_orient": results["pred_rotmat"][None, :, 0], # (B, T, 3, 3)
                "init_hand_pose": results["pred_rotmat"][None, :, 1:], # (B, T, 15, 3, 3)
                "init_trans": results["pred_trans"][None, :, 0],  # (B, T, 3)
                "init_betas": results["pred_shape"][None, :]  # (B, T, 10)
            }

            # flip left hand
            init_root = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
            init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
            if do_flip:
                init_root[..., 1] *= -1
                init_root[..., 2] *= -1
            data_out["init_root_orient"] = angle_axis_to_rotation_matrix(init_root)
            data_out["init_hand_pose"] = angle_axis_to_rotation_matrix(init_hand_pose)

            s_frame = frame_ck[0]
            e_frame = frame_ck[-1]

            for frame_id in range(s_frame, e_frame+1):
                result = {}
                result['beta'] = data_out['init_betas'][0, frame_id-s_frame].cpu().numpy()
                result['hand_pose'] = data_out['init_hand_pose'][0, frame_id-s_frame].cpu().numpy()
                result['global_orient'] = data_out['init_root_orient'][0, frame_id-s_frame].cpu().numpy()
                result['transl'] = data_out['init_trans'][0, frame_id-s_frame].cpu().numpy()
                
                if idx == 0:
                    left_results[frame_id] = result
                else:
                    right_results[frame_id] = result
    
    reformat_results = {'left': left_results, 'right': right_results}

    return reformat_results

