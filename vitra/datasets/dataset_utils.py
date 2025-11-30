import numpy as np

class ActionFeature(object):
    """Action feature indices for human and robot hand parameters.
    
    Defines the start and end indices for different hand feature components
    in the concatenated action feature vector.
    """

    ALL_FEATURES = (0, 192)

    HUMAN_LEFT_HAND = (0, 51)
    HUMAN_RIGHT_HAND = (51, 102)
    HUMAN_LEFT_TRANS = (0, 3)
    HUMAN_LEFT_ROT = (3, 6)
    HUMAN_LEFT_6D = (0, 6)
    HUMAN_LEFT_JOINTS = (6, 51)
    HUMAN_RIGHT_TRANS = (51, 54)
    HUMAN_RIGHT_ROT = (54, 57)
    HUMAN_RIGHT_6D = (51, 57)
    HUMAN_RIGHT_JOINTS = (57, 102)
    PADDING_FEATURES = (102, 192)    # not used now
    @classmethod
    def get_concatenated_action_feature_from_dict(cls, action_feature_dict):
        """Concatenate action features from a dictionary into a single feature vector.
        
        Args:
            action_feature_dict: Dictionary mapping feature names to their values
            
        Returns:
            Tuple of (features, feature_mask) where features is the concatenated array
            and feature_mask indicates which features are present
        """
        batch_size = next(iter(action_feature_dict.values())).shape[0]
        features = np.zeros((batch_size, cls.ALL_FEATURES[1]), dtype=np.float32)
        feature_mask = np.zeros((batch_size, cls.ALL_FEATURES[1]), dtype=bool)
        
        for key, value in action_feature_dict.items():
            assert len(value.shape) == 2
            start, end = getattr(cls, key)
            k = value.shape[1]
            features[:, start:start + k] = value
            feature_mask[:, start:start + k] = True
        return features, feature_mask
    
    @classmethod
    def get_dict_from_concatenated_action_feature(cls, feature, feature_mask):
        """Extract action features from concatenated vector into a dictionary.
        
        Args:
            feature: Concatenated feature array
            feature_mask: Boolean mask indicating which features are present
            
        Returns:
            Dictionary mapping feature names to their extracted values
        """
        action_feature_dict = {}
        consts = {
            name: getattr(cls, name)
            for name in dir(cls)
            if name.isupper() and "ALL" not in name
        }
        for key, (start, end) in consts.items():
            k = np.sum(feature_mask[0, start:end])
            if k == 0:
                continue
            action_feature_dict[key] = feature[:, start:start + k]
        return action_feature_dict
    
    @classmethod
    def get_loss_components(cls, action_type='angle'):
        """Get loss component definitions for different action types.
        
        Uses existing feature index constants to avoid hardcoding numbers.
        
        Args:
            action_type: 'angle' or 'keypoints'
            
        Returns:
            dict: Dictionary mapping component names to (start, end, weight) tuples
        """
        if action_type == 'angle':
            # Directly use class constants - no hardcoded numbers!
            return {
                'left_hand_6d': (*cls.HUMAN_LEFT_6D, 1.0),
                'left_hand_joints': (*cls.HUMAN_LEFT_JOINTS, 1.0),
                'right_hand_6d': (*cls.HUMAN_RIGHT_6D, 1.0),
                'right_hand_joints': (*cls.HUMAN_RIGHT_JOINTS, 1.0),
            }
        elif action_type == 'keypoints':
            # For keypoints type, joints have different dimensions (21*3=63)
            left_joints_start = cls.HUMAN_LEFT_6D[1]  # After 6D
            left_joints_end = left_joints_start + 63    # 21 joints * 3D
            right_joints_start = cls.HUMAN_RIGHT_6D[1]
            right_joints_end = right_joints_start + 63
            
            return {
                'left_6d': (*cls.HUMAN_LEFT_6D, 1.0),
                'left_joints': (left_joints_start, left_joints_end, 1.0),
                'right_6d': (*cls.HUMAN_RIGHT_6D, 1.0),
                'right_joints': (right_joints_start, right_joints_end, 1.0),
            }
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    @classmethod
    def get_hand_group_mapping(cls, action_type='angle'):
        """Get mapping from loss components to hand groups for weighted averaging.
        
        Returns:
            dict: Dictionary mapping hand group names to list of component names
        """
        return {
            'left_hand': ['left_trans', 'left_rot', 'left_joints'],
            'right_hand': ['right_trans', 'right_rot', 'right_joints'],
        }
    
    @classmethod
    def get_xhand_loss_components(cls):
        """Get loss components specific to XHand dataset."""
        return {
            'left_hand_6d': (*cls.HUMAN_LEFT_6D, 1.6),
            'left_hand_joints': (*cls.HUMAN_LEFT_JOINTS, 0.4),
            'right_hand_6d': (*cls.HUMAN_RIGHT_6D, 1.6),
            'right_hand_joints': (*cls.HUMAN_RIGHT_JOINTS, 0.4),
        }
    

class StateFeature(ActionFeature):
    """Extended feature indices including state features like hand shape parameters (beta).
    
    Inherits from ActionFeature and adds additional state-specific features.
    """
    ALL_FEATURES = (0, 212)
    HUMAN_LEFT_BETA = (192, 202)  # MANO shape parameters for left hand, not used now
    HUMAN_RIGHT_BETA = (202, 212)  # MANO shape parameters for right hand, not used now

def calculate_fov(h, w, intrinsics):
    """Calculate horizontal and vertical field of view (FOV) from camera intrinsics.
    Args:
        h: Image height
        w: Image width
        intrinsics: 3x3 camera intrinsic matrix
    Returns:
        fov: np.array of shape (2,) containing horizontal and vertical FOV in radians
    """

    hfov = 2 * np.arctan(w / (2 * intrinsics[0][0])) # fx is the horizontal focal length
    vfov = 2 * np.arctan(h / (2 * intrinsics[1][1])) # fy is the vertical focal length
    fov = np.array([hfov, vfov], dtype=np.float32)

    return fov

def compute_new_intrinsics_crop(original_intrinsics, original_size, crop_size, resize_size):
    """Compute new camera intrinsics after square crop and resize operations.
    
    Args:
        original_intrinsics: Original 3x3 camera intrinsic matrix
        original_size: Original image size (single dimension for square)
        crop_size: Size of the square crop
        resize_size: Target size after resizing
        
    Returns:
        Updated 3x3 intrinsic matrix accounting for crop and resize
    """
    original_fx = original_intrinsics[0][0]
    original_fy = original_intrinsics[1][1]
    original_cx = original_intrinsics[0][2]
    original_cy = original_intrinsics[1][2]
    
    # Compute the crop offset (top-left corner of the crop)
    crop_offset = (original_size - crop_size) / 2
    
    # Update the principal point after the crop
    cropped_cx = original_cx - crop_offset
    cropped_cy = original_cy - crop_offset
    
    # Compute the scaling factor for resizing
    scale = resize_size / crop_size
    
    # Update the focal lengths and principal point after resizing
    new_fx = original_fx * scale
    new_fy = original_fy * scale
    new_cx = cropped_cx * scale
    new_cy = cropped_cy * scale
    
    intrinsics_matrix = np.array([
        [new_fx, 0, new_cx],
        [0, new_fy, new_cy],
        [0, 0, 1]
    ])
    return intrinsics_matrix

def compute_new_intrinsics_resize(original_intrinsics, resize_size):
    """Compute new camera intrinsics after resize operation.
    
    Args:
        original_intrinsics: Original 3x3 camera intrinsic matrix
        resize_size: Target size as (H, W) tuple
        
    Returns:
        Updated 3x3 intrinsic matrix accounting for the resize
    """
    original_fx = original_intrinsics[0][0]
    original_fy = original_intrinsics[1][1]
    original_cx = original_intrinsics[0][2]
    original_cy = original_intrinsics[1][2]

    H, W = resize_size

    # Compute the scaling factors for resizing
    scale_x = W / (2*original_cx)
    scale_y = H / (2*original_cy)

    # Update the focal lengths and principal point after resizing
    new_fx = original_fx * scale_x
    new_fy = original_fy * scale_y
    new_cx = original_cx * scale_x
    new_cy = original_cy * scale_y

    intrinsics_matrix = np.array([
        [new_fx, 0, new_cx],
        [0, new_fy, new_cy],
        [0, 0, 1]
    ])

    return intrinsics_matrix
