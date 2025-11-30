"""
interp_utils.py

Interpolation utilities for time-series data with masking support.
Provides functions for upsampling trajectories, rotation conversions, and MANO state interpolation.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d


def upsample_euler_with_mask(
    points: np.ndarray,
    mask: np.ndarray,
    upsample_factor: float = 2.0,
    method: str = "linear",
    t_new: np.ndarray = None,
):
    """
    Upsample point sequences with mask support, ensuring mask==0 regions don't affect mask==1 regions.
    
    Args:
        points: (N, D) array - N timesteps with D-dimensional coordinates/vectors
        mask: (N,) array - Binary mask (0 or 1)
        upsample_factor: Interpolation rate between points (can be float)
        method: Interpolation method - "linear", "quadratic", "cubic", or "pchip"
        t_new: (M,) array - Optional custom time axis; if None, auto-generated from upsample_factor
        
    Returns:
        new_points: (M, D) array - Upsampled point sequence
        new_mask: (M,) array - Upsampled mask
    """
    points = np.asarray(points)
    mask = np.asarray(mask).astype(bool)
    N, D = points.shape

    # Generate time axes
    t = np.arange(N)
    if t_new is None:
        M = int((N - 1) * upsample_factor + 1)
        t_new = np.linspace(0, N - 1, M)

    new_points = np.zeros((len(t_new), D), dtype=points.dtype)
    new_mask = np.zeros(len(t_new), dtype=mask.dtype)

    # Find continuous mask segments
    mask_diff = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(mask_diff == 1)[0]  # Mask transitions from 0 to 1
    ends = np.where(mask_diff == -1)[0]  # Mask transitions from 1 to 0

    # Interpolate each valid segment
    for s, e in zip(starts, ends):
        t_seg = t[s:e]
        pts_seg = points[s:e]
        idx = np.where((t_new >= s) & (t_new <= e))[0]

        if len(t_seg) == 0:
            continue
        elif len(t_seg) == 1:
            # Single point segment - direct fill
            new_points[idx] = pts_seg[0]
        else:
            t_seg_new = t_new[idx]
            
            # Perform interpolation
            if method == "pchip":
                interp_func = PchipInterpolator(t_seg, pts_seg, axis=0)
                new_points[idx] = interp_func(t_seg_new)
            else:
                interp_func = interp1d(
                    t_seg, pts_seg, kind=method, axis=0, bounds_error=False, fill_value="extrapolate"
                )
                new_points[idx] = interp_func(t_seg_new)

        new_mask[idx] = 1

    # Fill mask==0 regions (doesn't affect mask==1 segments)
    zero_idx = np.where(new_mask == 0)[0]
    if len(zero_idx) > 0:
        orig_zero_idx = np.where(mask == 0)[0]
        fill_value = points[0] if len(orig_zero_idx) == 0 else points[orig_zero_idx[0]]
        new_points[zero_idx] = fill_value
        new_mask[zero_idx] = 0

    return new_points, new_mask


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to 3x3 rotation matrices.
    
    Uses Gram-Schmidt orthogonalization to construct orthonormal basis.
    
    Args:
        d6: Array of shape (..., 6) - 6D rotation representation
        
    Returns:
        Array of shape (..., 3, 3) - Rotation matrices
    """
    d6 = np.asarray(d6)
    a1, a2 = d6[..., :3], d6[..., 3:]

    # Normalize first vector
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)

    # Orthogonalize second vector
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)

    # Cross product for third vector
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack([b1, b2, b3], axis=-2)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrices to 6D rotation representation.
    
    Extracts first two columns of rotation matrix as 6D representation.
    
    Args:
        matrix: Array of shape (..., 3, 3) - Rotation matrices
        
    Returns:
        Array of shape (..., 6) - 6D rotation representation
    """
    matrix = np.asarray(matrix)
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,))


def transform_mat_from_R_t(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Construct 4x4 transformation matrix from rotation and translation.
    
    Args:
        R: Array of shape (..., 3, 3) - Rotation matrices
        t: Array of shape (..., 3) or (..., 3, 1) - Translation vectors
        
    Returns:
        Array of shape (..., 4, 4) - Homogeneous transformation matrices
    """
    t = t[..., None] if t.ndim == R.ndim - 1 else t
    shape = R.shape[:-2] + (4, 4)
    T = np.zeros(shape, dtype=R.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3:4] = t
    T[..., 3, 3] = 1.0
    return T


def interp_mano_state(R, t, mano_R, mano_joints_manospace, mask, upsample_factor=1, method="pchip"):
    """
    Interpolate MANO hand state including global pose, translation, articulated pose, and joint positions.
    
    Preserves bone lengths during interpolation by:
    1. Interpolating bone directions and lengths separately
    2. Reconstructing joint positions from root using forward kinematics
    
    Args:
        R: (T, 3, 3) - Global rotation matrices
        t: (T, 3) - Global translation vectors
        mano_R: (T, 15, 3, 3) - MANO joint rotation matrices (15 articulated joints)
        mano_joints_manospace: (T, 21, 3) - Joint positions in MANO space
        mask: (T,) - Binary mask for valid frames
        upsample_factor: Interpolation rate (default: 2)
        method: Interpolation method (default: "pchip")
        
    Returns:
        R_interp: (M, 3, 3) - Interpolated global rotations
        t_interp: (M, 3) - Interpolated global translations
        mano_R_interp: (M, 15, 3, 3) - Interpolated MANO joint rotations
        mano_joints_manospace_interp: (M, 21, 3) - Interpolated joint positions
        mask_interp: (M,) - Interpolated mask
    """
    # MANO kinematic tree: parent joint indices for 21 joints
    parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]

    # === Interpolate Global Rotation ===
    rot6d = matrix_to_rotation_6d(R)  # (T,3,3) -> (T,6)
    rot6d_interp, mask_interp = upsample_euler_with_mask(rot6d, mask, upsample_factor, method)
    R_interp = rotation_6d_to_matrix(rot6d_interp)  # (M,3,3)

    # === Interpolate Global Translation ===
    t_interp, _ = upsample_euler_with_mask(t, mask, upsample_factor, method)  # (M,3)

    # === Interpolate MANO Joint Rotations ===
    mano_rot6d = matrix_to_rotation_6d(mano_R)  # (T,15,3,3) -> (T,15,6)
    mano_rot6d = mano_rot6d.reshape(len(mano_R), -1)  # (T,90)
    mano_rot6d_interp, _ = upsample_euler_with_mask(mano_rot6d, mask, upsample_factor, method)
    mano_rot6d_interp = mano_rot6d_interp.reshape(-1, 15, 6)  # (M,15,6)
    mano_R_interp = rotation_6d_to_matrix(mano_rot6d_interp)  # (M,15,3,3)

    # === Interpolate Joint Positions ===
    T, N, _ = mano_joints_manospace.shape
    parents_arr = np.array(parents)

    # Initialize bone vectors and lengths
    bone_vectors = np.zeros_like(mano_joints_manospace)  # [T, 21, 3]
    bone_lengths = np.zeros([T, N])  # [T, 21]

    # Find non-root joints
    child_idx = np.where(parents_arr != -1)[0]
    parent_idx = parents_arr[child_idx]

    # Compute bone vectors (vectorized)
    bone_vectors[:, child_idx, :] = (
        mano_joints_manospace[:, child_idx, :] - mano_joints_manospace[:, parent_idx, :]
    )

    # Compute bone lengths
    bone_lengths[:, child_idx] = np.linalg.norm(bone_vectors[:, child_idx, :], axis=-1)

    # Interpolate bone directions
    bone_vectors_interp, _ = upsample_euler_with_mask(bone_vectors.reshape(T, -1), mask, upsample_factor, method)
    bone_vectors_interp = bone_vectors_interp.reshape(-1, N, 3)  # (M,21,3)
    bone_vectors_interp = bone_vectors_interp / (np.linalg.norm(bone_vectors_interp, axis=-1, keepdims=True) + 1e-8)

    # Interpolate bone lengths
    bone_lengths_interp, _ = upsample_euler_with_mask(bone_lengths, mask, upsample_factor, method)

    # Reconstruct bone vectors with interpolated lengths
    bone_vectors_interp = bone_vectors_interp * bone_lengths_interp[..., None]

    # === Forward Kinematics: Reconstruct Joint Positions ===
    M = bone_vectors_interp.shape[0]
    mano_joints_manospace_interp = np.zeros((M, N, 3), dtype=bone_vectors_interp.dtype)

    # Root joint at origin in MANO space
    for i in range(1, N):
        p = parents[i]
        mano_joints_manospace_interp[:, i, :] = mano_joints_manospace_interp[:, p, :] + bone_vectors_interp[:, i, :]

    return R_interp, t_interp, mano_R_interp, mano_joints_manospace_interp, mask_interp