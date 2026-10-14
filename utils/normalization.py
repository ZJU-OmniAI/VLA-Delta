import math
import torch
import numpy as np


NORMALIZED_VALUE_CLIP = (-15.0, 15.0)

def rotation_norm(end_effector_state):
    '''
    Converts Euler angles [r, p, y] in the end-effector state to their sine and cosine representations.

    Args:
        end_effector_state (list): A list like [x, y, z, r, p, y, ...], where r, p, y are roll, pitch, yaw angles.

    Returns:
        list: [x, y, z, sin(r), cos(r), sin(p), cos(p), sin(y), cos(y), ...]
    '''
    r, p, y = end_effector_state[3:6]
    values = [math.sin(r), math.cos(r), math.sin(p), math.cos(p), math.sin(y), math.cos(y)]
    return end_effector_state[:3] + values + end_effector_state[6:]

def rotation_denorm(norm_end_effector_state):
    '''
    Recovers Euler angles [r, p, y] from their sine and cosine representations in the normalized end-effector state.

    Args:
        norm_end_effector_state (list): A list like [x, y, z, sin(r), cos(r), sin(p), cos(p), sin(y), cos(y), ...]

    Returns:
        list: [x, y, z, r, p, y, ...], where r, p, y are recovered roll, pitch, yaw angles.
    '''
    # Convert to list
    if isinstance(norm_end_effector_state, torch.Tensor):
        norm_end_effector_state = norm_end_effector_state.detach().cpu().tolist()
    elif isinstance(norm_end_effector_state, np.ndarray):
        norm_end_effector_state = norm_end_effector_state.tolist()
    # If list, nothing to do

    x, y, z = norm_end_effector_state[:3]
    sin_r, cos_r = norm_end_effector_state[3], norm_end_effector_state[4]
    sin_p, cos_p = norm_end_effector_state[5], norm_end_effector_state[6]
    sin_y, cos_y = norm_end_effector_state[7], norm_end_effector_state[8]

    r = math.atan2(sin_r, cos_r)
    p = math.atan2(sin_p, cos_p)
    y_ = math.atan2(sin_y, cos_y)

    end_effector_state = [x, y, z, r, p, y_] + norm_end_effector_state[9:]
    return end_effector_state

def min_max_norm_unclipped(values, key_stats, use_quantile, eps=1e-8):
    """Production quantile/min-max transform before the legacy safety clip."""
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)

    if values.ndim == 1:
        values = values.unsqueeze(0)

    if use_quantile:
        min_values = torch.from_numpy(key_stats["q01"]).to(dtype=values.dtype, device=values.device)
        max_values = torch.from_numpy(key_stats["q99"]).to(dtype=values.dtype, device=values.device)
    else:
        min_values = torch.from_numpy(key_stats["min"]).to(dtype=values.dtype, device=values.device)
        max_values = torch.from_numpy(key_stats["max"]).to(dtype=values.dtype, device=values.device)

    normed_0_1 = (values - min_values) / (max_values - min_values + eps)
    return normed_0_1 * 2 - 1


def min_max_norm(values, key_stats, use_quantile, eps=1e-8):
    """
    Per-dimension min-max normalization to [-1, 1].

    Args:
        values (array-like): shape (N, L) or (L,), can be list, numpy array, or torch tensor
        key_stats (dict): If use_quantile=True, use key_stats["q01"] and key_stats["q99"];
                          If use_quantile=False, use key_stats["min"] and key_stats["max"];
        use_quantile (bool): Whether to use q01/q99 (True) or min/max (False).
        eps (float): Epsilon for division safety.
    Returns:
        torch.Tensor: normalized tensor shape (N, L) or [1, L], values in [-1, 1]
    """
    normed = min_max_norm_unclipped(values, key_stats, use_quantile, eps=eps)
    return torch.clamp(normed, *NORMALIZED_VALUE_CLIP)

def min_max_denorm(values, key_stats, use_quantile=True, eps=1e-8):
    """
    Inverse of min-max normalization to [-1, 1].
    Args:
        values (array-like): shape (N, L) or (L,), can be list, numpy array, or torch tensor. Normalized values in [-1, 1]
        key_stats (dict): 
            If use_quantile=True: key_stats["q01"], key_stats["q99"], shape (L,)
            If use_quantile=False: key_stats["min"], key_stats["max"], shape (L,)
        use_quantile (bool): Whether to use q01/q99 or min/max.
        eps (float): Epsilon added during normalization.
    Returns:
        torch.Tensor: shape (N, L), denormalized values
    """
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    
    if values.ndim == 1:
        values = values.unsqueeze(0)
    
    if use_quantile:
        min_values = torch.from_numpy(key_stats["q01"]).to(dtype=values.dtype, device=values.device)
        max_values = torch.from_numpy(key_stats["q99"]).to(dtype=values.dtype, device=values.device)
    else:
        min_values = torch.from_numpy(key_stats["min"]).to(dtype=values.dtype, device=values.device)
        max_values = torch.from_numpy(key_stats["max"]).to(dtype=values.dtype, device=values.device)
    
    scale = max_values - min_values + eps
    # Inverse transform: orig = ((normed + 1)/2) * scale + min
    orig_values = ((values + 1) / 2) * scale + min_values
    return orig_values

# def xarm_tabletop_post_action_denorm(action_chunk, dataset_meta):
#     # denormalize each dim from [-1, 1] to [min, max]
#     action_chunk = min_max_denorm(action_chunk, dataset_meta.stats["action"])
#     action_chunk = action_chunk.tolist()
#     # denormalize the rotation dims: [x, y, z, sin(r), cos(r), ...] -> [x, y, z, r, ...]
#     action_chunk = [rotation_denorm(action) for action in action_chunk]
#     # the gripper's state is 0 or 1
#     for action in action_chunk:
#         action[-1] = 1 if action[-1] > 0.5 else 0

#     return action_chunk

# def post_action_denorm(action_chunk, dataset_meta):
#     # denormalize each dim from [-1, 1] to [min, max]
#     action_chunk = min_max_denorm(action_chunk, dataset_meta.stats["action"])
#     action_chunk = action_chunk.tolist()

#     return action_chunk
