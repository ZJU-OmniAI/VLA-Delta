"""Canonical 7D EEF contract for mixed pretraining.

The contract is anchored to MolmoAct: absolute world/base-frame EEF state
``[x, y, z, roll, pitch, yaw, gripper_open]`` and native-step relative action
``[dx, dy, dz, droll, dpitch, dyaw, gripper_open]``.  Translation is metres,
rotation is radians, RPY uses SciPy's lowercase ``xyz`` convention, and the
gripper channel is one for open and zero for closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


CANONICAL_DIM = 7
CANONICAL_SCHEMA_VERSION = 1
MAX_ABSOLUTE_POSITION_M = 2.0
MAX_TRANSLATION_STEP_M = 0.25
MAX_ROTATION_STEP_RAD = math.pi / 4.0
CANONICAL_SCHEMA = {
    "version": CANONICAL_SCHEMA_VERSION,
    "state": ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad", "gripper_open"],
    "action": ["dx_m", "dy_m", "dz_m", "droll_rad", "dpitch_rad", "dyaw_rad", "gripper_open"],
    "translation_frame": "robot base/world frame used by each source converter",
    "state_pose": "absolute EEF pose",
    "action_pose": "native-next-step relative EEF pose",
    "rotation": "SciPy lowercase xyz RPY; component deltas wrapped to [-pi,pi)",
    "gripper": "continuous openness for state; binary openness for action; 1=open, 0=closed",
    "time_alignment": "action row t is the command/achieved delta aligned to observation row t",
}


def _vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    return result


def wrap_angles(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_wxyz_to_rpy(value) -> np.ndarray:
    wxyz = _vector(value, 4, "quaternion_wxyz")
    norm = np.linalg.norm(wxyz)
    if not np.isfinite(norm) or not 0.9 <= norm <= 1.1:
        raise ValueError("quaternion_wxyz must be finite and near unit length")
    xyzw = wxyz[[1, 2, 3, 0]] / norm
    return Rotation.from_quat(xyzw).as_euler("xyz")


def rotvec_to_wrapped_rpy_delta(current_rpy, delta_rotvec) -> np.ndarray:
    current = _vector(current_rpy, 3, "current_rpy")
    delta = _vector(delta_rotvec, 3, "delta_rotvec")
    current_rotation = Rotation.from_euler("xyz", current)
    target_rotation = Rotation.from_rotvec(delta) * current_rotation
    return wrap_angles(target_rotation.as_euler("xyz") - current)


def _continuous_open(value: float, *, open_when_high: bool) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value if open_when_high else 1.0 - value


def _binary_open(value: float, *, open_when_low: bool, threshold: float) -> float:
    return float(value < threshold if open_when_low else value >= threshold)


def _state_valid(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values).all(axis=1) & np.all(
        np.abs(values[:, :3]) <= MAX_ABSOLUTE_POSITION_M, axis=1
    )


def _action_valid(values: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(values).all(axis=1)
        & (np.linalg.norm(values[:, :3], axis=1) <= MAX_TRANSLATION_STEP_M)
        & (np.linalg.norm(values[:, 3:6], axis=1) <= MAX_ROTATION_STEP_RAD)
    )


def canonical_molmo_arrays(states, actions):
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 7 or actions.shape != states.shape:
        raise ValueError("Molmo state/actions must both have shape [N,7]")
    canonical_states = states.copy()
    canonical_states[:, 3:6] = wrap_angles(canonical_states[:, 3:6])
    canonical_states[:, 6] = 1.0 - np.clip(canonical_states[:, 6], 0.0, 1.0)
    canonical_actions = actions.copy()
    canonical_actions[:, 3:6] = wrap_angles(canonical_actions[:, 3:6])
    canonical_actions[:, 6] = (canonical_actions[:, 6] < 0.5).astype(np.float64)
    return (
        canonical_states.astype(np.float32),
        canonical_actions.astype(np.float32),
        _state_valid(canonical_states),
        _action_valid(canonical_actions),
    )


def canonical_droid_arrays(state_poses, state_grippers, target_poses, target_grippers):
    state_poses = np.asarray(state_poses, dtype=np.float64)
    target_poses = np.asarray(target_poses, dtype=np.float64)
    state_grippers = np.asarray(state_grippers, dtype=np.float64).reshape(-1)
    target_grippers = np.asarray(target_grippers, dtype=np.float64).reshape(-1)
    if state_poses.ndim != 2 or state_poses.shape[1] != 6 or target_poses.shape != state_poses.shape:
        raise ValueError("DROID state/target poses must both have shape [N,6]")
    if len(state_grippers) != len(state_poses) or len(target_grippers) != len(state_poses):
        raise ValueError("DROID gripper arrays must align with poses")
    states = np.empty((len(state_poses), 7), dtype=np.float64)
    states[:, :3] = state_poses[:, :3]
    states[:, 3:6] = wrap_angles(state_poses[:, 3:6])
    states[:, 6] = 1.0 - np.clip(state_grippers, 0.0, 1.0)
    actions = np.empty((len(state_poses), 7), dtype=np.float64)
    actions[:, :3] = target_poses[:, :3] - state_poses[:, :3]
    actions[:, 3:6] = wrap_angles(target_poses[:, 3:6] - state_poses[:, 3:6])
    actions[:, 6] = (target_grippers < 0.2).astype(np.float64)
    return (
        states.astype(np.float32), actions.astype(np.float32),
        _state_valid(states), _action_valid(actions),
    )


def canonical_rh20t_arrays(states, actions, action_valid):
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    action_valid = np.asarray(action_valid, dtype=bool).reshape(-1)
    if states.ndim != 2 or states.shape[1] != 8:
        raise ValueError("RH20T state must have shape [N,8]")
    if actions.shape != (len(states), 7) or len(action_valid) != len(states):
        raise ValueError("RH20T action/action.valid must align and have seven dimensions")
    wxyz = states[:, 3:7]
    norms = np.linalg.norm(wxyz, axis=1)
    quaternion_valid = np.isfinite(wxyz).all(axis=1) & (norms >= 0.9) & (norms <= 1.1)
    canonical_states = np.full((len(states), 7), np.nan, dtype=np.float64)
    canonical_actions = actions.copy()
    if quaternion_valid.any():
        selected = np.flatnonzero(quaternion_valid)
        xyzw = wxyz[selected][:, [1, 2, 3, 0]] / norms[selected, None]
        rotations = Rotation.from_quat(xyzw)
        rpy = rotations.as_euler("xyz")
        canonical_states[selected, :3] = states[selected, :3]
        canonical_states[selected, 3:6] = rpy
        targets = Rotation.from_rotvec(actions[selected, 3:6]) * rotations
        canonical_actions[selected, 3:6] = wrap_angles(targets.as_euler("xyz") - rpy)
    canonical_states[:, 6] = np.clip(states[:, 7], 0.0, 1.0)
    canonical_actions[:, 6] = (actions[:, 6] >= 0.5).astype(np.float64)
    state_valid = quaternion_valid & _state_valid(canonical_states)
    valid = quaternion_valid & action_valid & _action_valid(canonical_actions)
    return canonical_states.astype(np.float32), canonical_actions.astype(np.float32), state_valid, valid


@dataclass(frozen=True)
class CanonicalChunk:
    state: np.ndarray
    action: np.ndarray
    temporal_mask: np.ndarray
    dimension_mask: np.ndarray

    def __post_init__(self) -> None:
        horizon = self.action.shape[0]
        if self.state.shape != (CANONICAL_DIM,):
            raise ValueError("canonical state must have shape [7]")
        if self.action.shape != (horizon, CANONICAL_DIM):
            raise ValueError("canonical action must have shape [H,7]")
        if self.temporal_mask.shape != (horizon,):
            raise ValueError("temporal mask must have shape [H]")
        if self.dimension_mask.shape != (CANONICAL_DIM,):
            raise ValueError("dimension mask must have shape [7]")

    @property
    def fm_count(self) -> int:
        return int(self.temporal_mask.sum() * self.dimension_mask.sum())


def _sequential(length: int, base: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    if length <= 0 or not 0 <= base < length or horizon <= 0:
        raise ValueError("invalid episode/base/horizon")
    raw = base + np.arange(horizon, dtype=np.int64)
    valid = raw < length
    return np.minimum(raw, length - 1), valid


def canonical_molmo_chunk(states, actions, base: int, horizon: int) -> CanonicalChunk:
    states, actions, state_valid, action_valid = canonical_molmo_arrays(states, actions)
    indices, temporal = _sequential(len(states), base, horizon)
    if not state_valid[base]:
        raise ValueError("Molmo canonical state is invalid")
    return CanonicalChunk(
        state=states[base], action=actions[indices],
        temporal_mask=temporal & action_valid[indices],
        dimension_mask=np.ones(7, dtype=bool),
    )


def canonical_droid_chunk(
    state_poses,
    state_grippers,
    target_poses,
    target_grippers,
    base: int,
    horizon: int,
) -> CanonicalChunk:
    states, actions, state_valid, action_valid = canonical_droid_arrays(
        state_poses, state_grippers, target_poses, target_grippers
    )
    indices, temporal = _sequential(len(states), base, horizon)
    if not state_valid[base]:
        raise ValueError("DROID canonical state is invalid")
    return CanonicalChunk(
        state=states[base], action=actions[indices],
        temporal_mask=temporal & action_valid[indices],
        dimension_mask=np.ones(7, dtype=bool),
    )


def canonical_rh20t_chunk(states, actions, action_valid, base: int, horizon: int) -> CanonicalChunk:
    states, actions, state_valid, action_valid = canonical_rh20t_arrays(
        states, actions, action_valid
    )
    indices, temporal = _sequential(len(states), base, horizon)
    if not state_valid[base]:
        raise ValueError("RH20T canonical state is invalid")
    return CanonicalChunk(
        state=states[base], action=actions[indices],
        temporal_mask=temporal & action_valid[indices],
        dimension_mask=np.ones(7, dtype=bool),
    )


def normalize(values, q01, q99, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    q01 = np.asarray(q01, dtype=np.float64)
    q99 = np.asarray(q99, dtype=np.float64)
    return np.clip(2.0 * (values - q01) / (q99 - q01 + eps) - 1.0, -15.0, 15.0)


def denormalize(values, q01, q99, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    q01 = np.asarray(q01, dtype=np.float64)
    q99 = np.asarray(q99, dtype=np.float64)
    return (values + 1.0) * 0.5 * (q99 - q01 + eps) + q01
