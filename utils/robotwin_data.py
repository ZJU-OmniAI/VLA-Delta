"""RoboTwin ALOHA AgileX inputs shared by training and evaluation."""

from __future__ import annotations

import numpy as np

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
JOINT_NAMES = tuple(
    f"{arm}_{joint}"
    for arm in ("left", "right")
    for joint in (
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
    )
)


def validate_robotwin_metadata(info, stats, *, use_quantile=True):
    """Check the three views and native 14D joint representation before padding."""
    if info.get("codebase_version") not in {"v2.0", "v2.1"}:
        raise ValueError("RoboTwin fine-tuning requires a converted LeRobot v2 dataset")
    if info.get("robot_type") not in {"aloha-agilex", "aloha_agilex"}:
        raise ValueError("RoboTwin recipe requires ALOHA AgileX joint actions")
    fps = info.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not np.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("RoboTwin metadata requires a positive finite FPS")
    features = info.get("features", {})
    cameras = tuple(
        key
        for key, value in features.items()
        if value.get("dtype") in {"video", "image"}
    )
    if cameras != CAMERA_KEYS:
        raise ValueError(f"RoboTwin camera order must be {CAMERA_KEYS}, got {cameras}")
    for key in CAMERA_KEYS:
        feature = features[key]
        shape, names = feature.get("shape", []), feature.get("names")
        if names == ["channels", "height", "width"]:
            rgb = len(shape) == 3 and shape[0] == 3
        elif names == ["height", "width", "channels"]:
            rgb = len(shape) == 3 and shape[2] == 3
        else:
            rgb = len(shape) == 3 and (shape[0] == 3 or shape[2] == 3)
        if not rgb or any(type(value) is not int or value <= 0 for value in shape):
            raise ValueError(f"{key}: expected a positive RGB image shape")
    for key in ("observation.state", "action"):
        feature = features.get(key, {})
        if tuple(feature.get("shape", ())) != (14,) or feature.get("dtype") not in {
            "float32",
            "float64",
        }:
            raise ValueError(
                f"RoboTwin {key} must contain 14 floating-point joint values"
            )
        names = feature.get("names")
        if names is not None:
            if len(names) == 1 and isinstance(names[0], list):
                names = names[0]
            if tuple(names) != JOINT_NAMES:
                raise ValueError(
                    f"RoboTwin {key} joint order must be left arm/gripper, then right arm/gripper"
                )
        low, high = ("q01", "q99") if use_quantile else ("min", "max")
        bounds = []
        for name in (low, high):
            values = np.asarray((stats or {}).get(key, {}).get(name), dtype=np.float64)
            if values.shape != (14,) or not np.isfinite(values).all():
                raise ValueError(
                    f"RoboTwin statistics require 14 finite values for {key}.{name}"
                )
            bounds.append(values)
        if np.any(bounds[0] > bounds[1]):
            raise ValueError(
                f"RoboTwin {key} normalization lower bounds exceed upper bounds"
            )
    return dict(
        camera_keys=list(CAMERA_KEYS),
        action_dim=14,
        state_dim=14,
        normalization="quantile" if use_quantile else "min_max",
    )
