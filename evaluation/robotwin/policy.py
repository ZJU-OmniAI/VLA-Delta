# Adapted from RoboTwin; original MIT license: licenses/robotwin.LICENSE.
import collections

import numpy as np

from utils.websocket_client_policy import WebsocketClientPolicy


def _get_required_arg(usr_args, key):
    if isinstance(usr_args, dict):
        if key not in usr_args:
            raise KeyError(f"Missing required config field: {key}")
        return usr_args[key]

    if not hasattr(usr_args, key):
        raise KeyError(f"Missing required config field: {key}")

    return getattr(usr_args, key)


def _prepare_image_no_resize(img):
    img = np.asarray(img)

    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]

    return np.ascontiguousarray(img)


def get_model(usr_args):
    host = str(_get_required_arg(usr_args, "host"))
    port = int(_get_required_arg(usr_args, "port"))
    n_action_steps = int(_get_required_arg(usr_args, "n_action_steps"))

    model = WebsocketClientPolicy(host, port)
    model.action_plan = collections.deque()
    model.n_action_steps = n_action_steps

    return model


def reset_model(model):
    if hasattr(model, "action_plan"):
        model.action_plan.clear()

    if hasattr(model, "reset"):
        model.reset()
    elif hasattr(model, "call"):
        model.call(func_name="reset_model")


def encode_obs(observation):
    obs = observation["observation"]

    head_img = _prepare_image_no_resize(obs["head_camera"]["rgb"])
    left_img = _prepare_image_no_resize(obs["left_camera"]["rgb"])
    right_img = _prepare_image_no_resize(obs["right_camera"]["rgb"])

    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)

    return head_img, left_img, right_img, state


def build_payload(TASK_ENV, observation):
    n_action_steps = int(TASK_ENV.n_action_steps)

    head_img, left_img, right_img, state = encode_obs(observation)

    payload = {
        "task": str(TASK_ENV.get_instruction()),
        "observation.state": state,
        "n_action_steps": n_action_steps,
        "observation.images.cam_high": head_img,
        "observation.images.cam_left_wrist": left_img,
        "observation.images.cam_right_wrist": right_img,
    }

    return payload


def call_policy_model(model, payload):
    result = model.infer(payload)

    if "actions" not in result:
        raise KeyError(f"Server response has no `actions`. Keys: {list(result.keys())}")

    actions = np.asarray(result["actions"], dtype=np.float32)

    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]

    if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite actions shape (T, 14), got {actions.shape}")

    return actions


def eval(TASK_ENV, model, observation):
    if not hasattr(model, "action_plan"):
        model.action_plan = collections.deque()

    n_action_steps = int(TASK_ENV.n_action_steps)

    if not model.action_plan:
        payload = build_payload(TASK_ENV, observation)
        action_chunk = call_policy_model(model, payload)

        if action_chunk.shape[0] < n_action_steps:
            raise ValueError(
                f"Need {n_action_steps} actions, but server returned {action_chunk.shape[0]}. "
                f"Action chunk shape: {action_chunk.shape}"
            )

        model.action_plan.extend(action_chunk[:n_action_steps])

    action = np.asarray(model.action_plan.popleft(), dtype=np.float32)
    TASK_ENV.take_action(action)

    return action
