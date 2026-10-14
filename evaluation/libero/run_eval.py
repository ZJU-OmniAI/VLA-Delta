'''
LIBERO's evaluation script.

Copied from openpi: https://github.com/Physical-Intelligence/openpi/blob/95aadc6b16d170e4b13ab2e0ac64fbf2d1bb8e31/examples/libero/main.py

Modified to fit our setup.
'''

from typing import Tuple
import collections
import dataclasses
import json
import logging
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from utils.websocket_client_policy import WebsocketClientPolicy
from utils import image_tools
from utils.evaluation_record import EvaluationRecord, validate_server

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 448
    replan_steps: int = 10
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "outputs/eval/libero"
    seed: int = 7
    task_ids: Tuple[int, ...] = ()


def _camera_keys(metadata):
    """Use the training field names advertised by the checkpoint's dataset."""
    cameras = metadata.get("dataset", {}).get("camera_keys", [])
    main = "observation.images.image"
    wrists = ("observation.images.image2", "observation.images.wrist_image")
    if len(cameras) == 2 and main in cameras:
        for wrist in wrists:
            if wrist in cameras:
                return main, wrist
    raise ValueError(
        "LIBERO requires observation.images.image and either "
        "observation.images.image2 or observation.images.wrist_image "
        f"in the checkpoint camera metadata; got {cameras!r}"
    )


def eval_libero(args: Args) -> None:
    # Respect the operator's rendering choice; imports initialize MuJoCo.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    from libero.libero import benchmark
    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    tasks = args.task_ids or tuple(range(suite.n_tasks))
    if args.num_trials_per_task < 1 or any(i < 0 or i >= suite.n_tasks for i in tasks):
        raise ValueError("invalid trial count or task index")
    limits = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300,
              "libero_10": 520, "libero_90": 400}
    client = WebsocketClientPolicy(args.host, args.port, connect_timeout=120)
    metadata = client.get_server_metadata()
    validate_server(metadata, action_dim=7, execution_horizon=args.replan_steps)
    main_camera, wrist_camera = _camera_keys(metadata)
    record = EvaluationRecord(args.video_out_path, expected=len(tasks) * args.num_trials_per_task,
        metadata=dict(benchmark="LIBERO", arguments=dataclasses.asdict(args), server=metadata,
            camera_mapping={main_camera: "agentview_image", wrist_camera: "robot0_eye_in_hand_image"},
            source_resolution=[256, 256], rotation_degrees=180, client_resize=[args.resize_size]*2,
            final_model_resolution=[224, 224], control_frequency_hz=20,
            success_definition="environment done=True within the task step limit"))
    try:
        for task_id in tasks:
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            if args.num_trials_per_task > len(initial_states):
                raise ValueError("trial count exceeds available initial states")
            env, description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            try:
                for episode_idx in range(args.num_trials_per_task):
                    frames, actions = [], collections.deque()
                    done, error, t = False, None, 0
                    try:
                        env.reset()
                        obs = env.set_init_state(initial_states[episode_idx])
                        for t in range(limits[args.task_suite_name] + args.num_steps_wait):
                            if t < args.num_steps_wait:
                                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                                continue
                            images = [image_tools.convert_to_uint8(image_tools.resize_with_pad(
                                np.ascontiguousarray(obs[key][::-1, ::-1]), args.resize_size, args.resize_size))
                                for key in ("agentview_image", "robot0_eye_in_hand_image")]
                            frames.append(images[0])
                            if not actions:
                                payload = {main_camera: images[0],
                                    wrist_camera: images[1],
                                    "observation.state": np.concatenate((obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"].copy()), obs["robot0_gripper_qpos"])),
                                    "task": str(description), "n_action_steps": args.replan_steps}
                                chunk = np.asarray(client.infer(payload)["actions"])
                                if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < args.replan_steps or not np.isfinite(chunk).all():
                                    raise ValueError("invalid action chunk from model server")
                                actions.extend(chunk[:args.replan_steps])
                            obs, _, done, _ = env.step(actions.popleft())
                            if done:
                                break
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    record.episode(task_id=task_id, episode=episode_idx, seed=args.seed,
                                   steps=t, success=done, error=error)
                    if frames:
                        imageio.mimwrite(Path(args.video_out_path) / f"task-{task_id:02d}_episode-{episode_idx:03d}.mp4", frames, fps=20)
                    if error:
                        raise RuntimeError(error)
            finally:
                env.close()
    finally:
        summary = record.close()
        client.close()
        logging.info("Evaluation summary: %s", json.dumps(summary))


def _get_libero_env(task, resolution, seed):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env, task.language


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    import tyro
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
