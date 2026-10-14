# Adapted from RoboTwin; original MIT license: licenses/robotwin.LICENSE.
import os
import sys
import subprocess
import traceback
import yaml
import importlib
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

# --------------------------------------------------------------------------------------
# Path setup
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = ROOT / "third_party/robotwin"
sys.path.insert(0, str(ROOT))
sys.path.extend([str(ROBOTWIN_ROOT), str(ROBOTWIN_ROOT / "script"),
                 str(ROBOTWIN_ROOT / "description/utils")])

CONFIGS_PATH = str(ROBOTWIN_ROOT / "task_config") + os.sep
from utils.websocket_client_policy import WebsocketClientPolicy
from generate_episode_instructions import generate_episode_descriptions
from evaluation.robotwin import policy as policy_model


def class_decorator(task_name):
    """Create RoboTwin task environment instance by task name."""
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception as e:
        raise SystemExit(f"No Task: {task_name}. Error: {repr(e)}")
    return env_instance


def eval_function_decorator(policy_name, model_name, conda_env=None):
    """Resolve the retained VLA policy; ZR0 remains a compatible config name."""
    if policy_name not in {"ZR0", "vla_delta"}:
        raise ValueError("Only the VLA-∆ WebSocket policy is included")
    return getattr(policy_model, model_name)


def get_camera_config(camera_type):
    camera_config_path = ROBOTWIN_ROOT / "task_config/_camera_config.yml"
    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    # Upstream tasks resolve asset/planner paths from the simulator directory.
    os.chdir(ROBOTWIN_ROOT)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]

    host = usr_args["host"]
    port = int(usr_args["port"])
    test_num = int(usr_args["n_episodes"])
    max_episode_steps = int(usr_args["max_episode_steps"])

    policy_conda_env = usr_args.get("policy_conda_env", None)

    # Kept only for compatibility with RoboTwin's original policy interface.
    # The actual ZR-0 model is loaded in server.py, not here.
    #_ = eval_function_decorator(policy_name, "get_model", conda_env=policy_conda_env)

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["max_episode_steps"] = max_episode_steps
    args["n_action_steps"] = int(usr_args["n_action_steps"])

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type_item):
        robot_file = _embodiment_types[embodiment_type_item]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(usr_args.get("output_dir") or f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    video_size = None
    if args.get("eval_video_log", False):
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        args["eval_video_save_dir"] = save_dir

    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\033[94mZR-0 Server:\033[0m " + f"{host}:{port}")
    print("\033[94mTest Num:\033[0m " + str(test_num))
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name

    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    # Attach non-image policy parameters to TASK_ENV.
    # Do not set any resize-related fields here. Images keep RoboTwin's original camera resolution.
    TASK_ENV.n_action_steps = int(usr_args["n_action_steps"])
    TASK_ENV.step_lim = max_episode_steps
    TASK_ENV.left_arm_dim = usr_args["left_arm_dim"]
    TASK_ENV.right_arm_dim = usr_args["right_arm_dim"]

    seed = int(usr_args["seed"])
    st_seed = 100000 * (1 + seed)

    # Connect to already-running ZR-0 websocket server.
    model = WebsocketClientPolicy(host, port, connect_timeout=120)

    from utils.evaluation_record import EvaluationRecord, validate_server
    metadata = model.get_server_metadata()
    validate_server(metadata, action_dim=14, execution_horizon=usr_args["n_action_steps"])
    recorder = EvaluationRecord(save_dir, expected=test_num, metadata={"benchmark": "RoboTwin",
        "arguments": usr_args, "server": metadata, "expert_check": True,
        "camera_config": args["camera"], "client_resize": False, "final_model_resolution": [224, 224]})
    args["evaluation_recorder"] = recorder
    try:
        st_seed, suc_num = eval_policy(
            task_name=task_name,
            TASK_ENV=TASK_ENV,
            args=args,
            model=model,
            st_seed=st_seed,
            test_num=test_num,
            video_size=video_size,
            instruction_type=instruction_type,
            policy_conda_env=policy_conda_env,
        )
    except BaseException as error:
        recorder.episode(success=False, error=type(error).__name__, task=task_name,
                         seed=getattr(recorder, "current_seed", st_seed))
        raise
    finally:
        recorder.close()
        model.close()
        try:
            TASK_ENV.close_env()
        except Exception:
            pass  # Preserve the original rollout/setup exception.
    file_path = os.path.join(save_dir, "_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(f"Success: {suc_num}/{test_num}\n")
        file.write(f"Success Rate: {suc_num / test_num:.6f}\n")

    print(f"Data has been saved to {file_path}")


def eval_policy(
    task_name,
    TASK_ENV,
    args,
    model,
    st_seed,
    test_num=100,
    video_size=None,
    instruction_type=None,
    policy_conda_env=None,
):
    # Upstream envs initialize relative asset paths while importing. Import only
    # after main() has selected the simulator directory, never during CLI help.
    from envs.utils.create_actor import UnStableError
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    # Official RoboTwin evaluation usually checks whether the sampled seed is solvable by expert first.
    # For debug, pass --expert_check False in overrides.
    expert_check = bool(args.get("expert_check", True))

    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval", conda_env=policy_conda_env)
    reset_func = eval_function_decorator(policy_name, "reset_model", conda_env=policy_conda_env)

    now_seed = st_seed
    clear_cache_freq = int(args.get("clear_cache_freq", 10))
    args["eval_mode"] = True

    recorder = args.pop("evaluation_recorder", None)
    while succ_seed < test_num:
        if now_seed - st_seed >= test_num * 100:
            raise RuntimeError("expert seed search exceeded 100 attempts per requested rollout")
        if recorder is not None:
            recorder.current_seed = now_seed
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        # ----------------------------------------------------------
        # 1. Expert check: find a feasible seed
        # ----------------------------------------------------------
        episode_info = None
        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                TASK_ENV.step_lim = int(args["max_episode_steps"])
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                print(" -------------")
                print("UnStableError: ", e)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception:
                TASK_ENV.close_env()
                raise  # Missing resources and simulator failures must remain incomplete.

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        # ----------------------------------------------------------
        # 2. Policy evaluation on the selected seed
        # ----------------------------------------------------------
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        TASK_ENV.step_lim = int(args["max_episode_steps"])

        if episode_info is not None:
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
            instruction = np.random.choice(results[0][instruction_type])
        else:
            # Fallback for debug if expert_check=False.
            results = generate_episode_descriptions(args["task_name"], [], test_num)
            try:
                instruction = np.random.choice(results[0][instruction_type])
            except Exception:
                instruction = getattr(TASK_ENV, "instruction", args["task_name"])

        TASK_ENV.set_instruction(instruction=instruction)

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False

        # Very important: reset per-episode action queue in policy/zr0_robotwin.py.
        reset_func(model)

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()

            # eval_func must internally cache action_chunk:
            #   if queue empty -> call model.infer(...)
            #   else -> pop one action and TASK_ENV.take_action(action)
            eval_func(TASK_ENV, model, observation)

            if TASK_ENV.eval_success:
                succ = True
                break

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if getattr(TASK_ENV, "render_freq", 0):
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1
        if recorder is not None:
            recorder.episode(task=task_name, profile=args["task_config"], seed=now_seed,
                             episode=now_id - 1, success=succ, steps=TASK_ENV.take_action_cnt)

        print(
            f"\033[93m{task_name}\033[0m | "
            f"\033[94m{args['policy_name']}\033[0m | "
            f"\033[92m{args['task_config']}\033[0m | "
            f"\033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m "
            f"=> \033[95m{round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%\033[0m, "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )

        now_seed += 1

    return now_seed, TASK_ENV.suc

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    cli_args = parser.parse_args()

    with open(cli_args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Empty yaml config: {cli_args.config}")

    required_keys = [
        "policy_name",
        "task_name",
        "task_config",
        "ckpt_setting",
        "seed",
        "instruction_type",
        "host",
        "port",
        "max_episode_steps",
        "n_episodes",
        "n_action_steps",
    ]

    missing_keys = [key for key in required_keys if key not in config or config[key] is None]
    if missing_keys:
        raise KeyError(
            f"Missing required keys in yaml config {cli_args.config}: {missing_keys}"
        )

    config["port"] = int(config["port"])
    config["seed"] = int(config["seed"])
    config["max_episode_steps"] = int(config["max_episode_steps"])
    config["n_episodes"] = int(config["n_episodes"])
    config["n_action_steps"] = int(config["n_action_steps"])

    return config


if __name__ == "__main__":
    usr_args = parse_args_and_config()
    main(usr_args)
