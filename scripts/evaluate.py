#!/usr/bin/env python3
"""Evaluate LIBERO or RoboTwin against an independently running model server."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def jobs(args):
    output = Path(args.output_dir).resolve()
    if args.trials < 1 or args.execution_horizon < 1:
        raise ValueError("trials and execution horizon must be positive")
    if args.benchmark == "libero":
        for suite in args.suites:
            if suite not in {"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"}:
                raise ValueError(f"unknown LIBERO suite: {suite}")
            command = [sys.executable, "-m", "evaluation.libero.run_eval",
                "--args.host", args.host, "--args.port", str(args.port),
                "--args.task-suite-name", suite, "--args.num-trials-per-task", str(args.trials),
                "--args.replan-steps", str(args.execution_horizon), "--args.seed", str(args.seed),
                "--args.video-out-path", str(output / suite)]
            if args.tasks:
                command += ["--args.task-ids", *args.tasks]
            yield suite, command, ROOT, None
    else:
        import yaml
        robotwin = ROOT / "third_party/robotwin"
        if not (robotwin / "task_config/_eval_step_limit.yml").is_file():
            raise FileNotFoundError("Download RoboTwin into third_party/robotwin first; see evaluation/robotwin/README.md")
        limits = yaml.safe_load((robotwin / "task_config/_eval_step_limit.yml").read_text())
        for profile in args.profiles:
            if profile not in {"demo_clean", "demo_randomized"}:
                raise ValueError("RoboTwin profile must be demo_clean or demo_randomized")
            for task in args.tasks or sorted(limits):
                if task not in limits:
                    raise ValueError(f"unknown RoboTwin task: {task}")
                name = f"{profile}__{task}"
                config = dict(policy_name="ZR0", task_name=task, task_config=profile,
                    max_episode_steps=limits[task], n_episodes=args.trials,
                    n_action_steps=args.execution_horizon, ckpt_setting="vla_delta",
                    seed=args.seed, instruction_type="unseen", host=args.host, port=args.port,
                    output_dir=str(output / name))
                # Native simulator libraries may reset LC_CTYPE to C. Keep Python
                # file decoding in UTF-8 for asset paths containing e.g. VLA-∆.
                command = [sys.executable, "-X", "utf8", str(ROOT / "evaluation/robotwin/run_eval.py"),
                           "--config", str(output / "configs" / f"{name}.json")]
                yield name, command, robotwin, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--benchmark", choices=("libero", "robotwin"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--profiles", nargs="+", default=["demo_clean", "demo_randomized"])
    parser.add_argument("--tasks", nargs="+", help="LIBERO integer task IDs, or RoboTwin task names")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    from utils.runtime_config import config_arguments
    args = parser.parse_args(config_arguments(parser, sys.argv[1:]))
    if args.execute and args.dry_run:
        parser.error("choose --execute or --dry-run")
    plan = list(jobs(args))
    for name, command, cwd, config in plan:
        print(f"{name}: (cd {shlex.quote(str(cwd))} && {shlex.join(command)})")
    if not args.execute:
        return
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    (output / "experiment.md").write_text("# VLA-∆ evaluation\n\nStarted: " + datetime.now(timezone.utc).isoformat()
        + "\n\n```json\n" + json.dumps(vars(args), indent=2) + "\n```\n\n"
        "Actual checkpoint and preprocessing are read from the model server and saved per group in evaluation.json.\n")
    status = dict(status="running", completed_groups=0, expected_groups=len(plan), successes=0, completed=0)
    failure = None
    try:
        for name, command, cwd, config in plan:
            if config:
                (output / "configs" / f"{name}.json").write_text(json.dumps(config, indent=2))
            with (output / f"{name}.log").open("x") as stream:
                subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)
            result = json.loads((output / name / "summary.json").read_text())
            if not result["complete"]:
                raise RuntimeError(f"incomplete evaluation group: {name}")
            status["completed_groups"] += 1
            status["successes"] += result["successes"]
            status["completed"] += result["completed"]
        status["status"] = "complete"
    except BaseException as error:
        failure = error
        status.update(status="incomplete", failed_group=name, error=type(error).__name__)
        raise
    finally:
        status["success_rate"] = status["successes"] / status["completed"] if status["completed"] else None
        status["ended_at"] = datetime.now(timezone.utc).isoformat()
        (output / "summary.json").write_text(json.dumps(status, indent=2) + "\n")
        with (output / "experiment.md").open("a") as stream:
            stream.write("\n## Result\n\n```json\n" + json.dumps(status, indent=2) + "\n```\n")


if __name__ == "__main__":
    main()
