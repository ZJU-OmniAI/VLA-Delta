#!/usr/bin/env python3
"""Launch the existing trainer with one portable recipe; never trains by default."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "lerobot")]


def build_command(args, overrides):
    from utils.cli_options import parse_train_options
    from utils.runtime_config import load_mapping
    config = load_mapping(args.config, resolve=False)
    training = ["--config", str(Path(args.config).resolve()), *overrides]
    if args.resume:
        source = str(Path(args.resume).resolve())
        training += ["--resume_training", "--vlm_name_or_path", source,
            "--init_from_checkpoint", "", "--action_expert_config_path", str(Path(source) / "action_expert_config.json")]
        if config.get("training_stage"):
            training += ["--resume_from_checkpoint", source]
        if config.get("training_stage") not in {"stage1_ar", "stage2_aux"}:
            training += ["--action_expert_name_or_path", source]
    options = parse_train_options(training)
    if not options.portable_run:
        raise ValueError("public launcher requires portable_run=true")
    from utils.portable_run import configure_run
    configure_run(options)
    expected = args.num_processes * options.per_device_train_batch_size * options.gradient_accumulation_steps
    if options.expected_global_batch_size != expected:
        raise ValueError(f"global batch mismatch: {expected}; adjust batch size/accumulation or expected_global_batch_size explicitly")
    command = [sys.executable, "-m", "accelerate.commands.launch", "--config_file",
        str(Path(args.accelerate_config).resolve()), "--num_processes", str(args.num_processes),
        "--main_process_port", str(args.main_process_port), "--mixed_precision", "bf16",
        "--gradient_accumulation_steps", str(options.gradient_accumulation_steps),
        str(ROOT / "train_vla.py"), *training]
    return command, options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--accelerate-config", default=str(ROOT / "configs/accelerate.yaml"),
        help="Accelerate YAML; defaults to the repository ZeRO-2 configuration")
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--main-process-port", type=int, default=29500)
    parser.add_argument("--resume", help="Complete same-stage latest-model-optimizer-lr directory")
    parser.add_argument("--execute", action="store_true", help="Start training; omitted prints the command only")
    parser.add_argument("--dry-run", action="store_true", help="Explicitly print the command only")
    args, overrides = parser.parse_known_args()
    if args.execute and args.dry_run:
        parser.error("choose --execute or --dry-run")
    command, options = build_command(args, overrides)
    print(shlex.join(command), flush=True)
    print(json.dumps({"stage": options.training_stage, "loss_type": options.loss_type,
        "global_batch_size": options.expected_global_batch_size, "output": options.output_ckpt_dir,
        "resume": options.resume_training}, indent=2), flush=True)
    if args.execute:
        raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
