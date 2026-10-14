"""Release-run records, exact-resume contracts and portable inference metadata."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from utils.checkpoint_metadata import checkpoint_metadata, validate_normalization_snapshot
import json
from pathlib import Path
import shutil
import subprocess

RECORD = "portable_run.json"
META_FILES = ("info.json", "stats.json", "tasks.jsonl", "episodes.jsonl")
# Changes here require a new initialization, never an implicit resume migration.
RESUME_FIELDS = ("training_stage", "loss_type", "action_horizon", "window_size",
    "max_pad_state_and_action_length", "num_difference_queries", "num_flow_queries",
    "per_device_train_batch_size", "gradient_accumulation_steps", "expected_global_batch_size",
    "epochs", "max_train_steps", "seed", "peak_learning_rate", "min_lr_rate", "warmup_ratio",
    "adam_beta1", "adam_beta2", "adam_epsilon", "lr_scheduler", "component_optimizer_groups",
    "vlm_loss_weight", "action_expert_loss_weight", "slot_loss_weight", "optical_flow_loss_weight",
    "tune_vlm", "tune_action_expert", "detach_vlm_outputs_for_action_expert", "max_length",
    "dataset_entries", "dataset_sample_ratios")


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    pending.replace(path)


def update_run(options, **fields):
    """Merge main-rank run records without changing the resolved options."""
    path = Path(options.output_ckpt_dir) / RECORD
    record = json.loads(path.read_text())
    record.update(fields)
    write_json(path, record)


def record_training_summary(options, *, completed, limit, epoch, scheduler_steps):
    update_run(options, training_summary=dict(successful_updates=completed,
        requested_update_limit=limit, reached_update_limit=completed >= limit,
        last_epoch=epoch, scheduler_steps=scheduler_steps))


def create_tensorboard_writer(options, accelerator):
    if not accelerator.is_main_process or not getattr(options, "enable_tensorboard", False):
        return None
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(options.tensorboard_log_dir)


def configure_run(options):
    """Pure validation/option resolution, shared by dry-run and all ranks."""
    if getattr(options, "downstream_runtime_state", False):
        raise ValueError("portable_run already manages fine-tuning recovery; do not combine it with the legacy downstream runtime")
    if options.wandb_project:
        if options.wandb_failure_policy != "required":
            raise ValueError("portable training requires wandb_failure_policy=required when W&B is enabled")
        import os
        if os.environ.get("WANDB_MODE", "online") != "online":
            raise ValueError("enabled W&B requires online mode")
    if not options.save_optimizer_and_lr_states:
        raise ValueError("portable training requires full optimizer/runtime checkpoints")
    saved_options = {}
    if options.resume_training:
        source = Path(options.vlm_name_or_path)
        if source.name != "latest-model-optimizer-lr":
            raise ValueError("resume needs latest-model-optimizer-lr; model exports do not contain optimizer state")
        saved = json.loads((source / RECORD).read_text())
        different = [key for key in RESUME_FIELDS if saved["options"].get(key) != getattr(options, key)]
        if different:
            raise ValueError(f"resume configuration changed: {different}")
        saved_options = saved["options"]
    if not options.wandb_project:
        options.wandb_run_id = ""
        options.wandb_run_name = ""
        options.wandb_resume = "never"
    elif saved_options.get("wandb_project"):
        if saved_options["wandb_project"] != options.wandb_project:
            raise ValueError("resume must use the original W&B project")
        options.wandb_run_id = saved_options["wandb_run_id"]
        options.wandb_run_name = saved_options["wandb_run_name"]
        options.wandb_resume = "must"
    else:
        options.wandb_run_id = options.wandb_run_id or hashlib.sha256(str(Path(options.output_ckpt_dir).resolve()).encode()).hexdigest()[:16]
        options.wandb_run_name = options.wandb_run_name or Path(options.output_ckpt_dir).name
        options.wandb_resume = "never"
    return options


def begin_run(options):
    root = Path(options.output_ckpt_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"choose a new output directory, including for resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    def git(*args):
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    try:
        commit, status = git("rev-parse", "HEAD"), git("status", "--short", "-uno")
    except (OSError, subprocess.CalledProcessError):
        commit, status = "unavailable (source archive)", "unavailable"
    import os, shlex, sys
    payload = dict(version=1, created_at=datetime.now(timezone.utc).isoformat(), owner=os.environ.get("USER", "unknown"),
                   commit=commit, git_status=status, command=shlex.join([sys.executable, *sys.argv]),
                   options=vars(options).copy(), status="starting")
    write_json(root / RECORD, payload)


def bind_run(model, options, accelerator, dataset):
    model.portable_run = True
    model.portable_options = vars(options).copy()
    if not accelerator.is_main_process:
        return
    modules = {"vision_encoder": model.backbone.model.model.visual,
               "language_model": model.backbone.model.model.language_model,
               "difference_query": model.backbone.difference_query,
               "action_expert": model.action_expert, "slot_aux": model.slot_aux,
               "optical_flow_aux": model.optical_flow_aux}
    # Merger is also included in vision_encoder; record it explicitly without
    # serializing thousands of individual parameter names.
    modules["vision_merger"] = getattr(model.backbone.model.model.visual, "merger", None)
    details = {name: dict(present=module is not None,
                         parameters=sum(p.numel() for p in module.parameters()) if module is not None else 0,
                         trainable_parameters=sum(p.numel() for p in module.parameters() if p.requires_grad) if module is not None else 0)
               for name, module in modules.items()}
    counts = {ds.spec.dataset_entry: len(ds) for ds in dataset.datasets}
    image_processor = model.backbone.processor.image_processor
    preprocessing = dict(final_image_size=[224, 224], resize="PIL RGB bicubic; explicit square size",
        preserve_aspect_ratio=False, crop=False, pad=False, augmentation=False,
        image_mean=image_processor.image_mean, image_std=image_processor.image_std,
        rescale_factor=image_processor.rescale_factor,
        do_rescale=getattr(image_processor, "do_rescale", True),
        do_normalize=getattr(image_processor, "do_normalize", True), processor_resize=False,
        multiview="separate image tokens in resolved manifest camera order")
    sources = {}
    for ds in dataset.datasets:
        info = json.loads((Path(ds.spec.dataset_path) / "meta/info.json").read_text())
        sources[ds.spec.dataset_entry] = {key: info.get(key) for key in ("codebase_version", "fps", "splits", "total_frames", "total_episodes", "features")}
    import torch
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    update_run(options,
        module_summary=dict(components=details, vision_merger_included_in="vision_encoder"),
        data_and_preprocessing=dict(datasets=sources, preprocessing=preprocessing, sample_counts=counts),
        training=dict(world_size=accelerator.num_processes, gpu_models=devices,
            global_batch_size=accelerator.num_processes * options.per_device_train_batch_size * options.gradient_accumulation_steps,
            precision=accelerator.mixed_precision, optimizer="AdamW", weight_decay=0.01,
            gradient_clipping=1.0, gradient_checkpointing=bool(options.tune_vlm),
            prediction_horizon=options.action_horizon, execution_horizon=None,
            padded_dimension=options.max_pad_state_and_action_length,
            loss_weights=dict(ar=options.vlm_loss_weight, slot=options.slot_loss_weight,
                optical_flow=options.optical_flow_loss_weight, flow_matching=options.action_expert_loss_weight),
            sampling="shuffled concatenated admitted frames", validation=None, early_stopping=None))


def save_portable_artifacts(model, checkpoint):
    options = model.portable_options
    checkpoint = Path(checkpoint)
    run = Path(options["output_ckpt_dir"])
    source = run / RECORD
    record = json.loads(source.read_text()) if source.is_file() else {"version": 1}
    record.update(options=options, exported_at=datetime.now(timezone.utc).isoformat())
    identity = run / "wandb_identity.json"
    if options.get("wandb_project") and identity.is_file():
        record["wandb"] = json.loads(identity.read_text())
    else:
        record.pop("wandb", None)
    write_json(checkpoint / RECORD, checkpoint_metadata(record))
    if not options.get("export_inference_metadata", False):
        return
    entries = []
    manifest = model.resolved_dataset_manifest
    for number, entry in enumerate(manifest["entries"]):
        if entry["resolved_adapter"] != "lerobot_v2":
            continue
        relative = f"inference_metadata/{number:03d}"
        target = checkpoint / relative / "meta"
        target.mkdir(parents=True, exist_ok=True)
        for name in META_FILES:
            shutil.copyfile(Path(entry["dataset_path"]) / "meta" / name, target / name)
        validate_normalization_snapshot(json.loads((target / "stats.json").read_text()), entry)
        entries.append(dict(dataset_entry=entry["dataset_entry"], directory=relative,
                            files=list(META_FILES)))
    if entries:
        write_json(checkpoint / "inference_metadata.json", dict(version=1, entries=entries))


def inference_dataset_configs(configs, checkpoint, *, stats_file=None):
    # A complete, explicitly selected external meta directory is sufficient
    # for inference; the published model need not carry duplicate metadata.
    if stats_file is not None and len(configs) == 1:
        stats_path = Path(stats_file).resolve()
        meta = stats_path.parent
        if (stats_path.is_file() and meta.name == "meta"
                and all((meta / name).is_file() for name in META_FILES if name != "stats.json")):
            return {name: {**entry, "dataset_path": str(meta.parent)}
                    for name, entry in configs.items()}
    checkpoint = Path(checkpoint).resolve()
    index_path = checkpoint / "inference_metadata.json"
    if not index_path.is_file():
        return configs
    from utils.dataset_manifest import load_resolved_dataset_manifest
    manifest = load_resolved_dataset_manifest(checkpoint)
    saved = {row["dataset_entry"]: row for row in manifest["entries"]}
    index = json.loads(index_path.read_text())
    if index.get("version") != 1:
        raise ValueError("unsupported portable inference metadata version")
    result = {name: dict(entry) for name, entry in configs.items()}
    found = set()
    for row in index["entries"]:
        name = row["dataset_entry"]
        if name in found or name not in saved:
            raise ValueError("invalid portable inference dataset identity")
        found.add(name)
        root = (checkpoint / row["directory"]).resolve()
        if not root.is_relative_to(checkpoint) or set(row["files"]) != set(META_FILES):
            raise ValueError("invalid portable inference metadata path/files")
        for filename in row["files"]:
            # Explicit external statistics replace only this selected dataset's
            # stats.json. Its numerical bounds are checked by the policy loader.
            if filename == "stats.json" and stats_file is not None and name in configs:
                continue
            path = root / "meta" / filename
            if not path.resolve().is_relative_to(checkpoint) or not path.is_file():
                raise ValueError("portable inference metadata file is missing or outside checkpoint")
        if stats_file is None or name not in configs:
            validate_normalization_snapshot(json.loads((root / "meta/stats.json").read_text()), saved[name])
        if name in result:
            result[name]["dataset_path"] = str(root)
    if set(result) - found:
        raise ValueError("requested dataset is absent from portable checkpoint")
    return result


def finish_run(options, error=None):
    root = Path(options.output_ckpt_dir)
    path = root / RECORD
    if not path.exists():
        return
    record = json.loads(path.read_text())
    record.update(ended_at=datetime.now(timezone.utc).isoformat(), status="failed" if error else "finished",
                  error=type(error).__name__ if error else None)
    metrics = root / "training_metrics.jsonl"
    losses, steps = [], []
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            steps.append(row.get("step", row.get("global_step", 0)))
            if isinstance(row.get("loss"), (int, float)):
                losses.append(row["loss"])
    record.update(last_logged_step=max(steps, default=0), minimum_logged_loss=min(losses, default=None),
                  checkpoint_exports=sorted({p.name
                      for pattern in ("model-final", "checkpoint-*", "step-*")
                      for p in root.glob(pattern) if p.is_dir()}))
    summary = root / "training_summary.json"
    if "training_summary" not in record and summary.is_file():
        record["training_summary"] = json.loads(summary.read_text())
    write_json(path, record)
