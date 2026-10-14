"""Opt-in downstream state, immutable recovery checkpoints and runtime assertions.

No auxiliary objective is enabled here.  The legacy trainer is unchanged when
``downstream_runtime_state`` is absent.  Cursor units are prepared rank-local
micro-batches, not successful optimizer updates.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import uuid

import torch
import torch.distributed as dist
from utils.checkpoint_metadata import checkpoint_metadata, validate_normalization_snapshot

TAG = "latest-model-optimizer-lr"
VERSION = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    with pending.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def collective_check(accelerator, function):
    """Propagate a local validation error before any rank enters the next phase."""
    error, result = None, None
    try:
        result = function()
    except Exception as exc:
        error = repr(exc)
    if accelerator.num_processes > 1:
        errors = [None] * accelerator.num_processes
        dist.all_gather_object(errors, error)
    else:
        errors = [error]
    if any(errors):
        raise RuntimeError(f"downstream collective validation failed: {errors}")
    return result


def validate_complete_checkpoint(directory, *, identity=None):
    directory = Path(directory).resolve()
    receipt = json.loads((directory.parent / "checkpoint_complete.json").read_text())
    if receipt.get("version") not in (1, 2) or directory.name != TAG:
        raise ValueError("not a complete downstream recovery checkpoint")
    from utils.downstream_contracts import checkpoint_version
    if checkpoint_version(directory) != receipt["version"]:
        raise ValueError("checkpoint receipt/contract version mismatch")
    if identity is not None and receipt["experiment_id"] != identity["experiment_id"]:
        raise ValueError("recovery checkpoint belongs to a different experiment")
    required = {"scheduler.pt", "run_identity.json", "resolved_dataset_manifest.json",
                "dataset_artifacts/meta/stats.json", "dataset_artifacts/meta/info.json",
                "dataset_artifacts/vision_contract.json", "config.json", "tokenizer.json",
                "tokenizer_config.json", "mp_rank_00_model_states.pt", "data_seen_state.npz",
                "action_expert.safetensors", "action_expert_config.json", "difference_query_config.json",
                "difference_query.safetensors", "preprocessor_config.json"}
    if receipt["version"] == 2:
        required -= {"dataset_artifacts/meta/stats.json", "dataset_artifacts/meta/info.json", "dataset_artifacts/vision_contract.json"}
        required |= {"dataset_artifacts/index.json", "downstream_contract.json"}
        from utils.downstream_artifacts import validate_dataset_artifacts
        validate_dataset_artifacts(directory)
    elif identity is not None and identity.get("downstream_contract_version", 1) != 1:
        raise ValueError("legacy recovery cannot resume a v2 experiment")
    for rank in range(receipt["world_size"]):
        required.add(f"training_runtime_rank{rank}.pt")
        required.add(f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")
    # Older receipts always required W&B; new local-only runs explicitly opt out.
    if receipt.get("wandb_enabled", True):
        required.add("wandb_identity.json")
    if not required.issubset(receipt["files"]):
        raise ValueError("checkpoint receipt is missing required runtime/model/data artifacts")
    if "model.safetensors.index.json" in receipt["files"]:
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        if not set(index["weight_map"].values()).issubset(receipt["files"]):
            raise ValueError("checkpoint receipt is missing VLM weight shards")
    elif "model.safetensors" not in receipt["files"]:
        raise ValueError("checkpoint receipt is missing VLM weights")
    for name, record in receipt["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError(f"incomplete checkpoint file: {name}")
    saved = json.loads((directory / "run_identity.json").read_text())
    if identity is not None and checkpoint_metadata(saved) != checkpoint_metadata(identity):
        raise ValueError("recovery source/code/data identity changed")
    return receipt


def validate_model_artifacts(directory, *, live=True):
    from utils.checkpoint_contract import validate_stage05_checkpoint_for_purpose
    from utils.dataset_manifest import load_resolved_dataset_manifest
    root = Path(directory)
    resolved = validate_stage05_checkpoint_for_purpose(root, purpose="inference", requested_action_horizon=16)
    if resolved.config.action_horizon != 16:
        raise ValueError("saved Action Expert is not H16")
    manifest = load_resolved_dataset_manifest(root)
    entries = manifest["entries"]
    from utils.downstream_contracts import checkpoint_version
    if checkpoint_version(root) == 2:
        from utils.downstream_artifacts import validate_dataset_artifacts
        processor = json.loads((root / "preprocessor_config.json").read_text())
        if (processor.get("image_mean") != [.5]*3 or processor.get("image_std") != [.5]*3
                or not math.isclose(processor.get("rescale_factor", 0), 1/255)
                or processor.get("do_normalize") is not True or processor.get("do_rescale") is not True):
            raise ValueError("saved processor normalization changed")
        return dict(action_horizon=16, entries=validate_dataset_artifacts(root, live=live))
    if len(entries) != 1:
        raise ValueError("downstream checkpoint must contain exactly one dataset")
    entry = entries[0]
    info = json.loads((root / "dataset_artifacts/meta/info.json").read_text())
    cameras = [key for key, value in info["features"].items() if value["dtype"] == "video"]
    if entry["camera_keys"] != cameras or entry["action_horizon"] != 16:
        raise ValueError("saved camera/horizon contract changed")
    stats_file = root / "dataset_artifacts/meta/stats.json"
    validate_normalization_snapshot(json.loads(stats_file.read_text()), entry)
    for name in (("info.json", "stats.json") if live else ()):
        if (root / "dataset_artifacts/meta" / name).read_bytes() != (Path(entry["dataset_path"]) / "meta" / name).read_bytes():
            raise ValueError("live dataset differs from checkpoint snapshot")
    return dict(action_horizon=16, cameras=cameras, stats_file=str(stats_file))


class DownstreamRuntime:
    def __init__(self, options, model, accelerator, dataloader, dataset, tracker, wandb_logger):
        self.options, self.accelerator, self.tracker = options, accelerator, tracker
        self.contract_version = getattr(options, "downstream_contract_version", 1)
        self.root = Path(options.output_ckpt_dir)
        self.identity = json.loads((self.root / "run_identity.json").read_text())
        self.attempt = int(getattr(options, "downstream_attempt", 0))
        self.events = self.root / f"runtime_attempt{self.attempt:03d}_rank{accelerator.process_index}.jsonl"
        self.consumed_windows = self.skipped_windows = self.consecutive_skipped = 0
        self.stop_reason = None
        self.stop_requested = False
        self.last_window_end = time.monotonic()
        self.last_saved = None
        self.bare = accelerator.unwrap_model(model)
        self.bare.downstream_runtime = self
        self.bare.validation_seen_tracker = tracker
        self.bare.training_data_cursor = {"epoch": 0, "batch_idx": 0}
        expected = math.ceil(len(dataset) / options.expected_global_batch_size) * options.gradient_accumulation_steps
        self.bare.training_sampler_contract = dict(version=VERSION, seed=options.seed,
            batch_size=options.per_device_train_batch_size, gradient_accumulation_steps=options.gradient_accumulation_steps,
            world_size=accelerator.num_processes, dataloader_length=len(dataloader),
            dataset_length=len(dataset), padded_epoch_samples=expected * options.per_device_train_batch_size * accelerator.num_processes,
            pad_epoch_to_optimizer_batch=True, dataset_identity=self.identity["dataset_identity"],
            probe_sha256=sha256(options.downstream_probe_manifest) if options.downstream_probe_manifest else None)
        def check():
            if len(dataloader) != expected or len(dataloader) % options.gradient_accumulation_steps:
                raise ValueError(f"prepared loader length {len(dataloader)} != expected {expected}")
            if (model.train_batch_size() != options.expected_global_batch_size or model.lr_scheduler is not None):
                raise ValueError("unexpected DeepSpeed batch or scheduler ownership")
            if (model.train_micro_batch_size_per_gpu() != options.per_device_train_batch_size
                    or model.gradient_accumulation_steps() != options.gradient_accumulation_steps):
                raise ValueError("DeepSpeed micro-batch/accumulation differs from prepared loader")
            if not self.bare.backbone.model.is_gradient_checkpointing:
                raise ValueError("VLM gradient checkpointing must be enabled")
            epsilon = options.adam_epsilon if self.contract_version == 2 else 1e-6
            if self.contract_version == 2 and self.identity["optimizer"] != dict(betas=[options.adam_beta1, options.adam_beta2], epsilon=epsilon, weight_decay=.01):
                raise ValueError("optimizer differs from frozen run identity")
            for group in model.optimizer.optimizer.param_groups:
                if (group["betas"] != (.9, .95) or group["eps"] != epsilon or group["weight_decay"] != .01):
                    raise ValueError("prepared AdamW hyperparameters differ from fixed contract")
            if model.gradient_clipping() != 1.:
                raise ValueError("prepared gradient clipping differs from fixed contract")
            if options.resume_training:
                validate_complete_checkpoint(options.vlm_name_or_path, identity=self.identity)
            elif model.global_steps != 0:
                raise ValueError("fresh downstream engine retained pretraining steps")
        collective_check(accelerator, check)
        def persist_identity():
            if not accelerator.is_main_process:
                return
            run = wandb_logger.run
            if run is None and options.wandb_project:
                raise RuntimeError("online W&B run is required")
            if run is not None:
                identity = dict(project=run.project, entity=run.entity, group=options.wandb_group,
                    run_name=options.wandb_run_name, run_id=run.id, url=run.url)
                source = Path(options.vlm_name_or_path) / "wandb_identity.json"
                if options.resume_training and source.is_file():
                    saved = json.loads(source.read_text())
                    if identity != saved:
                        raise ValueError("W&B identity changed on resume")
                atomic_json(self.root / "wandb_identity.json", identity)
            atomic_json(self.root / "prepared_batch_contract.json", self.bare.training_sampler_contract)
        collective_check(accelerator, persist_identity)
        accelerator.wait_for_everyone()
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self._request_stop)

    def _request_stop(self, *_):
        self.stop_requested = True
        self.stop_reason = "signal"

    def runtime_payload(self, engine):
        identity = self.root / "wandb_identity.json"
        return dict(downstream_runtime_version=VERSION, experiment_id=self.identity["experiment_id"],
            consumed_windows=self.consumed_windows, skipped_windows=self.skipped_windows,
            consecutive_skipped=self.consecutive_skipped,
            backend_global_steps=engine.global_steps, backend_micro_steps=engine.micro_steps,
            backend_skipped_steps=engine.skipped_steps, scheduler_ticks_per_update=self.accelerator.num_processes,
            wandb=json.loads(identity.read_text()) if self.options.wandb_project and identity.is_file() else None)

    def restore_payload(self, runtime):
        if runtime.get("downstream_runtime_version") != VERSION or runtime.get("experiment_id") != self.identity["experiment_id"]:
            raise ValueError("missing or foreign downstream runtime state")
        self.consumed_windows, self.skipped_windows = runtime["consumed_windows"], runtime["skipped_windows"]
        self.consecutive_skipped = runtime.get("consecutive_skipped", 0)
        self.saved_runtime = runtime

    def restore_engine_counters(self, engine):
        runtime = self.saved_runtime
        if engine.global_steps != runtime["backend_global_steps"] or engine.skipped_steps != runtime["backend_skipped_steps"]:
            raise ValueError("DeepSpeed restored counters differ from runtime")
        # DeepSpeed 0.15.4 does not persist micro_steps in its model checkpoint.
        engine.micro_steps = runtime["backend_micro_steps"]

    def check_scheduler(self, scheduler, step):
        state = scheduler.state_dict()
        ticks = self.accelerator.num_processes
        if state["last_epoch"] != step * ticks or state["_step_count"] != step * ticks + 1:
            raise ValueError(f"scheduler did not advance by successful updates: {step}, {state['last_epoch']}")
        warmup = int(self.options.max_train_steps * self.options.warmup_ratio)
        factor = step / warmup if step < warmup else (
            self.options.min_lr_rate + (1 - self.options.min_lr_rate) *
            .5 * (1 + math.cos(math.pi * (step - warmup) / (self.options.max_train_steps - warmup))))
        expected = self.options.peak_learning_rate * factor
        for value in scheduler.get_last_lr():
            if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-14):
                raise ValueError(f"LR mismatch: step={step}, actual={value}, expected={expected}")

    def before_window(self, engine, scheduler, indexed_window, step):
        self.check_scheduler(scheduler, step)
        self.scheduler_before = copy.deepcopy(scheduler.state_dict())
        self.lr_used = list(scheduler.get_last_lr())
        if len(indexed_window) != self.options.gradient_accumulation_steps:
            raise ValueError("incomplete optimizer window")
        for _, batch in indexed_window:
            if batch["input_ids"].shape[0] != self.options.per_device_train_batch_size:
                raise ValueError("prepared micro-batch size differs from selected configuration")
            if self.contract_version == 2:
                from utils.downstream_contracts import validate_batch
                collective_check(self.accelerator, lambda: validate_batch(batch, self.bare.resolved_dataset_manifest["entries"]))
                continue
            if tuple(batch["action"].shape[1:]) != (16, 64) or bool(batch["action_mask"][..., 14:].any()):
                raise ValueError("H16/padded action mask contract changed")
            grid = batch["image_grid_thw"]
            if tuple(grid.shape) != (self.options.per_device_train_batch_size*3,3) or not bool(
                    (grid == torch.tensor([1,14,14], device=grid.device)).all()):
                raise ValueError("actual processor output is not three 224x224 views per sample")

    def after_window(self, engine, scheduler, step, output):
        self.consumed_windows += 1
        if self.contract_version == 2:
            request = self.root / "stop_requested.json"
            if request.exists() and json.loads(request.read_text()).get("attempt") == self.attempt:
                self.stop_requested, self.stop_reason = True, "launcher_signal"
        applied = output["optimizer_update_applied"]
        self.skipped_windows += int(not applied)
        self.consecutive_skipped = 0 if applied else self.consecutive_skipped + 1
        if self.contract_version == 2 and self.consecutive_skipped >= 5:
            self.stop_requested, self.stop_reason = True, "five_consecutive_skipped_windows"
        self.check_scheduler(scheduler, step)
        if not applied:
            # Preserve the optimizer-step guard using native scheduler state.
            from torch.utils._pytree import tree_flatten
            before, before_spec = tree_flatten(self.scheduler_before)
            after, after_spec = tree_flatten(scheduler.state_dict())
            equal = before_spec == after_spec and all(
                torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
                for a, b in zip(before, after))
            if not equal:
                raise ValueError("scheduler changed during a skipped window")
        now = time.monotonic()
        record = dict(step=step, consumed_windows=self.consumed_windows, applied=applied,
            skipped_windows=self.skipped_windows, consecutive_skipped=self.consecutive_skipped, stop_reason=self.stop_reason,
            cursor=self.bare.training_data_cursor, lr=scheduler.get_last_lr(), lr_used=self.lr_used,
            elapsed_window_seconds=now-self.last_window_end,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self.accelerator.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(self.accelerator.device),
            free_bytes=torch.cuda.mem_get_info(self.accelerator.device)[0])
        if self.contract_version == 2:
            grad = engine.get_global_grad_norm()
            if grad is not None:
                record["gradient_norm"] = float(grad)
                record["gradient_clipped"] = record["gradient_norm"] > 1.
            record["samples_per_second"] = self.options.expected_global_batch_size / max(now-self.last_window_end, 1e-9)
            for key, value in output.items():
                if key.startswith("gr1_"):
                    record[key] = float(value)
        if "total_loss" in output:
            value = output["total_loss"]
            record["total_loss"] = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        with self.events.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if self.contract_version == 2:
            for name in ("samples_per_second", "peak_allocated_bytes", "peak_reserved_bytes", "free_bytes"):
                output[name] = torch.tensor(record[name], device=self.accelerator.device)
            output["lr_used"] = torch.tensor(self.lr_used[0], device=self.accelerator.device)
        self.last_window_end = now
        stop = torch.tensor(int(self.stop_requested), device=self.accelerator.device)
        self.stop_requested = bool(self.accelerator.reduce(stop, reduction="sum").item())

    def save_checkpoint(self, engine, scheduler, step, write_native):
        accel = self.accelerator
        paths = [None]
        if accel.is_main_process:
            paths[0] = str(self.root / "recovery_checkpoints" / f".pending-{uuid.uuid4().hex}")
        if accel.num_processes > 1:
            dist.broadcast_object_list(paths, src=0)
        pending = Path(paths[0])
        pending.mkdir(parents=True, exist_ok=True)
        collective_check(accel, lambda: write_native(pending))
        accel.wait_for_everyone()
        checkpoint = pending / TAG
        cursor = self.bare.training_data_cursor
        self.tracker.save(checkpoint, epoch=cursor["epoch"], global_step=step)
        def publish():
            if not accel.is_main_process:
                return
            for name in ("run_identity.json", "wandb_identity.json"):
                if name == "wandb_identity.json" and (
                        not self.options.wandb_project or not (self.root / name).is_file()):
                    continue
                atomic_json(checkpoint / name, checkpoint_metadata(json.loads((self.root / name).read_text())))
            if self.contract_version == 2:
                from utils.downstream_artifacts import save_dataset_artifacts
                save_dataset_artifacts(checkpoint, self.bare.resolved_dataset_manifest)
            else:
                self._save_legacy_artifacts(checkpoint)
            validation = validate_model_artifacts(checkpoint)
            files = {str(p.relative_to(checkpoint)): dict(size=p.stat().st_size)
                     for p in sorted(checkpoint.rglob("*")) if p.is_file()}
            atomic_json(pending / "checkpoint_complete.json", dict(version=self.contract_version, step=step,
                consumed_windows=self.consumed_windows, world_size=accel.num_processes,
                experiment_id=self.identity["experiment_id"], files=files, validation=validation,
                wandb_enabled=(checkpoint / "wandb_identity.json").is_file()))
            validate_complete_checkpoint(checkpoint, identity=self.identity)
            if self.contract_version == 2:
                from utils.downstream_artifacts import sync_tree
                sync_tree(pending)
            destination = pending.parent / f"checkpoint-{uuid.uuid4().hex}"
            if destination.exists():
                raise FileExistsError(destination)
            pending.rename(destination)
            final = destination / TAG
            if self.contract_version == 2:
                from utils.downstream_artifacts import finish_publication
                finish_publication(self.root, final, step, self.consumed_windows, self.identity,
                    export=(step > 0 and step % self.options.save_step_interval == 0))
            else:
                atomic_json(self.root / "latest_complete.json", dict(checkpoint=str(final), step=step,
                    consumed_windows=self.consumed_windows, complete=True))
        collective_check(accel, publish)
        accel.wait_for_everyone()
        self.last_saved = json.loads((self.root / "latest_complete.json").read_text())["checkpoint"]
        self.last_window_end = time.monotonic()
        return self.last_saved

    def _save_legacy_artifacts(self, checkpoint):
        entry = self.bare.resolved_dataset_manifest["entries"][0]
        artifacts = checkpoint / "dataset_artifacts/meta"
        artifacts.mkdir(parents=True)
        for name in ("stats.json", "info.json"):
            shutil.copy2(Path(entry["dataset_path"]) / "meta" / name, artifacts / name)
        atomic_json(checkpoint / "dataset_artifacts/vision_contract.json", dict(
            cameras=entry["camera_keys"], width=224, height=224, interpolation="bicubic",
            crop=False, pad=False, letterbox=False, mean=[.5]*3, std=[.5]*3))
