import os
import random
import uuid
import numpy as np

import torch
from utils.checkpoint_metadata import checkpoint_metadata


CHECKPOINT_TAG = "latest-model-optimizer-lr"
SCHEDULER_FILENAME = "scheduler.pt"


def model_export_tag(accelerator, *, final=False):
    """Choose a step-independent export name shared by every training rank."""
    if final:
        return "model-final"
    names = [f"checkpoint-{uuid.uuid4().hex}" if accelerator.is_main_process else None]
    if accelerator.num_processes > 1:
        torch.distributed.broadcast_object_list(names, src=0)
    return names[0]


def capture_rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def register_deepspeed_checkpoint_safe_globals() -> tuple[type, ...]:
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    from deepspeed.runtime.zero.config import ZeroStageEnum
    from deepspeed.utils.tensor_fragment import fragment_address

    safe_globals = (LossScaler, ZeroStageEnum, fragment_address)
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(list(safe_globals))
    return safe_globals


def _save_deepspeed_checkpoint(model, *args, **kwargs):
    """Save large checkpoints at non-ASCII paths in the usual torch format."""
    from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine

    class BufferedTorchCheckpointEngine(TorchCheckpointEngine):
        def save(self, state_dict, path):
            # PyTorch 2.6 uses unbuffered FileIO for non-ASCII filenames. Large
            # tensor writes can be short writes (>2 GiB on Linux). BufferedWriter
            # completes them, preserving the ordinary torch.save/load format.
            with open(path, "wb") as stream:
                torch.save(state_dict, stream)

    original_engine = model.checkpoint_engine
    missing = object()
    original_script_hook = vars(model).get("_copy_recovery_script", missing)
    suppress_script = callable(getattr(model, "_copy_recovery_script", None))
    try:
        if type(original_engine) is TorchCheckpointEngine:
            model.checkpoint_engine = BufferedTorchCheckpointEngine()
        if suppress_script:
            # We already export safetensors; keep DeepSpeed state without its
            # extra zero_to_fp32.py copy. Do not modify the installed package.
            model._copy_recovery_script = lambda save_path: None
        return model.save_checkpoint(*args, **kwargs)
    finally:
        model.checkpoint_engine = original_engine
        if suppress_script:
            if original_script_hook is missing:
                del model._copy_recovery_script
            else:
                model._copy_recovery_script = original_script_hook


def checkpoint_model_optimizer_scheduler(
    model,
    output_ckpt_dir,
    global_completed_steps,
    lr_scheduler,
    accelerator,
    *,
    _atomic=True,
) -> None:
    checkpoint_state = {"last_global_step": int(global_completed_steps)}
    unwrapped_model = accelerator.unwrap_model(model)
    downstream = getattr(unwrapped_model, "downstream_runtime", None)
    if downstream is not None and _atomic:
        return downstream.save_checkpoint(model, lr_scheduler, global_completed_steps,
            lambda pending: checkpoint_model_optimizer_scheduler(model, str(pending),
                global_completed_steps, lr_scheduler, accelerator, _atomic=False))
    from utils.optical_flow_checkpoint import reject_flow_zero3
    reject_flow_zero3(model=unwrapped_model, accelerator=accelerator)
    if getattr(unwrapped_model, "training_stage", None) or downstream is not None or getattr(unwrapped_model, "portable_run", False):
        checkpoint_state["training_data_cursor"] = unwrapped_model.training_data_cursor
    accelerator.print("==> saving model and optimizer <==")
    _save_deepspeed_checkpoint(
        model,
        output_ckpt_dir,
        tag=CHECKPOINT_TAG,
        client_state=checkpoint_state,
        save_latest=False,
    )

    checkpoint_directory = os.path.join(output_ckpt_dir, CHECKPOINT_TAG)
    if getattr(unwrapped_model, "training_stage", None) or downstream is not None or getattr(unwrapped_model, "portable_run", False):
        runtime = {"rng": capture_rng_state(), "cursor": unwrapped_model.training_data_cursor,
                   "global_step": int(global_completed_steps),
                   "world_size": accelerator.num_processes,
                   "sampler_contract": checkpoint_metadata(unwrapped_model.training_sampler_contract),
                   "skipped_flow_batches": getattr(unwrapped_model, "skipped_flow_batches", 0),
                   "scaler": accelerator.scaler.state_dict() if accelerator.scaler is not None else None}
        if hasattr(unwrapped_model, "training_scheduler_ticks_per_update"):
            runtime["scheduler_ticks_per_update"] = unwrapped_model.training_scheduler_ticks_per_update
        if downstream is not None:
            runtime.update(downstream.runtime_payload(model))
        torch.save(runtime, os.path.join(checkpoint_directory, f"training_runtime_rank{accelerator.process_index}.pt"))
    accelerator.print("==> saving lr scheduler <==")
    accelerator.save(
        lr_scheduler.state_dict(),
        os.path.join(checkpoint_directory, SCHEDULER_FILENAME),
    )

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(checkpoint_directory)


def _load_scheduler_state(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resume_model_optimizer_scheduler(
    model,
    old_output_ckpt_dir,
    lr_scheduler,
) -> int:
    register_deepspeed_checkpoint_safe_globals()
    load_path, checkpoint_state = model.load_checkpoint(
        old_output_ckpt_dir,
        tag=CHECKPOINT_TAG,
    )
    if load_path is None or checkpoint_state is None:
        raise RuntimeError(
            f"failed to load DeepSpeed checkpoint {CHECKPOINT_TAG!r} "
            f"from {old_output_ckpt_dir!r}"
        )

    scheduler_path = os.path.join(
        old_output_ckpt_dir,
        CHECKPOINT_TAG,
        SCHEDULER_FILENAME,
    )
    lr_scheduler.load_state_dict(_load_scheduler_state(scheduler_path))
    return int(checkpoint_state["last_global_step"])


def restore_stage_runtime(directory, model, accelerator, global_step):
    path = os.path.join(directory, f"training_runtime_rank{accelerator.process_index}.pt")
    runtime = _load_scheduler_state(path)
    if runtime["world_size"] != accelerator.num_processes or runtime["global_step"] != global_step:
        raise ValueError("resume runtime world size/global step mismatch")
    if checkpoint_metadata(runtime["sampler_contract"]) != checkpoint_metadata(model.training_sampler_contract):
        raise ValueError("resume sampler seed/batch/accumulation contract mismatch")
    if runtime["scaler"] is not None:
        if accelerator.scaler is None:
            raise ValueError("resume requires saved AMP scaler")
        accelerator.scaler.load_state_dict(runtime["scaler"])
    model.skipped_flow_batches = runtime["skipped_flow_batches"]
    model.training_data_cursor = runtime["cursor"]
    model.pending_resume_rng = runtime["rng"]
    downstream = getattr(model, "downstream_runtime", None)
    if downstream is not None:
        downstream.restore_payload(runtime)
    return runtime["cursor"]["epoch"], runtime["cursor"]["batch_idx"]
