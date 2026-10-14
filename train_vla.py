import sys

# Help is available without importing the model, dataset or CUDA stack.
if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    from utils.cli_options import build_train_parser
    build_train_parser().parse_args()

import os
import math
import time
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "lerobot"))
import torch
import torch.distributed as dist
from contextlib import nullcontext

from utils.load_training_dataset import (
    build_concat_streaming_dataset,
    create_dataloader_for_concat,
    resolve_dataloader_num_workers,
)
from model import FlowmatchingActionHeadConfig
from model import ZR0Model
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedType
from torch.optim import AdamW
from accelerate.utils import set_seed
from transformers import get_cosine_with_min_lr_schedule_with_warmup_lr_rate, get_constant_schedule_with_warmup
from utils.training_numerics import assert_all_finite
from utils.wandb_training_logger import WandbTrainingLogger
from utils.action_expert_config import load_action_expert_config, read_vlm_hidden_size
from utils.checkpoint_contract import (
    DOWNSTREAM_FINETUNE,
    INFERENCE,
    STAGE05_AR_RESUME,
    STAGE05_AR_TO_JOINT,
    STAGE05_JOINT_RESUME,
    _checkpoint_has_stage05_identity,
    validate_checkpoint_load_purpose_arguments,
    validate_generic_action_expert_contract,
    validate_stage05_checkpoint_for_purpose,
    validate_stage05_resume_artifacts,
)
from utils.cli_options import parse_train_options
from utils.training_checkpoint import (
    checkpoint_model_optimizer_scheduler,
    model_export_tag,
    resume_model_optimizer_scheduler,
)
from utils.dataset_manifest import (
    resolved_manifest_json,
    validate_resume_manifest,
    write_resolved_dataset_manifest,
)
from utils.training_tokenization import (
    TOKENIZATION_METRIC_SCHEMA,
    token_metric_validity_key,
)
from utils.optimizer_step_loss import (
    OptimizerStepMetricAccumulator,
    global_supervision_counts,
    scaled_microbatch_loss,
)
from utils.dataset_seen_tracker import DatasetSeenTracker

def parse_option(args=None):
    return parse_train_options(args)


def should_skip_resumed_batch(
    *, epoch: int, batch_idx: int, resume_epoch: int, resume_batch_idx: int
) -> bool:
    return epoch == resume_epoch and batch_idx < resume_batch_idx


def resume_data_position(
    *, global_completed_steps: int, dataloader_length: int, gradient_accumulation_steps: int
):
    if global_completed_steps < 0:
        raise ValueError("global_completed_steps must be non-negative")
    if dataloader_length < 1 or gradient_accumulation_steps < 1:
        raise ValueError("dataloader_length and gradient_accumulation_steps must be positive")
    steps_per_epoch = math.ceil(dataloader_length / gradient_accumulation_steps)
    resume_epoch = global_completed_steps // steps_per_epoch
    completed_steps_in_epoch = global_completed_steps % steps_per_epoch
    resume_batch_idx = min(
        completed_steps_in_epoch * gradient_accumulation_steps, dataloader_length
    )
    return resume_epoch, resume_batch_idx


def synchronize_global_step(global_completed_steps: int, accelerator) -> int:
    if accelerator.num_processes == 1:
        return global_completed_steps
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "multi-process global-step synchronization requires an initialized "
            "torch.distributed process group"
        )
    step_tensor = torch.tensor(
        global_completed_steps,
        device=accelerator.device,
        dtype=torch.long,
    )
    dist.broadcast(step_tensor, src=0)
    return int(step_tensor.item())


def advance_global_step(global_completed_steps, metrics, accelerator):
    from utils.optimizer_step_loss import optimizer_update_applied
    if optimizer_update_applied(metrics) and accelerator.is_main_process:
        global_completed_steps += 1
    return synchronize_global_step(global_completed_steps, accelerator)


def get_trainable_parameters(model):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters were found for the optimizer")
    return parameters


PARAMETER_OWNERS = ("vlm", "difference_query", "action_expert", "optical_flow_aux", "slot_aux")


def trainable_parameter_owner(name: str) -> str:
    for owner in ("optical_flow_aux", "slot_aux"):
        if name.startswith(owner + "."):
            return owner
    if name == "backbone.difference_query.weight" or name.startswith(
        "backbone.difference_query."
    ):
        return "difference_query"
    if name.startswith("backbone."):
        return "vlm"
    if name.startswith("action_expert."):
        return "action_expert"
    raise ValueError(f"trainable parameter has no diagnostic owner: {name}")


def validate_trainable_parameter_ownership(model) -> dict[str, int]:
    counts = {owner: 0 for owner in PARAMETER_OWNERS[:3]}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f"trainable parameter is registered more than once: {name}")
        seen.add(id(parameter))
        owner = trainable_parameter_owner(name)
        counts[owner] = counts.get(owner, 0) + parameter.numel()
    if not seen:
        raise ValueError("No trainable parameters were found for diagnostics")
    return counts


def write_wandb_finish_diagnostics(metrics_log, writer, *, step: int, diagnostics: dict):
    record = {"event": "wandb_finish", "step": int(step), **diagnostics}
    if writer is not None:
        for name, value in diagnostics.items():
            writer.add_scalar(name.replace("_", "-"), value, step)
    if metrics_log is not None:
        metrics_log.write(
            json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
        )
        metrics_log.flush()
    return record


def initialization_summary(model, opt, ownership_counts, resolved_action_expert_config):
    """Record initialization choices without copying or hashing model tensors."""
    query = model.backbone.difference_query
    query_config = Path(opt.vlm_name_or_path) / "difference_query_config.json"
    return {
        "vlm_source": opt.vlm_name_or_path,
        "trainable_parameter_ownership": ownership_counts,
        "difference_query": {
            "enabled": query is not None,
            "source": "checkpoint" if query_config.is_file() else f"random_seed_{opt.seed}",
        },
        "action_expert": {
            "constructed": model.action_expert is not None,
            "source": opt.action_expert_name_or_path or f"random_seed_{opt.seed}",
            "config_path": str(resolved_action_expert_config.path),
            "source_action_horizon": resolved_action_expert_config.source_action_horizon,
            "resolved_action_horizon": resolved_action_expert_config.config.action_horizon,
            "action_horizon_overridden": resolved_action_expert_config.action_horizon_overridden,
        },
    }


def module_gradient_norms(model, accelerator) -> dict[str, torch.Tensor]:
    """Compute owner norms without changing optimizer parameter groups."""
    unwrapped = accelerator.unwrap_model(model)
    is_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED
    squared = {
        owner: torch.zeros((), dtype=torch.float64, device=accelerator.device)
        for owner in PARAMETER_OWNERS
    }
    parameter_counts = {owner: 0 for owner in PARAMETER_OWNERS}
    fragment_counts = {owner: 0 for owner in PARAMETER_OWNERS}
    parameter_owners = {}
    for name, parameter in unwrapped.named_parameters():
        if not parameter.requires_grad:
            continue
        owner = trainable_parameter_owner(name)
        parameter_counts[owner] += 1
        parameter_owners[id(parameter)] = owner

    if is_deepspeed:
        zero_optimizer = getattr(model, "optimizer", None)
        gradients_by_group = getattr(zero_optimizer, "averaged_gradients", None)
        parameters_by_group = getattr(zero_optimizer, "params_in_partition", None)
        if not isinstance(gradients_by_group, dict) or parameters_by_group is None:
            raise RuntimeError("DeepSpeed ZeRO-2 partition gradients are unavailable")
        for group_index, parameters in enumerate(parameters_by_group):
            gradients = gradients_by_group.get(group_index)
            if gradients is None:
                raise RuntimeError(
                    f"DeepSpeed ZeRO-2 gradient partition {group_index} is unavailable"
                )
            if len(gradients) < len(parameters):
                raise RuntimeError(
                    "DeepSpeed ZeRO-2 gradient/parameter partition lengths differ: "
                    f"group={group_index}, gradients={len(gradients)}, "
                    f"parameters={len(parameters)}"
                )
            for parameter, gradient in zip(parameters, gradients):
                owner = parameter_owners.get(id(parameter))
                if owner is None:
                    raise RuntimeError(
                        "DeepSpeed ZeRO-2 partition parameter has no ownership mapping"
                    )
                fragment_counts[owner] += 1
                squared[owner] += gradient.detach().to(torch.float32).square().sum().to(
                    torch.float64
                )
    else:
        for name, parameter in unwrapped.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            owner = trainable_parameter_owner(name)
            fragment_counts[owner] += 1
            squared[owner] += parameter.grad.detach().to(torch.float32).square().sum().to(
                torch.float64
            )

    metrics = {}
    for owner in PARAMETER_OWNERS:
        if parameter_counts[owner] == 0:
            continue
        if is_deepspeed:
            global_squared = accelerator.reduce(squared[owner], reduction="sum")
            global_fragments = accelerator.reduce(
                torch.tensor(
                    fragment_counts[owner],
                    dtype=torch.long,
                    device=accelerator.device,
                ),
                reduction="sum",
            )
            if global_fragments.item() == 0:
                raise RuntimeError(
                    f"DeepSpeed exposed no gradient fragments for {owner}"
                )
        else:
            global_squared = accelerator.reduce(squared[owner], reduction="mean")
        metrics[f"{owner}_grad_norm"] = global_squared.clamp_min(0).sqrt()
    return metrics


def build_adamw_optimizer(model, *, learning_rate, beta1, beta2, epsilon, component_groups=False):
    parameters = get_trainable_parameters(model)
    if component_groups:
        grouped = {owner: [] for owner in PARAMETER_OWNERS}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                grouped[trainable_parameter_owner(name)].append(parameter)
        parameters = [{"params": values, "component": owner} for owner, values in grouped.items() if values]
    return AdamW(
        parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=0.01,
    )


def effective_total_optimizer_steps(epoch_horizon: int, max_train_steps):
    if epoch_horizon < 1:
        raise ValueError("epoch_horizon must be positive")
    if max_train_steps is None:
        return epoch_horizon
    if max_train_steps < 1:
        raise ValueError("max_train_steps must be positive")
    return min(epoch_horizon, max_train_steps)


def calculate_warmup_steps(total_optimizer_steps: int, num_processes: int, warmup_ratio):
    if warmup_ratio is None:
        per_process_steps = min(20000, int(total_optimizer_steps * 0.08))
    else:
        per_process_steps = int(total_optimizer_steps * warmup_ratio)
    return per_process_steps * num_processes


def loss_output_metrics(outputs):
    metric_keys = (
        "loss",
        "total_loss",
        "vlm_loss",
        "action_expert_loss",
        "ar_loss",
        "flow_matching_loss",
        "weighted_vlm_loss",
        "weighted_action_expert_loss",
        "weighted_ar_loss",
        "weighted_flow_matching_loss",
        "vlm_loss_weight",
        "action_expert_loss_weight",
    )
    reference = next(
        (value for value in outputs.values() if isinstance(value, torch.Tensor)),
        None,
    )
    metrics = {}
    for key in metric_keys:
        if key not in outputs:
            continue
        value = outputs[key]
        metrics[key] = (
            value
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(
                value, device=reference.device if reference is not None else None
            )
        )
    return metrics


def batch_token_metrics(batch):
    metrics = {}
    for key, spec in TOKENIZATION_METRIC_SCHEMA.items():
        value = batch.get(key)
        validity = batch.get(token_metric_validity_key(key))
        if not isinstance(value, torch.Tensor) or not isinstance(validity, torch.Tensor):
            continue
        values = value.reshape(-1)
        valid = validity.to(device=value.device, dtype=torch.bool).reshape(-1)
        if values.numel() != valid.numel():
            raise ValueError(f"token metric {key!r} and validity mask shapes differ")
        selected = values[valid].float()
        if selected.numel() == 0:
            continue
        if spec.aggregation == "ratio":
            metrics[f"{key}_ratio"] = selected.mean()
        else:
            metrics[key] = selected.mean()
            metrics[f"{key}_min"] = selected.min()
            metrics[f"{key}_max"] = selected.max()
    return metrics


def iter_optimizer_step_windows(indexed_batches, gradient_accumulation_steps: int):
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    window = []
    for item in indexed_batches:
        window.append(item)
        if len(window) == gradient_accumulation_steps:
            yield window
            window = []
    if window:
        yield window


def run_optimizer_step_window(
    *,
    model,
    batches,
    accelerator,
    optimizer,
    lr_scheduler,
    training_progress: float,
    loss_type: str,
    vlm_loss_weight: float,
    action_expert_loss_weight: float,
    next_global_step: int,
    collect_training_diagnostics: bool = False,
    optical_flow_config=None,
    training_stage=None,
    slot_config=None,
    batch_metric_reductions: bool = False,
):
    batches = list(batches)
    if not batches:
        raise ValueError("optimizer-step accumulation window must not be empty")
    slot_enabled = slot_config is not None and slot_config.enabled
    if loss_type == "aux" and not slot_enabled and (optical_flow_config is None or not optical_flow_config.enabled):
        raise ValueError("aux optimizer windows require enabled Slot or Flow configuration")
    vlm_loss_weight = ZR0Model._validate_loss_weight(vlm_loss_weight, "vlm_loss_weight",
        required=loss_type in {"vlm", "vlm_and_action"})
    action_expert_loss_weight = ZR0Model._validate_loss_weight(action_expert_loss_weight, "action_expert_loss_weight",
        required=loss_type in {"action", "vlm_and_action"})
    if collect_training_diagnostics and accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)
    step_started = time.perf_counter()
    counts = global_supervision_counts(
        batches, loss_type=loss_type, accelerator=accelerator,
        optical_flow_config=optical_flow_config,
        slot_config=slot_config,
    )
    activity = counts.activity_metrics(loss_type=loss_type, vlm_loss_weight=vlm_loss_weight,
        action_expert_loss_weight=action_expert_loss_weight, optical_flow_config=optical_flow_config, slot_config=slot_config)
    bare_model = getattr(model, "module", model)
    group_guard = getattr(bare_model, "component_update_guard", None)
    if group_guard is not None:
        group_guard.begin(activity, detach_action=getattr(bare_model, "detach_vlm_outputs_for_action_expert", False),
            global_step=next_global_step - 1)
    if not activity["active_supervision_available"]:
        from utils.optical_flow_config import stage_loss_metadata
        from utils.optimizer_step_loss import optimizer_step_result
        optimizer.zero_grad()
        return {**optimizer_step_result(applied=False, reason="no_supervision"), **activity,
                **(group_guard.finish(applied=False) if group_guard else {}),
                **({"flow_eligible_samples": counts.flow_samples, "flow_coverage": counts.flow_samples} if counts.flow_samples is not None else {}),
                **({"slot_sample_coverage": counts.slot_covered_samples / counts.batch_samples.clamp_min(1),
                    "slot_available": bool(counts.slot_samples.sum() > 0),
                    **{f"slot_{q}_valid_count": counts.slot_samples[i] for i, q in enumerate(slot_config.slot_task_weights)},
                    **{f"slot_{q}_available": bool(counts.slot_samples[i] > 0) for i, q in enumerate(slot_config.slot_task_weights)}} if slot_enabled else {}),
                **stage_loss_metadata(training_stage or ("stage2_aux" if loss_type == "aux" else None), slot_enabled=slot_enabled)}
    from utils.optimizer_step_loss import OptimizerWindowStep
    step = OptimizerWindowStep(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, accelerator=accelerator)
    flow_sums = {}
    slot_sums = {}
    accumulator = OptimizerStepMetricAccumulator(
        loss_type=loss_type,
        vlm_loss_weight=vlm_loss_weight,
        action_expert_loss_weight=action_expert_loss_weight,
        collect_diagnostics=collect_training_diagnostics,
        device=accelerator.device,
        batch_metric_reductions=batch_metric_reductions,
    )
    optimizer.zero_grad()
    is_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED
    diagnostic_metrics = {}
    original_deepspeed_step = None
    deepspeed_diagnostic_calls = 0
    if collect_training_diagnostics and is_deepspeed:
        zero_optimizer = getattr(model, "optimizer", None)
        original_deepspeed_step = getattr(zero_optimizer, "step", None)
        if original_deepspeed_step is None:
            raise RuntimeError("DeepSpeed optimizer step is unavailable for diagnostics")

        def diagnostic_deepspeed_step(*args, **kwargs):
            nonlocal deepspeed_diagnostic_calls
            deepspeed_diagnostic_calls += 1
            diagnostic_metrics.update(module_gradient_norms(model, accelerator))
            return original_deepspeed_step(*args, **kwargs)

        zero_optimizer.step = diagnostic_deepspeed_step

    try:
        for microbatch_index, batch in enumerate(batches):
            is_boundary = microbatch_index == len(batches) - 1
            if is_deepspeed:
                if not hasattr(model, "set_gradient_accumulation_boundary"):
                    raise RuntimeError(
                        "DeepSpeed optimizer-step windows require the engine boundary API"
                    )
                model.set_gradient_accumulation_boundary(is_boundary)
            sync_context = (
                nullcontext()
                if is_deepspeed or is_boundary
                else accelerator.no_sync(model)
            )
            with sync_context:
                outputs = model(
                    batch,
                    training_progress,
                    vlm_loss_weight=vlm_loss_weight,
                    action_expert_loss_weight=action_expert_loss_weight,
                )
                backward_loss = scaled_microbatch_loss(
                    outputs,
                    counts=counts,
                    loss_type=loss_type,
                    vlm_loss_weight=vlm_loss_weight,
                    action_expert_loss_weight=action_expert_loss_weight,
                    gradient_accumulation_steps=accelerator.gradient_accumulation_steps,
                    data_parallel_world_size=accelerator.num_processes,
                    optical_flow_loss_weight=(optical_flow_config.optical_flow_loss_weight if optical_flow_config else 0.0),
                    slot_config=slot_config,
                )
                if slot_enabled:
                    for key, value in outputs.items():
                        if key.startswith("slot_") and isinstance(value, torch.Tensor) and key not in {"slot_loss_raw", "slot_loss_weighted"}:
                            value = value.detach().float().to(accelerator.device)
                            slot_sums[key] = slot_sums.get(key, torch.zeros_like(value)) + value
                if optical_flow_config is not None and optical_flow_config.enabled:
                    for key in ("optical_flow_loss_sum", "flow_epe_sum", "flow_motion_epe_sum", "flow_zero_epe_sum",
                                "flow_valid_fraction_sum", "flow_motion_fraction_sum", "flow_eligible_pixels_sum",
                                "flow_batch_samples"):
                        if key not in outputs:
                            continue
                        value = outputs[key].detach().to(device=accelerator.device, dtype=torch.float32)
                        flow_sums[key] = flow_sums.get(key, torch.zeros_like(value)) + value
                assert_all_finite(
                    accelerator, backward_loss, "loss", next_global_step
                )
                accumulator.update(outputs, batch)
                accelerator.backward(backward_loss)
    finally:
        if original_deepspeed_step is not None:
            zero_optimizer.step = original_deepspeed_step

    if collect_training_diagnostics and is_deepspeed:
        if deepspeed_diagnostic_calls != 1:
            raise RuntimeError(
                "DeepSpeed optimizer-step diagnostics must run exactly once, got "
                f"{deepspeed_diagnostic_calls}"
            )
    elif collect_training_diagnostics:
        diagnostic_metrics.update(module_gradient_norms(model, accelerator))
    step_result = step.finish()
    if group_guard is not None:
        diagnostic_metrics.update(group_guard.finish(applied=step_result["optimizer_update_applied"]))
    optimizer.zero_grad()
    metrics = accumulator.finalize(accelerator)
    if slot_enabled:
        from utils.slot_loss import finalize_slot_metrics
        reduced = {key: accelerator.reduce(value, reduction="sum") for key, value in slot_sums.items()}
        metrics.update(finalize_slot_metrics(reduced, counts.slot_samples, slot_config))
        metrics["loss"] = metrics.get("loss", 0) + metrics.get("slot_loss_weighted", 0)
        metrics["total_loss"] = metrics["loss"]
    if optical_flow_config is not None and optical_flow_config.enabled:
        reduced = {key: accelerator.reduce(value, reduction="sum") for key, value in flow_sums.items()}
        count = counts.flow_samples.clamp_min(1)
        for key, value in reduced.items():
            if key.endswith("_sum"):
                metrics[key.removesuffix("_sum")] = value / count
        if "flow_eligible_pixels_sum" in reduced:
            metrics["flow_eligible_pixels"] = reduced["flow_eligible_pixels_sum"]
        metrics["flow_eligible_samples"] = counts.flow_samples
        metrics["flow_coverage"] = counts.flow_samples / reduced["flow_batch_samples"].clamp_min(1)
        metrics["weighted_optical_flow_loss"] = metrics["optical_flow_loss"] * optical_flow_config.optical_flow_loss_weight
        metrics["optical_flow_loss_weight"] = torch.tensor(optical_flow_config.optical_flow_loss_weight, device=accelerator.device)
        metrics["loss"] = metrics.get("loss", 0) + metrics["weighted_optical_flow_loss"]
        metrics["total_loss"] = metrics["loss"]
    if collect_training_diagnostics:
        metrics["optimizer_microbatches_per_rank"] = torch.tensor(
            len(batches), dtype=torch.float64, device=accelerator.device
        )
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize(accelerator.device)
        local_elapsed = torch.tensor(
            time.perf_counter() - step_started,
            dtype=torch.float64,
            device=accelerator.device,
        )
        elapsed = accelerator.gather(local_elapsed.reshape(1)).max()
        local_samples = sum(
            int(batch["input_ids"].shape[0]) for batch in batches
        )
        global_samples = accelerator.reduce(
            torch.tensor(
                local_samples, dtype=torch.float64, device=accelerator.device
            ),
            reduction="sum",
        )
        diagnostic_metrics.update(
            optimizer_step_seconds=elapsed,
            optimizer_step_global_samples=global_samples,
            samples_per_second=global_samples / elapsed.clamp_min(1e-12),
        )
        if accelerator.device.type == "cuda":
            memory = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(accelerator.device),
                    torch.cuda.max_memory_reserved(accelerator.device),
                ],
                dtype=torch.float64,
                device=accelerator.device,
            )
            memory = accelerator.gather(memory.reshape(1, 2)).max(dim=0).values
            diagnostic_metrics.update(
                gpu_peak_memory_allocated_gib=memory[0] / (1024**3),
                gpu_peak_memory_reserved_gib=memory[1] / (1024**3),
            )
    metrics.update(diagnostic_metrics)
    metrics.update(activity)
    metrics.update(step_result)
    if "training_stage" in metrics:
        from utils.optical_flow_checkpoint import record_stage_training_window
        record_stage_training_window(accelerator.unwrap_model(model), metrics)
    return metrics


def tensorboard_loss_value(value) -> float:
    return float(value.detach().float()) if isinstance(value, torch.Tensor) else float(value)


def json_scalar_metrics(metrics: dict, *, batch_tensors: bool = False) -> dict:
    if not batch_tensors:
        return {
            name: value if isinstance(value, (str, bool)) else tensorboard_loss_value(value)
            for name, value in metrics.items()
        }
    result, device_tensors = {}, {}
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("only one element tensors can be converted to Python scalars")
            result[name] = None
            device_tensors.setdefault(value.device, []).append((name, value))
        else:
            result[name] = value if isinstance(value, (str, bool)) else tensorboard_loss_value(value)
    for entries in device_tensors.values():
        values = torch.stack([value.detach().float().reshape(()) for _, value in entries]).cpu().tolist()
        result.update(zip((name for name, _ in entries), values))
    return result


def tensorboard_loss_metric_name(output_name: str) -> str:
    return {
        "vlm_loss": "vlm-loss",
        "action_expert_loss": "action-expert-loss",
    }.get(output_name, output_name)

def write_tensorboard_training_metrics(writer, metrics, step, *, local_scalars=None):
    summary = None
    if local_scalars is not None:
        from tensorboard.compat.proto.summary_pb2 import Summary
        from torch.utils.tensorboard.summary import scalar
        summary = Summary()
    for name, value in metrics.items():
        if isinstance(value, str):
            writer.add_text('train-' + name, value, step)
            continue
        tag = 'train-{}'.format(tensorboard_loss_metric_name(name))
        value = tensorboard_loss_value(local_scalars[name] if local_scalars is not None else value)
        if summary is None:
            writer.add_scalar(tag, value, step)
        else:
            summary.value.extend(scalar(tag, value).value)
    if summary is not None and summary.value:
        writer.file_writer.add_summary(summary, step)


def get_absolute_path(path):
    if os.path.isabs(path):
        return path
    else:
        current_path = os.getcwd()
        return os.path.join(current_path, path)

def print_model_parameters(model):
    for name, p in model.named_parameters():
        print(f"{name}: {p.numel()}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"# all params: {total_params}")
    print(f"# trainable params: {trainable_params}")

def integrity_check(batch_data, processor):
    labels = batch_data.get("labels")
    if labels is None:
        for input_id, attn_mask in zip(
            batch_data["input_ids"][0], batch_data["attention_mask"][0]
        ):
            print(
                f"{input_id} -> {repr(processor.tokenizer.decode(input_id))} -> {attn_mask}"
            )
    else:
        for input_id, label, attn_mask in zip(
            batch_data["input_ids"][0], labels[0], batch_data["attention_mask"][0]
        ):
            print(
                f"{input_id} -> {repr(processor.tokenizer.decode(input_id))} -> {label} -> {attn_mask}"
            )
    print("integrity check ends.")

def save_model(accelerator: Accelerator, model, output_ckpt_dir, tag):
    from utils.optical_flow_checkpoint import reject_flow_zero3
    reject_flow_zero3(model=accelerator.unwrap_model(model), accelerator=accelerator)
    accelerator.print(f"save model at {tag}")
    # accelerator.wait_for_everyone()
    # save checkpoint
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(os.path.join(output_ckpt_dir, tag))
    # accelerator.wait_for_everyone()

def resolve_action_expert_config(opt, *, return_resolved=False):
    purpose = getattr(opt, "checkpoint_load_purpose", None)
    validate_checkpoint_load_purpose_arguments(
        purpose, resume_training=getattr(opt, "resume_training", False)
    )
    if purpose == INFERENCE:
        raise ValueError("checkpoint_load_purpose=inference is not a training mode")
    explicit_path = getattr(opt, "action_expert_config_path", None)
    if purpose is None and getattr(opt, "training_stage", None) is None:
        for candidate in (
            getattr(opt, "vlm_name_or_path", None),
            getattr(opt, "action_expert_name_or_path", None),
        ):
            if not candidate:
                continue
            candidate_path = Path(candidate)
            if candidate_path.is_dir() and _checkpoint_has_stage05_identity(candidate_path):
                raise ValueError(
                    "Stage05 checkpoint requires an explicit checkpoint_load_purpose; "
                    "booleans and legacy config loading cannot bypass its contract"
                )
            metadata_path = candidate_path / "zr0_checkpoint_metadata.json"
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception as error:
                    raise ValueError(
                        f"failed to read checkpoint metadata {metadata_path}: {error}"
                    ) from error
                if isinstance(metadata, dict) and "action_expert_contract" in metadata:
                    # Validate newer generic contracts before any legacy
                    # config lookup or optional horizon override.
                    validate_generic_action_expert_contract(candidate_path)
    checkpoint_path = None
    if opt.action_expert_name_or_path:
        checkpoint_path = os.path.join(
            opt.action_expert_name_or_path, "action_expert_config.json"
        )
    config_source = checkpoint_path or explicit_path
    if config_source is None:
        candidate = os.path.join(opt.vlm_name_or_path, "action_expert_config.json")
        if os.path.isfile(candidate):
            config_source = candidate
    if config_source is None:
        raise ValueError(
            "Action Expert configuration has no authoritative source; pass "
            "--action_expert_config_path or load a checkpoint containing it"
        )
    hidden_size = read_vlm_hidden_size(opt.vlm_name_or_path)
    if purpose == STAGE05_AR_RESUME:
        if opt.loss_type != "vlm" or opt.action_expert_name_or_path:
            raise ValueError("stage05_ar_resume requires loss_type=vlm and no Action Expert weights")
        validated = validate_stage05_checkpoint_for_purpose(
            opt.vlm_name_or_path, purpose=purpose, resume_training=opt.resume_training,
            external_config_path=explicit_path,
            requested_action_horizon=opt.action_horizon,
            expected_action_dim=opt.max_pad_state_and_action_length,
            expected_state_dim=opt.max_pad_state_and_action_length,
            expected_num_difference_queries=getattr(opt, "num_difference_queries", 32) or 32,
        )
        from model.difference_query import resolve_difference_query_config

        resolve_difference_query_config(
            opt.vlm_name_or_path, None,
            use_difference_query=opt.use_difference_query,
            num_difference_queries=opt.num_difference_queries,
            vlm_attention_backend=opt.vlm_attention_backend,
        )
        return validated if return_resolved else validated.config
    if purpose == STAGE05_AR_TO_JOINT:
        if explicit_path is None:
            raise ValueError(
                "stage05_ar_to_joint requires an AR checkpoint and explicit Expert config"
            )
        validated = validate_stage05_checkpoint_for_purpose(
            opt.vlm_name_or_path,
            purpose=purpose,
            external_config_path=explicit_path,
            requested_action_horizon=opt.action_horizon,
            expected_action_dim=opt.max_pad_state_and_action_length,
            expected_state_dim=opt.max_pad_state_and_action_length,
            expected_num_difference_queries=getattr(opt, "num_difference_queries", 32) or 32,
        )
        return validated if return_resolved else validated.config
    if purpose == STAGE05_JOINT_RESUME:
        if not checkpoint_path:
            raise ValueError("stage05_joint_resume requires a Joint checkpoint")
        validated = validate_stage05_resume_artifacts(
            opt.vlm_name_or_path,
            external_config_path=explicit_path,
            requested_action_horizon=opt.action_horizon,
            expected_action_dim=opt.max_pad_state_and_action_length,
            expected_state_dim=opt.max_pad_state_and_action_length,
            expected_num_difference_queries=getattr(opt, "num_difference_queries", 32) or 32,
        )
        return validated if return_resolved else validated.config
    if purpose == DOWNSTREAM_FINETUNE:
        if not opt.action_expert_name_or_path:
            raise ValueError(
                "downstream_finetune requires --action_expert_name_or_path so "
                "pretrained Action Expert weights are loaded"
            )
        source_checkpoint = opt.action_expert_name_or_path or opt.vlm_name_or_path
        validated = validate_stage05_checkpoint_for_purpose(
            source_checkpoint,
            purpose=purpose,
            external_config_path=explicit_path,
            requested_action_horizon=opt.action_horizon,
            resume_training=getattr(opt, "resume_training", False),
            expected_action_dim=opt.max_pad_state_and_action_length,
            expected_state_dim=opt.max_pad_state_and_action_length,
        )
        return validated if return_resolved else validated.config

    allow_horizon_override = (
        checkpoint_path is not None
        and not getattr(opt, "resume_training", False)
        and getattr(opt, "loss_type", None) in ("action", "vlm_and_action")
        and purpose is None
    )
    if getattr(opt, "portable_run", False) and not opt.resume_training:
        allow_horizon_override = True
    horizon_arguments = (
        {"action_horizon_override": opt.action_horizon}
        if allow_horizon_override
        else {"expected_action_horizon": opt.action_horizon}
    )
    resolved = load_action_expert_config(
        config_source,
        expected_action_dim=opt.max_pad_state_and_action_length,
        expected_state_dim=opt.max_pad_state_and_action_length,
        expected_vlm_hidden_size=hidden_size,
        **horizon_arguments,
    )
    if checkpoint_path and explicit_path:
        explicit = load_action_expert_config(
            explicit_path,
            expected_action_dim=opt.max_pad_state_and_action_length,
            expected_state_dim=opt.max_pad_state_and_action_length,
            expected_vlm_hidden_size=hidden_size,
            **horizon_arguments,
        )
        if explicit.payload != resolved.payload:
            raise ValueError(
                "explicit Action Expert config does not match the loaded checkpoint config"
            )
    return resolved if return_resolved else resolved.config


def train(opt):
    if getattr(opt, "dataset_registry", None):
        from utils.runtime_config import configure_datasets
        configure_datasets(opt.dataset_registry)
    from utils.training_limits import validate_update_limits, execution_limit, check_next_update, consecutive_skips
    validate_update_limits(opt)
    downstream_enabled = getattr(opt, "downstream_runtime_state", False)
    runtime = None
    set_seed(opt.seed)
    audit_cache = None
    if getattr(opt, "preparation_audit_cache", None):
        from utils.preparation_audit_cache import load_preparation_audit_cache
        audit_cache = load_preparation_audit_cache(opt.preparation_audit_cache)
        audit_cache.check_all()
    from utils.optical_flow_config import OpticalFlowConfig
    flow_config = OpticalFlowConfig(**{name: getattr(opt, name, field.default)
        for name, field in OpticalFlowConfig.__dataclass_fields__.items()})
    from utils.slot_config import SlotConfig
    for name, default in SlotConfig().to_dict().items():
        if not hasattr(opt, name):
            setattr(opt, name, default)
    slot_config = SlotConfig(**{name: getattr(opt, name) for name in SlotConfig.__dataclass_fields__})
    slot_reader = None
    if slot_config.enabled:
        from utils.slot_routing import load_slot_supervision
        slot_reader = load_slot_supervision(opt.slot_supervision_dir, audit_cache=audit_cache)
    if getattr(opt, "resume_from_checkpoint", None) and opt.training_stage in {"stage3_joint", "downstream_flow"}:
        opt.action_expert_name_or_path = opt.resume_from_checkpoint

    # Resolve and, for checkpoint-backed modes, validate the load contract before
    # constructing Accelerator, DeepSpeed engines, models, or allocating GPUs.
    resolved_action_expert_config = resolve_action_expert_config(
        opt, return_resolved=True
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(even_batches=False),
        step_scheduler_with_optimizer=not getattr(opt, "component_optimizer_groups", False),
    )
    accelerator.print(opt)

    total_batch_size = (
        opt.per_device_train_batch_size
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )
    if accelerator.gradient_accumulation_steps != opt.gradient_accumulation_steps:
        raise RuntimeError(
            "Accelerator gradient accumulation mismatch: "
            f"actual={accelerator.gradient_accumulation_steps}, "
            f"requested={opt.gradient_accumulation_steps}"
        )
    if (
        opt.expected_global_batch_size is not None
        and total_batch_size != opt.expected_global_batch_size
    ):
        raise RuntimeError(
            "global batch size mismatch: "
            f"actual={total_batch_size} = {accelerator.num_processes} x "
            f"{opt.per_device_train_batch_size} x "
            f"{accelerator.gradient_accumulation_steps}, "
            f"expected={opt.expected_global_batch_size}"
        )
    accelerator.print(
        "requested training scale: "
        f"world_size={accelerator.num_processes}, "
        f"micro_batch={opt.per_device_train_batch_size}, "
        f"gradient_accumulation_steps={accelerator.gradient_accumulation_steps}, "
        f"nominal_global_batch={total_batch_size}"
    )
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        deepspeed_config = accelerator.state.deepspeed_plugin.deepspeed_config
        from utils.optical_flow_checkpoint import reject_flow_zero3
        reject_flow_zero3(config=deepspeed_config, flow_enabled=flow_config.enabled or slot_config.enabled)
        runtime_batch_config = {
            "gradient_accumulation_steps": opt.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": opt.per_device_train_batch_size,
            "train_batch_size": total_batch_size,
        }
        configured_gas = deepspeed_config.get("gradient_accumulation_steps")
        if configured_gas != opt.gradient_accumulation_steps:
            raise RuntimeError(
                "DeepSpeed plugin gradient accumulation mismatch before prepare: "
                f"plugin={configured_gas}, requested={opt.gradient_accumulation_steps}"
            )
        deepspeed_config.update(runtime_batch_config)
        accelerator.print(
            "DeepSpeed batch config before prepare: "
            f"{runtime_batch_config}"
        )
    from utils.portable_run import create_tensorboard_writer
    writer = create_tensorboard_writer(opt, accelerator)

    concat_dataset = build_concat_streaming_dataset(
        dataset_entries=opt.dataset_entries,
        model_name_or_path=opt.vlm_name_or_path,
        fast_tokenizer_path=opt.FAST_tokenizer_path,
        window_size=opt.window_size,
        action_horizon=opt.action_horizon,
        accelerator=accelerator,
        process_mode="train",
        max_pad_state_and_action_length=opt.max_pad_state_and_action_length,
        loss_type=opt.loss_type,
        max_length=opt.max_length,
        dataset_sample_ratios=opt.dataset_sample_ratios,
        optical_flow_config=flow_config,
        optical_flow_data_root=getattr(opt, "optical_flow_data_root", None),
        slot_config=slot_config,
        slot_reader=slot_reader,
        aux_dataset_config=getattr(opt, "aux_dataset_config", None),
        downstream_runtime_state=downstream_enabled,
        downstream_contract_version=opt.downstream_contract_version,
        downstream_probe_manifest=getattr(opt, "downstream_probe_manifest", None),
    )
    resolved_dataset_manifest = concat_dataset.resolved_dataset_manifest
    if getattr(opt, "fast_resume_data_skip", False):
        from utils.load_training_dataset import validate_fast_resume_datasets
        validate_fast_resume_datasets(concat_dataset, downstream=(downstream_enabled or opt.training_stage == "downstream_flow"))
    seen_tracker = DatasetSeenTracker(
        concat_dataset,
        accelerator,
        resume_directory=opt.vlm_name_or_path if opt.resume_training else None,
        flow_config=flow_config,
    )
    if accelerator.is_main_process:
        accelerator.print("resolved dataset manifest:")
        accelerator.print(resolved_manifest_json(resolved_dataset_manifest))
        write_resolved_dataset_manifest(
            opt.output_ckpt_dir, resolved_dataset_manifest
        )
    if opt.resume_training:
        validate_resume_manifest(
            resolved_dataset_manifest,
            opt.vlm_name_or_path,
            allow_legacy_missing=opt.allow_legacy_checkpoint_without_manifest,
            allow_legacy_missing_observation_contract=(
                opt.allow_legacy_checkpoint_without_observation_contract
            ),
        )
    accelerator.print("len(concat_dataset):", len(concat_dataset))

    dataloader_num_workers = resolve_dataloader_num_workers(
        concat_dataset, opt.dataloader_num_workers
    )
    accelerator.print("dataloader workers:", dataloader_num_workers)
    dataloader = create_dataloader_for_concat(
        concat_dataset,
        batch_size_per_device=opt.per_device_train_batch_size,
        num_processes=accelerator.num_processes,
        num_workers=dataloader_num_workers,
        prefetch_factor=getattr(opt, "prefetch_factor", 3),
        seed=opt.seed,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        pad_epoch_to_optimizer_batch=getattr(opt, "pad_epoch_to_optimizer_batch", False),
    )
    epoch_sampler = dataloader.batch_sampler
    accelerator.wait_for_everyone()
    accelerator.print("len(dataloader):", len(dataloader))

    if opt.action_expert_name_or_path:
        accelerator.print(f"using a pre-trained action expert from {opt.action_expert_name_or_path}, thus we will load its configurations.")
    else:
        accelerator.print(
            "no Action Expert weights requested; reusing a saved config when "
            "available and allocating random weights only for action-capable modes."
        )
    action_expert_config = resolved_action_expert_config.config
    accelerator.print(
        "resolved Action Expert config: "
        f"path={resolved_action_expert_config.path}, "
        f"source_action_horizon={resolved_action_expert_config.source_action_horizon}, "
        f"resolved_action_horizon={action_expert_config.action_horizon}, "
        "action_horizon_overridden="
        f"{resolved_action_expert_config.action_horizon_overridden}, "
        f"source_sha256={resolved_action_expert_config.source_sha256}, "
        f"resolved_sha256={resolved_action_expert_config.parsed_sha256}"
    )

    if opt.use_lora:
        lora_args = {
            "use_lora": opt.use_lora,
            "target_modules": opt.target_modules,
            "r": opt.r,
            "lora_alpha": opt.lora_alpha,
            "lora_dropout": opt.lora_dropout
        }
    else:
        lora_args = None
    
    # initialize ZR-0 model
    model = ZR0Model(
        vlm_name_or_path = opt.vlm_name_or_path,
        action_expert_name_or_path = opt.action_expert_name_or_path,
        action_expert_config = action_expert_config,
        tune_vlm = opt.tune_vlm,
        tune_action_expert = opt.tune_action_expert,
        detach_vlm_outputs_for_action_expert = opt.detach_vlm_outputs_for_action_expert,
        lora_args = lora_args,
        use_difference_query = opt.use_difference_query,
        num_difference_queries = opt.num_difference_queries,
        vlm_attention_backend = opt.vlm_attention_backend,
        loss_type=None if getattr(opt, "training_stage", None) else opt.loss_type,
        training_stage=getattr(opt, "training_stage", None),
        slot_aux_type=getattr(opt, "slot_aux_type", "none"),
        slot_config=slot_config,
        slot_supervision_stats=slot_reader.stats if slot_reader else None,
        optical_flow_config=flow_config,
        init_from_checkpoint=getattr(opt, "init_from_checkpoint", None),
        resume_from_checkpoint=getattr(opt, "resume_from_checkpoint", None),
        action_expert_init_seed=opt.seed,
        allow_aux_loss_weight_override=getattr(opt, "allow_aux_loss_weight_override", False),
        checkpoint_load_purpose=getattr(opt, "checkpoint_load_purpose", None),
        resume_training=getattr(opt, "resume_training", False),
        action_expert_config_path=getattr(opt, "action_expert_config_path", None),
    )
    model.resolved_dataset_manifest = resolved_dataset_manifest
    if getattr(opt, "portable_run", False):
        from utils.portable_run import bind_run
        bind_run(model, opt, accelerator, concat_dataset)
    model.action_expert_source_action_horizon = int(
        resolved_action_expert_config.source_action_horizon
    )
    model.action_expert_config_source_bytes = (
        resolved_action_expert_config.path.read_bytes()
    )
    model.action_expert_config_source_sha256 = (
        resolved_action_expert_config.source_sha256
    )
    accelerator.wait_for_everyone()

    ownership_counts = validate_trainable_parameter_ownership(model)
    accelerator.print("trainable parameter ownership:", ownership_counts)
    if accelerator.is_main_process and getattr(opt, "portable_run", False):
        from utils.portable_run import update_run
        update_run(opt, initialization=initialization_summary(
            model, opt, ownership_counts, resolved_action_expert_config))

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters in model: {total_params}")

        total_backbone_params = sum(p.numel() for p in model.backbone.parameters())
        print(f"Total parameters in model.backbone: {total_backbone_params}")

        total_action_expert_params = sum(
            p.numel() for p in model.action_expert.parameters()
        ) if model.action_expert is not None else 0
        print(f"Total parameters in model.action_expert: {total_action_expert_params}")

        total_dit_params = sum(
            p.numel() for p in model.action_expert.dit.parameters()
        ) if model.action_expert is not None else 0
        print(f"Total parameters in model.action_expert.DiT: {total_dit_params}")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_ratio = trainable_params / total_params

        print(f"# Trainable parameters: {trainable_params}")
        print(f"Trainable ratio: {trainable_ratio:.2%}")

    accelerator.wait_for_everyone()
    '''single lr (start)'''
    optimizer = build_adamw_optimizer(
        model,
        learning_rate=opt.peak_learning_rate,
        beta1=opt.adam_beta1,
        beta2=opt.adam_beta2,
        epsilon=opt.adam_epsilon,
        component_groups=getattr(opt, "component_optimizer_groups", False),
    )
    if downstream_enabled:
        from utils.downstream_training_runtime import collective_check
        def verify_downstream_optimizer():
            if optimizer.state or len(optimizer.param_groups) != 1:
                raise ValueError("downstream optimizer must be newly constructed before prepare")
        collective_check(accelerator, verify_downstream_optimizer)
    '''single lr (end)'''
    
    epoch_optimizer_steps = math.ceil(opt.epochs * math.ceil(len(concat_dataset) / total_batch_size))
    if (getattr(opt, "portable_run", False) and opt.max_train_steps is not None
            and epoch_optimizer_steps < opt.max_train_steps):
        raise ValueError("epoch cap cannot reach max_train_steps; increase epochs or correct dataset selection")
    num_total_batches = effective_total_optimizer_steps(
        epoch_optimizer_steps, opt.max_train_steps
    )
    if downstream_enabled:
        num_total_batches = opt.max_train_steps
    warmup_steps = calculate_warmup_steps(
        num_total_batches, 1 if getattr(opt, "component_optimizer_groups", False) else accelerator.num_processes, opt.warmup_ratio
    )
    update_limit = execution_limit(opt, num_total_batches)

    wandb_logger = WandbTrainingLogger(
        accelerator,
        project=opt.wandb_project,
        run_name=opt.wandb_run_name,
        run_id=opt.wandb_run_id,
        resume=opt.wandb_resume,
        log_dir=opt.wandb_dir,
        group=opt.wandb_group,
        tags=opt.wandb_tags,
        failure_policy=opt.wandb_failure_policy,
        pending_capacity=opt.wandb_pending_capacity,
        retry_base_steps=opt.wandb_retry_base_steps,
        retry_max_steps=opt.wandb_retry_max_steps,
        finish_max_attempts=opt.wandb_finish_max_attempts,
        finish_timeout_seconds=opt.wandb_finish_timeout_seconds,
        batch_metric_reductions=getattr(opt, "batch_metric_reductions", False),
        resume_safe_steps=downstream_enabled or getattr(opt, "portable_run", False),
        resume_attempt=getattr(opt, "downstream_attempt", 0),
        config={
            **vars(opt),
            "global_batch_size": total_batch_size,
            "num_processes": accelerator.num_processes,
            "num_total_steps": num_total_batches,
            "warmup_steps": warmup_steps // (1 if getattr(opt, "component_optimizer_groups", False) else accelerator.num_processes),
        },
    )
    if (getattr(opt, "portable_run", False) and accelerator.is_main_process
            and wandb_logger.run is not None):
        run = wandb_logger.run
        with (Path(opt.output_ckpt_dir) / "wandb_identity.json").open("x") as stream:
            json.dump({"project": opt.wandb_project, "group": opt.wandb_group,
                "run_name": opt.wandb_run_name, "run_id": run.id, "url": run.url}, stream, indent=2)
    accelerator.wait_for_everyone()

    if opt.lr_scheduler == "cosine":
        # learning rate scheduler (linear warm up and cosine decay)
        cosine_factory = get_cosine_with_min_lr_schedule_with_warmup_lr_rate
        if downstream_enabled:
            from transformers import get_cosine_with_min_lr_schedule_with_warmup
            cosine_factory = get_cosine_with_min_lr_schedule_with_warmup
        lr_scheduler = cosine_factory(
            optimizer = optimizer,
            num_warmup_steps = warmup_steps,
            num_training_steps = num_total_batches * (1 if getattr(opt, "component_optimizer_groups", False) else accelerator.num_processes),
            min_lr_rate = opt.min_lr_rate
        )
    elif opt.lr_scheduler == "constant":
        # learning rate scheduler (linear warm up and keeping constant)
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, 
            num_warmup_steps = warmup_steps
        )
    else:
        raise ValueError("lr_scheduler should be in [cosine, constant].")

    accelerator.wait_for_everyone()
    optimizer, model, dataloader, lr_scheduler = accelerator.prepare(
        optimizer, model, dataloader, lr_scheduler
    )
    actual_gradient_accumulation_steps = accelerator.gradient_accumulation_steps
    actual_micro_batch = opt.per_device_train_batch_size
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        engine_gradient_accumulation_steps = int(
            model.gradient_accumulation_steps()
        )
        engine_micro_batch = int(model.train_micro_batch_size_per_gpu())
        if engine_gradient_accumulation_steps != opt.gradient_accumulation_steps:
            raise RuntimeError(
                "DeepSpeed gradient accumulation mismatch after prepare: "
                f"engine={engine_gradient_accumulation_steps}, "
                f"requested={opt.gradient_accumulation_steps}"
            )
        if engine_micro_batch != opt.per_device_train_batch_size:
            raise RuntimeError(
                "DeepSpeed micro-batch mismatch after prepare: "
                f"engine={engine_micro_batch}, "
                f"requested={opt.per_device_train_batch_size}"
            )
        actual_gradient_accumulation_steps = engine_gradient_accumulation_steps
        actual_micro_batch = engine_micro_batch
    actual_global_batch = (
        accelerator.num_processes
        * actual_micro_batch
        * actual_gradient_accumulation_steps
    )
    if actual_global_batch != total_batch_size:
        raise RuntimeError(
            "prepared global batch size changed: "
            f"actual={actual_global_batch}, requested={total_batch_size}"
        )
    accelerator.print(
        "prepared training scale: "
        f"world_size={accelerator.num_processes}, "
        f"micro_batch={actual_micro_batch}, "
        f"gradient_accumulation_steps={actual_gradient_accumulation_steps}, "
        f"nominal_global_batch={actual_global_batch}"
    )
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    metrics_log = None
    if accelerator.is_main_process:
        metrics_log = open(
            os.path.join(opt.output_ckpt_dir, "training_metrics.jsonl"),
            "a",
            encoding="utf-8",
            buffering=1,
        )

    global_completed_steps = 0
    # set model to the train() mode
    model.train()
    if getattr(opt, "training_stage", None) or getattr(opt, "portable_run", False):
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.training_sampler_contract = {"seed": opt.seed, "batch_size": opt.per_device_train_batch_size,
            "gradient_accumulation_steps": opt.gradient_accumulation_steps, "dataloader_length": len(dataloader)}
        unwrapped.training_data_cursor = {"epoch": 0, "batch_idx": 0}
        if opt.training_stage == "downstream_flow" or getattr(opt, "portable_run", False):
            unwrapped.validation_seen_tracker = seen_tracker
            unwrapped.training_scheduler_ticks_per_update = 1 if opt.component_optimizer_groups else accelerator.num_processes
    if downstream_enabled:
        from utils.downstream_training_runtime import DownstreamRuntime
        runtime = DownstreamRuntime(opt, model, accelerator, dataloader, concat_dataset, seen_tracker, wandb_logger)
    
    if opt.resume_training:
        old_output_ckpt_dir = os.path.dirname(opt.vlm_name_or_path)
        global_completed_steps = resume_model_optimizer_scheduler(
            model,
            old_output_ckpt_dir,
            lr_scheduler,
        )
        
        if not downstream_enabled:
            resume_epoch, resume_batch_idx = resume_data_position(
                global_completed_steps=global_completed_steps,
                dataloader_length=len(dataloader),
                gradient_accumulation_steps=accelerator.gradient_accumulation_steps,
            )
        if getattr(opt, "training_stage", None) or downstream_enabled or getattr(opt, "portable_run", False):
            from utils.training_checkpoint import restore_stage_runtime
            resume_epoch, resume_batch_idx = restore_stage_runtime(opt.vlm_name_or_path,
                accelerator.unwrap_model(model), accelerator, global_completed_steps)
            if runtime is not None:
                runtime.restore_engine_counters(model)

        accelerator.print("resume epoch:", resume_epoch)
        accelerator.print("resume batch index:", resume_batch_idx)
        accelerator.print("resume training from {} steps.".format(global_completed_steps))

        accelerator.print("resumed lr scheduler state dict:", lr_scheduler.state_dict())

    if getattr(opt, "component_optimizer_groups", False):
        from utils.component_updates import attach_component_guard
        attach_component_guard(model, optimizer, lr_scheduler, accelerator, diagnostics=opt.log_training_diagnostics,
            diagnostics_mode=getattr(opt, "component_update_diagnostics", None))
    accelerator.wait_for_everyone()
    st = time.time()
    reached_max_train_steps = global_completed_steps >= update_limit
    skipped_windows = 0
    if reached_max_train_steps:
        accelerator.print(
            f"Training is already complete at step {global_completed_steps}; "
            f"effective limit is {num_total_batches}."
        )
    import itertools
    epochs = (itertools.count(resume_epoch if opt.resume_training else 0)
              if downstream_enabled and opt.downstream_contract_version == 1 else range(opt.epochs))
    for epoch in epochs:
        if reached_max_train_steps:
            break
        # set the epoch into each dataset
        for ds in concat_dataset.datasets:
            ds.set_epoch(epoch)
        from utils.load_training_dataset import set_dataloader_epoch
        set_dataloader_epoch(dataloader, epoch_sampler, epoch)

        if opt.resume_training and resume_epoch > epoch:
            accelerator.print("skip {}-th epoch".format(epoch))
            accelerator.wait_for_everyone()
            continue

        accelerator.print(f"{epoch=}")

        epoch_dataloader, first_epoch_batch_idx = dataloader, 0
        if (opt.resume_training and epoch == resume_epoch and
                getattr(opt, "fast_resume_data_skip", False)):
            from utils.load_training_dataset import resume_dataloader_at_batch
            epoch_dataloader = resume_dataloader_at_batch(dataloader, epoch_sampler,
                epoch=epoch, batch_idx=resume_batch_idx)
            first_epoch_batch_idx = resume_batch_idx
            accelerator.print(f"resume data indices: epoch={epoch}, batch_idx={resume_batch_idx}; historical dataset reads=0")

        def resumed_batches():
            for batch_idx, batch in enumerate(epoch_dataloader, start=first_epoch_batch_idx):
                if opt.resume_training and should_skip_resumed_batch(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    resume_epoch=resume_epoch,
                    resume_batch_idx=resume_batch_idx,
                ):
                    accelerator.print("skip {}-th batch".format(batch_idx))
                    continue
                yield batch_idx, batch

        window_iterator = iter(iter_optimizer_step_windows(resumed_batches(), accelerator.gradient_accumulation_steps))
        for indexed_window in window_iterator:
            check_next_update(global_completed_steps, update_limit)
            first_batch_idx, first_batch = indexed_window[0]
            unwrapped = accelerator.unwrap_model(model) if getattr(opt, "training_stage", None) or downstream_enabled or getattr(opt, "portable_run", False) else None
            if getattr(unwrapped, "pending_resume_rng", None) is not None:
                from utils.training_checkpoint import restore_rng_state
                restore_rng_state(unwrapped.pending_resume_rng)
                unwrapped.pending_resume_rng = None
            if runtime is not None:
                runtime.before_window(model, lr_scheduler, indexed_window, global_completed_steps)
            seen_tracker.update([batch for _, batch in indexed_window])
            if accelerator.is_main_process and first_batch_idx == 0 and epoch == 0:
                integrity_check(first_batch, model.backbone.processor)
            training_progress = global_completed_steps / num_total_batches
            next_global_step = global_completed_steps + 1
            collect_training_diagnostics = opt.log_training_diagnostics and (
                next_global_step == 1
                or next_global_step % opt.logging_steps == 0
                or next_global_step >= num_total_batches
            )
            output_metrics = run_optimizer_step_window(
                model=model,
                batches=[batch for _, batch in indexed_window],
                accelerator=accelerator,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                training_progress=training_progress,
                loss_type=opt.loss_type,
                vlm_loss_weight=opt.vlm_loss_weight,
                action_expert_loss_weight=opt.action_expert_loss_weight,
                next_global_step=next_global_step,
                collect_training_diagnostics=collect_training_diagnostics,
                optical_flow_config=flow_config,
                training_stage=getattr(opt, "training_stage", None),
                slot_config=slot_config,
                batch_metric_reductions=getattr(opt, "batch_metric_reductions", False),
            )
            if unwrapped is not None:
                unwrapped.training_data_cursor = {"epoch": epoch, "batch_idx": indexed_window[-1][0] + 1}
            global_completed_steps = advance_global_step(global_completed_steps, output_metrics, accelerator)
            if runtime is not None:
                runtime.after_window(model, lr_scheduler, global_completed_steps, output_metrics)
                if runtime.stop_requested:
                    checkpoint_model_optimizer_scheduler(model, opt.output_ckpt_dir, global_completed_steps, lr_scheduler, accelerator)
                    reached_max_train_steps = True
                    break
            skipped_windows = consecutive_skips(skipped_windows, output_metrics["optimizer_update_applied"],
                getattr(opt, "max_consecutive_skipped_windows", None))
            if output_metrics["optimizer_update_skipped"]:
                reason = output_metrics["optimizer_skip_reason"]
                if unwrapped is not None and reason == "no_supervision":
                    unwrapped.skipped_flow_batches = getattr(unwrapped, "skipped_flow_batches", 0) + 1
                accelerator.print(f"optimizer update skipped: reason={reason}, global_step={global_completed_steps}")
                if flow_config.enabled:
                    output_metrics["flow_skipped_batches"] = torch.tensor(getattr(unwrapped, "skipped_flow_batches", 0), device=accelerator.device)
                wandb_logger.log(step=global_completed_steps,
                    mean_metrics={f"train/{key}": value for key, value in output_metrics.items()}, scalar_metrics={})
                if accelerator.is_main_process and metrics_log is not None:
                    metrics_log.write(json.dumps({"global_step": global_completed_steps,
                        **json_scalar_metrics(output_metrics, batch_tensors=getattr(opt, "batch_metric_reductions", False))}) + "\n")
                continue
            if flow_config.enabled:
                output_metrics["flow_skipped_batches"] = torch.tensor(getattr(unwrapped, "skipped_flow_batches", 0), device=accelerator.device)
            for metric_name, metric_value in output_metrics.items():
                if metric_name.endswith("grad_norm"):
                    assert_all_finite(
                        accelerator,
                        metric_value,
                        metric_name.replace("_", " "),
                        next_global_step,
                    )

            global_grad_norm = model.get_global_grad_norm()
            if global_grad_norm is not None:
                assert_all_finite(
                    accelerator,
                    global_grad_norm,
                    "global gradient norm",
                    next_global_step,
                )

            reached_max_train_steps = global_completed_steps >= update_limit

            do_save = global_completed_steps > 0 and (
                global_completed_steps % opt.save_step_interval == 0
                or reached_max_train_steps
            )
            if do_save and accelerator.is_main_process and getattr(opt, "portable_run", False):
                from utils.portable_run import record_training_summary
                record_training_summary(opt, completed=global_completed_steps, limit=update_limit,
                    epoch=epoch, scheduler_steps=num_total_batches)
            if do_save and runtime is not None:
                checkpoint_model_optimizer_scheduler(model, opt.output_ckpt_dir, global_completed_steps, lr_scheduler, accelerator)
            elif do_save:
                accelerator.wait_for_everyone()
                export_tag = model_export_tag(accelerator, final=reached_max_train_steps)
                save_model(accelerator, model, opt.output_ckpt_dir, export_tag)
                accelerator.wait_for_everyone()
                if opt.save_optimizer_and_lr_states:
                    checkpoint_model_optimizer_scheduler(model, opt.output_ckpt_dir, global_completed_steps, lr_scheduler, accelerator)
                if opt.save_optimizer_and_lr_states:
                    seen_tracker.save(
                        os.path.join(opt.output_ckpt_dir, "latest-model-optimizer-lr"),
                        epoch=epoch,
                        global_step=global_completed_steps,
                    )
                seen_tracker.save(
                    opt.output_ckpt_dir, epoch=epoch, global_step=global_completed_steps
                )
                accelerator.wait_for_everyone()

            do_log = (
                global_completed_steps == 1
                or global_completed_steps % opt.logging_steps == 0
                or reached_max_train_steps
            )
            if do_log:
                local_metrics = (json_scalar_metrics(output_metrics, batch_tensors=True)
                    if accelerator.is_main_process and getattr(opt, "batch_metric_reductions", False) else None)
                if accelerator.is_main_process and writer is not None:
                    writer.add_scalar('learning-rate', lr_scheduler.get_last_lr()[0], global_completed_steps)
                    if global_grad_norm is not None:
                        writer.add_scalar('grad-norm', global_grad_norm, global_completed_steps)
                    writer.add_scalar('training progress', training_progress, global_completed_steps)

                    write_tensorboard_training_metrics(writer, output_metrics, global_completed_steps,
                                                       local_scalars=local_metrics)

                mean_metrics = {
                    f"train/{name}": value for name, value in output_metrics.items()
                }
                scalar_metrics = {
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/progress": global_completed_steps / num_total_batches,
                    "train/epoch": epoch,
                }
                if global_grad_norm is not None:
                    scalar_metrics["train/grad_norm"] = float(global_grad_norm)

                wandb_diagnostics = wandb_logger.log(
                    step=global_completed_steps,
                    mean_metrics=mean_metrics,
                    scalar_metrics=scalar_metrics,
                )

                if accelerator.is_main_process:
                    local_record = {
                        "step": global_completed_steps,
                        "epoch": epoch,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        **(local_metrics if local_metrics is not None else json_scalar_metrics(output_metrics)),
                        **wandb_diagnostics,
                    }
                    if writer is not None:
                        for name, value in wandb_diagnostics.items():
                            writer.add_scalar(name.replace("_", "-"), value, global_completed_steps)
                    line = json.dumps(local_record, ensure_ascii=True, sort_keys=True)
                    print(f"optimizer_step_metrics={line}", flush=True)
                    metrics_log.write(line + "\n")

            if reached_max_train_steps:
                break

        if reached_max_train_steps:
            accelerator.wait_for_everyone()
            accelerator.print(
                f"Reached successful optimizer update limit: {update_limit} (scheduler length {num_total_batches})"
            )
            break
        
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            accelerator.print(f"Epoch {epoch} finished, saving checkpoint")

        do_epoch_save = (
            (epoch + 1) % opt.save_ckpt_interval == 0
            or epoch == opt.epochs - 1
        )

        if runtime is not None:
            # Fixed optimizer-step cadence; the real cursor handles epoch boundaries.
            do_epoch_save = False
            if opt.downstream_contract_version == 2 and epoch + 1 == opt.epochs:
                checkpoint_model_optimizer_scheduler(model, opt.output_ckpt_dir, global_completed_steps, lr_scheduler, accelerator)
                raise RuntimeError("epoch cap reached before successful optimizer update limit")
        if do_epoch_save:
            if accelerator.is_main_process and getattr(opt, "portable_run", False):
                from utils.portable_run import record_training_summary
                record_training_summary(opt, completed=global_completed_steps, limit=update_limit,
                    epoch=epoch, scheduler_steps=num_total_batches)
            accelerator.wait_for_everyone()
            export_tag = model_export_tag(accelerator, final=(
                epoch == opt.epochs - 1
                and (opt.max_train_steps is None or global_completed_steps >= update_limit)
            ))
            save_model(accelerator, model, opt.output_ckpt_dir, export_tag)
            accelerator.wait_for_everyone()
            if opt.save_optimizer_and_lr_states:
                checkpoint_model_optimizer_scheduler(model, opt.output_ckpt_dir, global_completed_steps, lr_scheduler, accelerator)
            if opt.save_optimizer_and_lr_states:
                seen_tracker.save(
                    os.path.join(opt.output_ckpt_dir, "latest-model-optimizer-lr"),
                    epoch=epoch,
                    global_step=global_completed_steps,
                )
            seen_tracker.save(
                opt.output_ckpt_dir, epoch=epoch, global_step=global_completed_steps
            )
            accelerator.wait_for_everyone()

    if getattr(opt, "portable_run", False):
        if accelerator.is_main_process:
            from utils.portable_run import record_training_summary
            record_training_summary(opt, completed=global_completed_steps, limit=update_limit,
                epoch=epoch, scheduler_steps=num_total_batches)
        if opt.max_train_steps is not None and global_completed_steps < update_limit:
            raise RuntimeError("epoch cap reached before successful optimizer update limit; final state saved")
    try:
        finish_diagnostics = wandb_logger.finish(exit_code=0)
        if accelerator.is_main_process:
            write_wandb_finish_diagnostics(
                metrics_log,
                writer,
                step=global_completed_steps,
                diagnostics=finish_diagnostics,
            )
    finally:
        if writer is not None:
            writer.close()
        if metrics_log is not None:
            metrics_log.close()

if __name__ == "__main__":
    opt = parse_option()
    portable = getattr(opt, "portable_run", False)
    main_rank = int(os.environ.get("RANK", "0")) == 0
    if portable:
        from utils.portable_run import configure_run, begin_run, finish_run
        configure_run(opt)
        if main_rank:
            begin_run(opt)
    failure = None
    try:
        train(opt)
    except BaseException as error:
        failure = error
        if opt.downstream_contract_version == 2:
            from utils.downstream_failure import failure_kind
            from utils.downstream_training_runtime import atomic_json
            rank = int(os.environ.get("RANK", "0"))
            try:
                atomic_json(Path(opt.output_ckpt_dir) / f"failure_attempt{opt.downstream_attempt:03d}_rank{rank}.json",
                            dict(kind=failure_kind(error), error=repr(error), rank=rank))
            except OSError:
                pass  # Preserve the original failure when storage itself failed.
        raise

    finally:
        if portable and main_rank:
            finish_run(opt, failure)
