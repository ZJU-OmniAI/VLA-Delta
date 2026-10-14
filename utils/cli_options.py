import argparse
import json
import math
import sys
from utils.optical_flow_config import OpticalFlowConfig, STAGES, resolve_stage


def _add_difference_query_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use_difference_query",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable learnable Difference Query conditioning for the action expert.",
    )
    parser.add_argument(
        "--num_difference_queries",
        type=int,
        default=None,
        help=(
            "Number of Difference Query tokens; requires --use_difference_query "
            "for random initialization."
        ),
    )
    parser.add_argument(
        "--vlm_attention_backend",
        choices=["eager", "flash_attention_2", "sdpa"],
        default=None,
        help="Optional Qwen attention backend override; Difference Query requires sdpa.",
    )


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="JSON/YAML options; CLI overrides configuration")
    parser.add_argument("--dataset_registry", help="Portable dataset registry YAML")
    parser.add_argument("--portable_run", action="store_true", help="Record resolved run metadata and exact data/RNG recovery state")
    parser.add_argument("--export_inference_metadata", action=argparse.BooleanOptionalAction, default=False,
                        help="Copy training meta files into portable checkpoints; default uses external metadata/statistics")
    parser.add_argument("--allow_aux_loss_weight_override", action="store_true", help="Explicit Stage2-to-Stage3 outer loss weight changes; never same-stage resume")
    parser.add_argument("--training_stage", choices=tuple(STAGES), default=None)
    parser.add_argument("--downstream_runtime_state", action="store_true",
                        help="Opt-in complete downstream RNG/cursor/checkpoint and optional W&B run recovery.")
    parser.add_argument("--pad_epoch_to_optimizer_batch", action="store_true",
                        help="Deterministically repeat the epoch prefix to complete every optimizer batch.")
    parser.add_argument("--downstream_probe_manifest", default=None,
                        help="Explicit fixed real-frame input manifest for isolated batch probes only.")
    parser.add_argument("--downstream_contract_version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--downstream_attempt", type=int, default=0)
    parser.add_argument("--component_optimizer_groups", action="store_true")
    parser.add_argument("--component_update_diagnostics", choices=("full", "inactive"), default=None,
                        help="Full per-group state deltas, or only inactive-group invariance checks; "
                             "unset preserves legacy diagnostics. Does not disable update protection or ordinary metrics.")
    parser.add_argument("--fast_resume_data_skip", action="store_true",
                        help="Skip historical batch indices before dataset reads on frozen three-stage datasets.")
    parser.add_argument("--save_and_exit_after_updates", type=int, default=None,
                        help="Save full state and exit at this successful global update, without shortening the scheduler.")
    parser.add_argument("--max_consecutive_skipped_windows", type=int, default=None)
    parser.add_argument("--slot_aux_type", default="none")
    from utils.slot_config import SlotConfig
    parser.add_argument("--slot_supervision_dir")
    parser.add_argument("--preparation_audit_cache", default=None)
    parser.add_argument("--aux_dataset_config", help="Versioned per-dataset auxiliary artifact paths")
    for name, value in SlotConfig().to_dict().items():
        if name == "slot_aux_type":
            continue
        parser.add_argument("--" + name, type=json.loads if isinstance(value, dict) else type(value), default=value)
    parser.add_argument("--init_from_checkpoint")
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--optical_flow_data_root")
    def optional_value(converter):
        return lambda value: None if value == "None" else converter(value)

    for name, field in OpticalFlowConfig.__dataclass_fields__.items():
        if name == "num_flow_queries":
            argument_type, argument_kwargs = optional_value(int), {}
        elif field.default is None:
            argument_type, argument_kwargs = optional_value(str), {}
        else:
            argument_type, argument_kwargs = type(field.default), {}
        parser.add_argument("--" + name, type=argument_type, default=field.default, **argument_kwargs)
    parser.add_argument(
        "--vlm_name_or_path", type=str, help="file path of pretrained VLM"
    )
    parser.add_argument(
        "--action_expert_name_or_path",
        type=str,
        help=(
            "path of the pretrained action expert. Unset means that we will use "
            "a randomly initialized action expert."
        ),
    )
    parser.add_argument(
        "--action_expert_config_path",
        type=str,
        help=(
            "Explicit JSON architecture source for a randomly initialized Action "
            "Expert, or a configuration to verify against loaded Expert weights."
        ),
    )
    parser.add_argument(
        "--checkpoint_load_purpose",
        choices=(
            "stage05_ar_resume",
            "stage05_ar_to_joint",
            "stage05_joint_resume",
            "downstream_finetune",
            "inference",
        ),
        default=None,
        help=(
            "Explicit checkpoint contract: Stage05 AR resume, AR-to-Joint, same-experiment "
            "Joint resume, or a new downstream fine-tune initialization."
        ),
    )
    parser.add_argument(
        "--FAST_tokenizer_path",
        type=str,
        help="file path of pretrained FAST action tokenizer",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="batch size per gpu device.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of micro-batches in one optimizer step.",
    )
    parser.add_argument(
        "--expected_global_batch_size",
        type=int,
        default=None,
        help=(
            "Optional runtime assertion for world_size * per-device batch * "
            "gradient accumulation steps."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--save_ckpt_interval", type=int, default=1)
    parser.add_argument("--save_step_interval", type=int, default=20000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Optional effective optimizer-step limit for training, scheduler, and progress.",
    )
    parser.add_argument(
        "--peak_learning_rate", type=float, default=1e-5, help="peak learning rate"
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=None,
        help="Optional warmup ratio; unset preserves the legacy 8 percent capped schedule.",
    )
    parser.add_argument(
        "--min_lr_rate",
        type=float,
        default=0.1,
        help="the minimal learning rate in the end of training (percent of peak LR)",
    )
    parser.add_argument("--enable_tensorboard", action=argparse.BooleanOptionalAction, default=False,
                        help="Write optional TensorBoard events; disabled by default")
    parser.add_argument(
        "--tensorboard_log_dir", type=str, default="./outputs/train/tensorboard"
    )
    parser.add_argument(
        "--output_ckpt_dir", type=str, default="./outputs/train"
    )
    parser.add_argument("--wandb_project", type=str, default="",
        help="W&B project; empty disables W&B, regardless of login/environment settings.")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_run_id", type=str, default="")
    parser.add_argument(
        "--wandb_resume", choices=["never", "must"], default="never"
    )
    parser.add_argument("--wandb_dir", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--wandb_tags", type=str, nargs="*")
    parser.add_argument(
        "--wandb_failure_policy",
        choices=["", "best_effort", "required"],
        default="best_effort",
        help="W&B failure handling; an empty value selects required when enabled, best_effort when disabled.",
    )
    parser.add_argument("--wandb_pending_capacity", type=int, default=256)
    parser.add_argument("--wandb_retry_base_steps", type=int, default=1)
    parser.add_argument("--wandb_retry_max_steps", type=int, default=128)
    parser.add_argument("--wandb_finish_max_attempts", type=int, default=2)
    parser.add_argument("--wandb_finish_timeout_seconds", type=float, default=15.0)
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log optimizer-step metrics at this interval.",
    )
    parser.add_argument(
        "--log_training_diagnostics",
        action="store_true",
        help=(
            "Log opt-in module gradient norms, throughput, memory, and data "
            "quality diagnostics without changing optimizer parameter groups."
        ),
    )
    parser.add_argument(
        "--tune_vlm", action="store_true", help="Whether to fine-tune the VLM"
    )
    parser.add_argument(
        "--batch_metric_reductions", action="store_true",
        help="Batch detached token-statistic and W&B scalar collectives without changing training objectives.",
    )
    parser.add_argument(
        "--tune_action_expert",
        action="store_true",
        help="Whether to fine-tune the projectors in the action expert",
    )
    parser.add_argument(
        "--detach_vlm_outputs_for_action_expert",
        action="store_true",
        help="whether to stop gradient from the action expert to VLM",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="vlm_and_action",
        help="support [vlm_and_action, vlm, action]; aux only with stage2_aux",
    )
    parser.add_argument(
        "--vlm_loss_weight",
        type=float,
        default=1.0,
        help=(
            "when setting loss type to vlm_and_action, we can control the weight "
            "of the VLM's loss"
        ),
    )
    parser.add_argument(
        "--action_expert_loss_weight",
        type=float,
        default=1.0,
        help=(
            "when setting loss type to vlm_and_action, we can control the weight "
            "of the action expert's loss"
        ),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="the type of the LR Scheduler. Avaliable: [cosine, constant]",
    )
    parser.add_argument(
        "--resume_training", action="store_true", help="whether to resume training"
    )
    parser.add_argument(
        "--allow_legacy_checkpoint_without_manifest",
        action="store_true",
        help=(
            "Explicitly allow a legacy checkpoint that predates resolved dataset "
            "manifests; dataset semantics cannot be verified in this mode."
        ),
    )
    parser.add_argument(
        "--allow_legacy_checkpoint_without_observation_contract",
        action="store_true",
        help=(
            "Explicitly accept a legacy checkpoint manifest that predates the "
            "versioned observation history contract."
        ),
    )
    parser.add_argument(
        "--save_optimizer_and_lr_states",
        action="store_true",
        help="Whether to save states of the optimizer and the LR scheduler",
    )
    _add_difference_query_options(parser)
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA to fine-tune the model",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="gate_proj, up_proj, down_proj",
        help="The names of the modules to apply the adapter to",
    )
    parser.add_argument(
        "--r", type=int, default=16, help="LoRA attention dimension (the `rank`)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="The alpha parameter for LoRA scaling. Typically setting to the double of `r`.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="The dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--dataset_entries",
        type=str,
        nargs="+",
        help=(
            "List of training dataset entries, e.g., bridge_orig_lerobot "
            "fractal20220817_data_lerobot libero_v21"
        ),
    )
    parser.add_argument(
        "--dataset_sample_ratios",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-entry sample-ratio overrides in dataset_entries order.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=1,
        help="size of the sliding window for historical image observations",
    )
    parser.add_argument(
        "--action_horizon", type=int, default=32, help="size of the action chunk"
    )
    parser.add_argument(
        "--max_pad_state_and_action_length",
        type=int,
        default=64,
        help="dim size of the max padded state and action",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1200,
        help="Maximum multimodal token length; 1200 preserves the legacy default.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=None,
        help=(
            "Optional DataLoader worker count. Unset keeps 24 for legacy datasets "
            "and uses 4 for episode-grouped datasets."
        ),
    )
    parser.add_argument("--prefetch_factor", type=int, default=3,
        help="DataLoader prefetch batches per worker; legacy default is 3.")
    return parser


def parse_train_options(args=None) -> argparse.Namespace:
    parser = build_train_parser()
    raw_args = list(sys.argv[1:] if args is None else args)
    from utils.runtime_config import config_arguments
    try:
        raw_args = config_arguments(parser, raw_args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    options = parser.parse_args(raw_args)
    options.wandb_project = options.wandb_project.strip()
    options.wandb_failure_policy = options.wandb_failure_policy or (
        "required" if options.wandb_project else "best_effort"
    )
    if options.wandb_project and not options.wandb_dir:
        from pathlib import Path
        options.wandb_dir = str(Path(options.output_ckpt_dir) / "wandb")
    if options.downstream_contract_version == 2 and not options.downstream_runtime_state:
        parser.error("downstream contract v2 requires downstream_runtime_state")
    if options.downstream_runtime_state:
        if (options.training_stage is not None or options.checkpoint_load_purpose != "downstream_finetune"
                or options.loss_type != "action" or not options.save_optimizer_and_lr_states
                or options.component_optimizer_groups or not options.pad_epoch_to_optimizer_batch
                or not options.max_train_steps):
            parser.error("downstream runtime requires action-only fresh/resume, full checkpoints and fixed batches")
        if options.wandb_project and options.wandb_failure_policy != "required":
            parser.error("downstream runtime requires wandb_failure_policy=required when W&B is enabled")
    elif options.downstream_probe_manifest and not (
            options.training_stage == "downstream_flow" and options.save_and_exit_after_updates):
        parser.error("downstream probes require downstream_runtime_state or bounded downstream_flow")
    explicit = {token[2:].split("=", 1)[0] for token in raw_args if token.startswith("--")}
    options.loss_type_explicit = "loss_type" in explicit
    options.optical_flow_explicit_fields = sorted(explicit & OpticalFlowConfig.__dataclass_fields__.keys())
    from utils.slot_config import SlotConfig, resolve_query_layout
    options.slot_explicit_fields = sorted(explicit & SlotConfig.__dataclass_fields__.keys())
    try:
        if options.init_from_checkpoint and (options.resume_from_checkpoint or options.resume_training):
            raise ValueError("init_from_checkpoint and resume are mutually exclusive")
        source = options.init_from_checkpoint or options.resume_from_checkpoint
        if source:
            if options.vlm_name_or_path and options.vlm_name_or_path != source:
                raise ValueError("checkpoint initialization path conflicts with vlm_name_or_path")
            options.vlm_name_or_path = source
        if options.resume_from_checkpoint:
            options.resume_training = True
            if options.action_expert_name_or_path and options.action_expert_name_or_path != source:
                raise ValueError("resume requires the same checkpoint for all model weights")
        flow = OpticalFlowConfig(**{name: getattr(options, name) for name in OpticalFlowConfig.__dataclass_fields__})
        from utils.slot_checkpoint import resolve_slot_checkpoint
        slot, _ = resolve_slot_checkpoint(source if options.training_stage else None,
            SlotConfig(**{name: getattr(options, name) for name in SlotConfig.__dataclass_fields__}),
            explicit_fields=options.slot_explicit_fields, resume=options.resume_training,
            stage=options.training_stage, initialize=bool(options.init_from_checkpoint),
            allow_loss_weight_override=options.allow_aux_loss_weight_override)
        for name, value in slot.to_dict().items():
            setattr(options, name, value)
        if options.training_stage is not None:
            from utils.optical_flow_checkpoint import resolve_flow_checkpoint
            flow, _ = resolve_flow_checkpoint(source, flow, explicit_fields=options.optical_flow_explicit_fields,
                                              stage=options.training_stage, resume=options.resume_training,
                                              initialize=bool(options.init_from_checkpoint),
                                              allow_loss_weight_override=options.allow_aux_loss_weight_override)
            for name, value in flow.to_dict().items():
                setattr(options, name, value)
        options.loss_type = resolve_stage(options.training_stage,
            options.loss_type if options.loss_type_explicit or options.training_stage is None else None,
            flow=flow, slot=slot)
        if options.training_stage == "downstream_flow":
            if (not source or options.checkpoint_load_purpose != "downstream_finetune"
                    or not options.tune_vlm or not options.tune_action_expert
                    or options.detach_vlm_outputs_for_action_expert or options.use_lora):
                raise ValueError("downstream_flow requires explicit downstream source and full connected fine-tuning")
            if options.action_expert_name_or_path != source:
                raise ValueError("downstream_flow requires the same source for VLM/Query/Flow/Expert")
            if "vlm_loss_weight" in explicit and options.vlm_loss_weight != 0:
                raise ValueError("downstream_flow forbids AR loss")
            options.vlm_loss_weight = 0.0
        if slot.enabled:
            if not options.resume_training and not options.init_from_checkpoint:
                raise ValueError("Slot stages require init_from_checkpoint or same-stage resume")
            if options.training_stage == "stage3_joint":
                options.tune_vlm = True
            from model.difference_query import resolve_difference_query_config
            query = resolve_difference_query_config(options.vlm_name_or_path, None,
                use_difference_query=options.use_difference_query, num_difference_queries=options.num_difference_queries,
                vlm_attention_backend=options.vlm_attention_backend)
            resolve_query_layout(query.num_difference_queries, flow.num_flow_queries, slot_enabled=True, query_enabled=query.enabled)
            if not options.slot_supervision_dir or options.window_size != 1:
                raise ValueError("Slot requires slot_supervision_dir and window_size=1")
        if options.training_stage in {"stage1_ar", "stage2_aux"}:
            if options.tune_action_expert or options.action_expert_name_or_path:
                raise ValueError("stage1/stage2 forbid Action Expert training or weight source")
            if "action_expert_loss_weight" in explicit and options.action_expert_loss_weight != 0:
                raise ValueError("stage1/stage2 forbid explicit FM loss weight")
            options.action_expert_loss_weight = 0.0
        if options.training_stage == "stage2_aux":
            if "vlm_loss_weight" in explicit and options.vlm_loss_weight != 0:
                raise ValueError("stage2_aux forbids explicit AR loss weight")
            options.vlm_loss_weight = 0.0
        if options.training_stage == "stage3_joint" and not options.tune_action_expert:
            raise ValueError("stage3_joint requires --tune_action_expert")
        if flow.enabled and (not options.optical_flow_data_root or options.window_size != 1):
            raise ValueError("OF training requires optical_flow_data_root and window_size=1")
    except (ValueError, NotImplementedError) as error:
        parser.error(str(error))
    if options.checkpoint_load_purpose == "stage05_ar_resume" and (
        not options.resume_training or options.loss_type != "vlm"
    ):
        parser.error("stage05_ar_resume requires --resume_training and --loss_type vlm")
    options.action_horizon_explicit = any(
        token == "--action_horizon" or token.startswith("--action_horizon=")
        for token in raw_args
    )
    if (
        options.checkpoint_load_purpose == "downstream_finetune"
        and not options.action_horizon_explicit
    ):
        parser.error(
            "--checkpoint_load_purpose downstream_finetune requires an explicit "
            "--action_horizon for fresh initialization or resume"
        )
    if options.max_train_steps is not None and options.max_train_steps < 1:
        parser.error("--max_train_steps must be at least 1")
    if options.per_device_train_batch_size < 1:
        parser.error("--per_device_train_batch_size must be positive")
    if options.gradient_accumulation_steps < 1:
        parser.error("--gradient_accumulation_steps must be positive")
    if (
        options.expected_global_batch_size is not None
        and options.expected_global_batch_size < 1
    ):
        parser.error("--expected_global_batch_size must be positive")
    if options.logging_steps < 1:
        parser.error("--logging_steps must be positive")
    for name in (
        "wandb_pending_capacity",
        "wandb_retry_base_steps",
        "wandb_retry_max_steps",
        "wandb_finish_max_attempts",
    ):
        if getattr(options, name) < 1:
            parser.error(f"--{name} must be positive")
    if options.wandb_retry_max_steps < options.wandb_retry_base_steps:
        parser.error("--wandb_retry_max_steps must be at least --wandb_retry_base_steps")
    if (
        not math.isfinite(options.wandb_finish_timeout_seconds)
        or options.wandb_finish_timeout_seconds <= 0
    ):
        parser.error("--wandb_finish_timeout_seconds must be finite and positive")
    if options.loss_type not in {"vlm", "action", "vlm_and_action", "aux"}:
        parser.error("--loss_type must be one of [vlm, action, vlm_and_action, aux (stage2_aux only)]")
    if options.loss_type == "vlm":
        if not options.tune_vlm:
            parser.error("--loss_type vlm requires --tune_vlm")
        if options.tune_action_expert:
            parser.error("--loss_type vlm forbids --tune_action_expert")
        if options.action_expert_name_or_path:
            parser.error(
                "--loss_type vlm forbids --action_expert_name_or_path"
            )
    for name in ("vlm_loss_weight", "action_expert_loss_weight"):
        value = getattr(options, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name} must be finite and non-negative")
    if options.loss_type in ("vlm", "vlm_and_action") and options.vlm_loss_weight == 0:
        parser.error("--vlm_loss_weight must be greater than zero for this --loss_type")
    if (
        options.loss_type in ("action", "vlm_and_action")
        and options.action_expert_loss_weight == 0
    ):
        parser.error(
            "--action_expert_loss_weight must be greater than zero for this --loss_type"
        )
    for name in ("adam_beta1", "adam_beta2"):
        value = getattr(options, name)
        if not math.isfinite(value) or not 0 <= value < 1:
            parser.error(f"--{name} must be finite and in [0, 1)")
    if not math.isfinite(options.adam_epsilon) or options.adam_epsilon <= 0:
        parser.error("--adam_epsilon must be finite and greater than zero")
    if options.warmup_ratio is not None and (
        not math.isfinite(options.warmup_ratio)
        or not 0 <= options.warmup_ratio <= 1
    ):
        parser.error("--warmup_ratio must be finite and in [0, 1]")
    if options.max_length < 1:
        parser.error("--max_length must be positive")
    if options.action_horizon < 1:
        parser.error("--action_horizon must be positive")
    if options.dataloader_num_workers is not None and options.dataloader_num_workers < 0:
        parser.error("--dataloader_num_workers must be non-negative")
    if options.prefetch_factor < 1:
        parser.error("--prefetch_factor must be positive")
    if options.dataset_sample_ratios is not None:
        if options.dataset_entries is None or len(options.dataset_sample_ratios) != len(
            options.dataset_entries
        ):
            parser.error(
                "--dataset_sample_ratios must have the same length as --dataset_entries"
            )
        if any(
            not math.isfinite(ratio) or ratio <= 0 or ratio > 1
            for ratio in options.dataset_sample_ratios
        ):
            parser.error("--dataset_sample_ratios values must be finite and in (0, 1]")
    try:
        from utils.training_limits import validate_update_limits
        validate_update_limits(options)
    except ValueError as error:
        parser.error(str(error))
    return options


def build_server_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="JSON/YAML options; CLI overrides configuration")
    parser.add_argument("--dataset_registry", help="Portable dataset registry YAML")
    parser.add_argument(
        "--dataset_entry",
        type=str,
        default="libero_v21",
        help="the pre-registration dataset entry in dataset2feature.yaml",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt_dir", type=str, help="checkpoint directory")
    parser.add_argument(
        "--stats_file", type=str, default=None,
        help="External LeRobot state/action stats.json for inference; must match checkpoint normalization.",
    )
    parser.add_argument(
        "--stats_key",
        type=str,
        default=None,
        help="Required explicit per-dataset normalization key for Stage05 mixed checkpoints.",
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        help="specify the inference mode, in `direct_action` or `subtask_then_action`",
    )
    parser.add_argument("--window_size", type=int, default=1, help="window size")
    parser.add_argument(
        "--num_denoised_steps",
        type=int,
        default=5,
        help="number of denoised steps",
    )
    parser.add_argument(
        "--max_pad_state_and_action_length",
        type=int,
        default=64,
        help="max padding length",
    )
    parser.add_argument("--port", type=int, default=8000, help="server port")
    parser.add_argument(
        "--action_conditioning_norm", choices=["final_rmsnorm", "raw"], default="final_rmsnorm",
        help="Expert conditioning representation; raw explicitly supports official legacy ZR-0 checkpoints.",
    )
    parser.add_argument(
        "--allow_legacy_checkpoint_without_manifest",
        action="store_true",
        help=(
            "Explicitly allow a legacy checkpoint that predates resolved dataset "
            "manifests; dataset semantics cannot be verified in this mode."
        ),
    )
    parser.add_argument(
        "--allow_legacy_checkpoint_without_observation_contract",
        action="store_true",
        help=(
            "Explicitly accept a legacy checkpoint manifest that predates the "
            "versioned observation history contract."
        ),
    )
    _add_difference_query_options(parser)
    return parser


def parse_server_options(args=None) -> argparse.Namespace:
    parser = build_server_parser()
    from utils.runtime_config import config_arguments
    try:
        return parser.parse_args(config_arguments(parser, list(sys.argv[1:] if args is None else args)))
    except (ValueError, OSError) as error:
        parser.error(str(error))
