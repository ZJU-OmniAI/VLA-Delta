"""Flow artifacts and explicit cross-stage initialization contracts."""

import hashlib
import json
import warnings
from pathlib import Path
import torch
from utils.checkpoint_metadata import checkpoint_metadata

from safetensors.torch import load_file, save_file

from utils.optical_flow_config import OpticalFlowConfig, resolve_stage, stage_description, stage_loss_metadata


CONFIG_NAME = "optical_flow_aux_config.json"
WEIGHTS_NAME = "optical_flow_aux.safetensors"
COMPUTED_FLAGS = ("ar_loss_computed", "slot_loss_computed", "flow_loss_computed", "fm_loss_computed")
STAGE_STATE_SCOPE = "current_stage_completed_optimizer_windows"
# Accepted only when reading old version-1 dense checkpoints; never used at runtime.
LEGACY_UNUSED_CONFIG_FIELDS = {
    "flow_vae_model_path", "flow_color_scale", "flow_label_source",
    "flow_sample_min_valid_fraction", "flow_latent_cache_dir", "flow_latent_cache_mode",
    "flow_latent_cache_manifest_sha256",
    "flow_v2_hidden_dim", "flow_v2_num_heads", "flow_v2_num_layers", "flow_v2_mlp_ratio",
    "flow_latent_shape",
}


def initial_stage_training_state(stage, config, slot_aux_type="none"):
    return {**stage_loss_metadata(stage, slot_enabled=slot_aux_type != "none"),
            **({"slot_aux_type": slot_aux_type} if slot_aux_type != "none" else {}),
            "stage_description": stage_description(stage, config, slot_aux_type),
            "stage_metadata_version": 2, "loss_computed_scope": STAGE_STATE_SCOPE,
            "legacy_history_unknown": False, "completed_optimizer_windows": 0}


def read_stage_training_state(payload, config):
    """Read training evidence; an empty update count means intentionally undisclosed."""
    slot_type = payload.get("slot_aux_type", "none")
    state = initial_stage_training_state(payload["training_stage"], config, slot_type)
    if (slot_type == "none" and payload["training_stage"] == "stage2_aux"
            and payload.get("stage_description") == "stage2 OF-only / Slot not implemented"):
        payload = {**payload, "stage_description": state["stage_description"]}
    if payload.get("stage_metadata_version") in (None, 1):
        warnings.warn("legacy checkpoint has no verified optimizer-update history; history is unknown",
                      RuntimeWarning, stacklevel=2)
        state.update({key: None for key in COMPUTED_FLAGS if key != "slot_loss_computed"})
        state["legacy_history_unknown"] = True
        # Only verified updates are counted; legacy_history_unknown marks this lower bound.
        state["completed_optimizer_windows"] = 0
        return state
    if payload.get("stage_metadata_version") != 2 or payload.get("loss_computed_scope") != STAGE_STATE_SCOPE:
        raise ValueError("invalid checkpoint stage metadata version/scope")
    if type(payload.get("legacy_history_unknown")) is not bool:
        raise ValueError("invalid checkpoint legacy_history_unknown")
    count = payload.get("completed_optimizer_windows")
    if count != "" and (type(count) is not int or count < 0):
        raise ValueError("invalid checkpoint completed_optimizer_windows")
    for key in ("training_stage", "stage_description", "provisional"):
        if payload.get(key) != state[key] or type(payload.get(key)) is not type(state[key]):
            raise ValueError(f"checkpoint stage metadata mismatch: {key}")
    for key in COMPUTED_FLAGS:
        if key not in payload or (type(payload[key]) is not bool
                                 and not (payload[key] is None and payload["legacy_history_unknown"])):
            raise ValueError(f"invalid checkpoint computed flag: {key}")
    if slot_type == "none" and payload["slot_loss_computed"] is not False:
        raise ValueError("Slot loss recorded while Slot is disabled")
    allowed = {"stage1_ar": {"ar_loss_computed"}, "stage2_aux": {"flow_loss_computed"},
               "downstream_flow": {"flow_loss_computed", "fm_loss_computed"},
               "stage3_joint": {"ar_loss_computed", "fm_loss_computed"} |
                               ({"flow_loss_computed"} if config.enabled else set())}[payload["training_stage"]]
    if slot_type != "none" and payload["training_stage"] in {"stage2_aux", "stage3_joint"}:
        allowed.add("slot_loss_computed")
    if any(payload[key] is True and key not in allowed for key in COMPUTED_FLAGS):
        raise ValueError("checkpoint computed flags conflict with training stage")
    return {key: payload[key] for key in state}


def checkpoint_stage_metadata(model):
    state = getattr(model, "stage_training_state", None)
    if state is None:
        state = initial_stage_training_state(model.training_stage, model.optical_flow_config,
                                            getattr(getattr(model, "slot_config", None), "slot_aux_type", "none"))
    if state.get("training_stage") != model.training_stage:
        raise ValueError("checkpoint training state belongs to a different stage")
    return read_stage_training_state(state, model.optical_flow_config)


def record_stage_training_window(model, metrics):
    """Called after an optimizer window, using flags already reduced across ranks."""
    if getattr(model, "training_stage", None) is None:
        return
    from utils.optimizer_step_loss import optimizer_update_applied
    if not optimizer_update_applied(metrics):
        return
    state = checkpoint_stage_metadata(model)
    if metrics.get("training_stage") != model.training_stage:
        raise ValueError("optimizer window stage does not match checkpoint training state")
    for key in COMPUTED_FLAGS:
        if type(metrics.get(key)) is not bool:
            raise ValueError(f"optimizer window is missing boolean {key}")
        if metrics[key]:
            state[key] = True
    # Keep an undisclosed historical total undisclosed; zero would invent history.
    # Optimizer/scheduler progress is restored separately from runtime state.
    if state["completed_optimizer_windows"] != "":
        state["completed_optimizer_windows"] += 1
    model.stage_training_state = read_stage_training_state(state, model.optical_flow_config)


def reject_flow_zero3(*, model=None, accelerator=None, config=None, flow_enabled=False):
    """Reject sharded exports until a collective full-parameter exporter exists."""
    head = getattr(model, "optical_flow_aux", None)
    slot_head = getattr(model, "slot_aux", None)
    if head is None:
        head = slot_head
    flow_enabled = flow_enabled or head is not None or getattr(getattr(model, "optical_flow_config", None), "enabled", False)
    if not flow_enabled:
        return
    if config is None and accelerator is not None:
        plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
        config = getattr(plugin, "deepspeed_config", None)
    zero = (config or {}).get("zero_optimization", {})
    stage = zero.get("stage", 0) if isinstance(zero, dict) else zero
    partitioned = head is not None and any(hasattr(p, "ds_id") for p in head.parameters())
    if str(stage) == "3" or partitioned:
        raise RuntimeError("Optical Flow does not support DeepSpeed ZeRO-3: full-parameter gather/export "
                           "is not implemented; refusing partitioned weights. Use ZeRO-2 or non-ZeRO.")
    if str(stage) not in {"0", "2"}:
        raise RuntimeError(f"Optical Flow does not support ZeRO stage {stage!r}; use ZeRO-2 or non-ZeRO.")


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def module_checksum(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def is_cross_stage_initialization(source_stage, stage, *, initialize, resume):
    from utils.optical_flow_config import STAGES
    return (initialize and not resume and source_stage in STAGES and stage in STAGES
            and source_stage != stage)


def read_flow_artifacts(directory):
    root = Path(directory)
    config_path, weight_path = root / CONFIG_NAME, root / WEIGHTS_NAME
    if not config_path.exists():
        if weight_path.exists():
            raise ValueError("OF weights exist without OF configuration")
        metadata_path = root / "zr0_checkpoint_metadata.json"
        if metadata_path.is_file() and json.loads(metadata_path.read_text()).get("training_stage") is not None:
            raise ValueError("staged checkpoint is missing its OF configuration sidecar")
        return None
    payload = checkpoint_metadata(json.loads(config_path.read_text()))
    if payload.get("version") != 1:
        raise ValueError("unsupported optical-flow checkpoint: only version-1 dense regression is supported")
    fields = set(OpticalFlowConfig.__dataclass_fields__)
    raw_config = payload.get("config")
    if (not isinstance(raw_config, dict) or not fields.issubset(raw_config)
            or set(raw_config) - fields - LEGACY_UNUSED_CONFIG_FIELDS):
        raise ValueError("incomplete or unknown OF config fields")
    config = OpticalFlowConfig(**{name: raw_config[name] for name in fields}).validate()
    payload["config"] = config.to_dict()
    from utils.slot_checkpoint import resolve_slot_checkpoint
    slot, _ = resolve_slot_checkpoint(root)
    if slot.slot_aux_type != payload.get("slot_aux_type", "none"):
        raise ValueError("OF sidecar declares Slot but artifacts are missing or conflicting")
    resolve_stage(payload["training_stage"], flow=config, slot=slot)
    stage_state = read_stage_training_state(payload, config)
    metadata_path = root / "zr0_checkpoint_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if "stage_metadata_version" in payload or "stage_metadata_version" in metadata:
            compared_keys = stage_state.keys() if payload.get("stage_metadata_version") == 2 else stage_state.keys() & payload.keys()
            if any(key not in metadata or type(metadata[key]) is not type(payload.get(key))
                   or metadata[key] != payload.get(key) for key in compared_keys):
                raise ValueError("ordinary checkpoint and OF sidecar stage metadata mismatch")
    payload.update(stage_state)
    if config.enabled:
        if not weight_path.is_file():
            raise ValueError("OF configuration exists without OF weights")
    elif weight_path.exists():
        raise ValueError("disabled OF checkpoint contains OF weights")
    from model.difference_query import _read_checkpoint_query_data
    query = _read_checkpoint_query_data(root)
    if config.enabled:
        if query is None or not query.enabled:
            raise ValueError("OF checkpoint requires complete enabled Difference Query artifacts")
        if config.num_flow_queries > query.num_difference_queries:
            raise ValueError("OF query count exceeds checkpoint Difference Query count")
        from safetensors import safe_open
        with safe_open(str(weight_path), framework="pt", device="cpu") as tensors:
            actual = {key: tensors.get_slice(key).get_shape() for key in tensors.keys()}
        hidden, input_dim = config.flow_head_hidden_dim, query.hidden_size
        expected = {"input_norm.weight": [input_dim], "input_norm.bias": [input_dim],
                    "input_projection.weight": [hidden, input_dim], "input_projection.bias": [hidden],
                    "grid": [1, config.flow_grid_size ** 2, hidden],
                    "refinement.0.weight": [hidden, hidden, 3, 3], "refinement.0.bias": [hidden],
                    "refinement.1.weight": [hidden], "refinement.1.bias": [hidden],
                    "output.weight": [2, hidden, 3, 3], "output.bias": [2]}
        shapes = expected
        for index in range(config.flow_head_num_layers):
            for key, shape in {"attention.in_proj_weight": [3 * hidden, hidden],
                "attention.in_proj_bias": [3 * hidden], "attention.out_proj.weight": [hidden, hidden],
                "attention.out_proj.bias": [hidden], "norm.weight": [hidden], "norm.bias": [hidden],
                "mlp.0.weight": [4 * hidden, hidden], "mlp.0.bias": [4 * hidden],
                "mlp.2.weight": [hidden, 4 * hidden], "mlp.2.bias": [hidden],
                "output_norm.weight": [hidden], "output_norm.bias": [hidden]}.items():
                shapes[f"layers.{index}.{key}"] = shape
        if actual != shapes:
            raise ValueError("OF weight shape / Difference Query hidden size mismatch")
    if query is not None and config.num_flow_queries is not None and metadata_path.is_file():
        from utils.slot_config import resolve_query_layout
        layout = resolve_query_layout(query.num_difference_queries, config.num_flow_queries,
            slot_enabled=slot.enabled, query_enabled=query.enabled)
        if metadata.get("query_role_layout", layout) != layout:
            raise ValueError("OF/common checkpoint query role layout mismatch")
    return payload


def resolve_flow_checkpoint(directory, requested=None, *, explicit_fields=None, stage=None, resume=False, initialize=False, allow_loss_weight_override=False):
    payload = read_flow_artifacts(directory) if directory else None
    if payload is None:
        if resume and stage is not None:
            raise ValueError("new-stage resume requires stage/OF checkpoint metadata")
        return (requested or OpticalFlowConfig()).validate(), None
    saved_payload = {name: payload["config"].get(name, getattr(OpticalFlowConfig(), name))
                     for name in OpticalFlowConfig.__dataclass_fields__}
    saved = OpticalFlowConfig(**saved_payload)
    if resume and payload["training_stage"] != stage:
        raise ValueError("resume_from_checkpoint requires the same training_stage")
    if requested is None:
        return saved, payload
    fields = set(requested.to_dict()) if explicit_fields is None else set(explicit_fields)
    cross_stage = is_cross_stage_initialization(
        payload["training_stage"], stage, initialize=initialize, resume=resume)
    replacing = (saved.enabled and requested.enabled and "optical_flow_aux_type" in fields
                 and requested.optical_flow_aux_type != saved.optical_flow_aux_type and cross_stage)
    removing = (saved.enabled and "optical_flow_aux_type" in fields and not requested.enabled and cross_stage)
    if removing and explicit_fields is None:
        defaults = OpticalFlowConfig()
        fields = {name for name in fields if getattr(requested, name) != getattr(defaults, name)} | {
            "optical_flow_aux_type", "optical_flow_loss_weight"}
    if saved.num_flow_queries is not None and "num_flow_queries" in fields and requested.num_flow_queries != saved.num_flow_queries:
        raise ValueError("explicit num_flow_queries conflicts with checkpoint role layout")
    weight_override = (allow_loss_weight_override and initialize and not resume
        and payload["training_stage"] == "stage2_aux" and stage == "stage3_joint"
        and saved.enabled and requested.enabled)
    for name in fields:
        if weight_override and name == "optical_flow_loss_weight":
            continue
        if removing and name in {"optical_flow_aux_type", "optical_flow_loss_weight"}:
            continue
        if not replacing and (saved.enabled or resume) and getattr(requested, name) != getattr(saved, name):
            raise ValueError(f"explicit OF config conflicts with checkpoint: {name}")
    merged = {**saved.to_dict(), **{name: getattr(requested, name) for name in fields}}
    if saved.num_flow_queries is not None:
        merged["num_flow_queries"] = saved.num_flow_queries
    if removing and "optical_flow_loss_weight" not in fields:
        merged["optical_flow_loss_weight"] = 0.
    if replacing:
        payload = {**payload, "replace_flow_head": True}
    return OpticalFlowConfig(**merged).validate(), payload


def save_flow_artifacts(model, directory):
    reject_flow_zero3(model=model)
    root = Path(directory)
    config = model.optical_flow_config
    stage_state = checkpoint_stage_metadata(model)
    weight_path = root / WEIGHTS_NAME
    if model.optical_flow_aux is not None:
        save_file(model.optical_flow_aux.state_dict(), str(weight_path))
    elif weight_path.exists():
        weight_path.unlink()
    payload = {"version": 1, **stage_state,
               "slot_aux_type": getattr(getattr(model, "slot_config", None), "slot_aux_type", "none"),
               "config": config.to_dict(),
               "initialization": getattr(model, "aux_initialization", {})}
    (root / CONFIG_NAME).write_text(json.dumps(checkpoint_metadata(payload), indent=2, sort_keys=True) + "\n")


def load_flow_weights(head, directory, payload):
    if payload and payload["config"]["optical_flow_aux_type"] != "none" and not payload.get("replace_flow_head"):
        if head is None:
            raise ValueError("cannot discard a saved OF head during training")
        head.load_state_dict(load_file(str(Path(directory) / WEIGHTS_NAME)), strict=True)
