"""Complete Slot artifacts, fixed statistics and stable Query role boundaries."""

import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import numpy as np
from utils.checkpoint_metadata import checkpoint_metadata

from utils.slot_config import SlotConfig, resolve_query_layout
from utils.slot_labels import VOCABULARIES

CONFIG_NAME = "slot_aux_config.json"
WEIGHTS_NAME = "slot_head.safetensors"
STATS_NAME = "slot_supervision_stats.json"


def _slot_runtime_config(metadata):
    if "slot_runtime_config" not in metadata:
        return None
    value = metadata["slot_runtime_config"]
    if not isinstance(value, dict) or set(value) != set(SlotConfig.__dataclass_fields__):
        raise ValueError("incomplete Slot runtime config fields")
    config = SlotConfig(**value).validate()
    if config.slot_aux_type != metadata.get("slot_aux_type", "none"):
        raise ValueError("Slot runtime/declaration conflict")
    return config


def validate_slot_stats(stats, config):
    if stats.get("version") == 2:
        from utils.slot_routing import combined_slot_stats
        for identity, item in stats["datasets"].items():
            if item.get("dataset_identity") != identity:
                raise ValueError("Slot statistics dataset identity mismatch")
            validate_slot_stats(item, config)
        if stats != combined_slot_stats(stats["datasets"]):
            raise ValueError("Slot aggregate training statistics mismatch")
        return stats
    from utils.slot_supervision import class_weights
    if stats.get("version") != 1 or stats.get("schema_version") != config.slot_schema_version or stats.get("split") != "train":
        raise ValueError("invalid Slot statistics schema/split")
    for q, vocab in VOCABULARIES.items():
        data = stats["classes"][q]
        if data["vocabulary"] != vocab or len(data["counts"]) != len(vocab):
            raise ValueError(f"Slot vocabulary/count mismatch: {q}")
        if any(type(n) is not int or n < 0 for n in data["counts"]):
            raise ValueError("invalid Slot class frequencies")
        expected = class_weights(data["counts"], config.slot_class_max_weight_ratio)
        if not np.allclose(expected, data["weights"], rtol=1e-12, atol=1e-12):
            raise ValueError("Slot class weights do not match fixed generation rule")
    for key in ("q5_mean", "q5_std"):
        values = np.asarray(stats[key])
        if values.shape != (3, 3) or not np.isfinite(values).all():
            raise ValueError(f"invalid {key}")
    if (np.asarray(stats["q5_std"]) < config.slot_q5_std_floor).any():
        raise ValueError("Q5 std is below configured floor")
    rules = stats["rules"]
    if (rules["max_weight_ratio"] != config.slot_class_max_weight_ratio or rules["std_floor_m"] != config.slot_q5_std_floor
            or rules["time_alignment"] != "source_semantic_anchor_only" or rules["camera"] not in {"first_view", "observation.images.exterior_1_left"} or rules["risk_range"] != [0, 1]):
        raise ValueError("Slot statistics generation rules conflict")
    return stats


def read_slot_artifacts(directory):
    root = Path(directory)
    paths = [root / n for n in (CONFIG_NAME, WEIGHTS_NAME, STATS_NAME)]
    meta_path = root / "zr0_checkpoint_metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    runtime = _slot_runtime_config(metadata)
    if not any(p.exists() for p in paths):
        if metadata.get("slot_aux_type", "none") != "none":
            raise ValueError("checkpoint declares Slot but artifacts are missing")
        return None
    if not all(p.is_file() for p in paths):
        raise ValueError("incomplete Slot config/weights/statistics artifacts")
    payload = checkpoint_metadata(json.loads(paths[0].read_text()))
    if payload.get("version") != 1:
        raise ValueError("Slot config version mismatch")
    if not isinstance(payload.get("config"), dict) or set(payload["config"]) != set(SlotConfig.__dataclass_fields__):
        raise ValueError("incomplete Slot config fields")
    config = SlotConfig(**payload["config"]).validate()
    if runtime is not None and runtime != config:
        raise ValueError("Slot runtime/artifact config conflict")
    if not config.enabled or (metadata and metadata.get("slot_aux_type") != config.slot_aux_type):
        raise ValueError("Slot declaration/config conflict")
    stats = checkpoint_metadata(json.loads(paths[2].read_text()))
    validate_slot_stats(stats, config)
    layout = payload["query_layout"]
    if resolve_query_layout(layout["num_difference_queries"], layout["num_flow_queries"], slot_enabled=True, query_enabled=True) != layout:
        raise ValueError("invalid Slot query groups")
    if metadata.get("query_role_layout", layout) != layout:
        raise ValueError("common and Slot query role layouts differ")
    from model.difference_query import _read_checkpoint_query_data
    query = _read_checkpoint_query_data(root)
    if query is None or not query.enabled or query.hidden_size != payload["hidden_size"] or query.num_difference_queries != layout["num_difference_queries"]:
        raise ValueError("Slot/Difference Query shape conflict")
    hidden = payload["hidden_size"]
    shapes = {}
    for q, width in {"Q1": 2, "Q2": 14, "Q3": 8, "Q4": 8, "Q5": 9, "Q6": 4, "Q9": 18}.items():
        shapes.update({f"heads.{q}.0.weight": [hidden], f"heads.{q}.0.bias": [hidden],
                       f"heads.{q}.1.weight": [width, hidden], f"heads.{q}.1.bias": [width]})
    shapes.update({"transition_norm.weight": [hidden], "transition_norm.bias": [hidden],
                   "q7.weight": [4, hidden], "q7.bias": [4], "q8.weight": [4, hidden], "q8.bias": [4]})
    with safe_open(str(paths[1]), framework="pt", device="cpu") as f:
        actual = {k: f.get_slice(k).get_shape() for k in f.keys()}
    if shapes != actual or shapes != payload["weight_shapes"]:
        raise ValueError("Slot weight shape mismatch")
    payload["statistics"] = stats
    return payload


def resolve_slot_checkpoint(directory, requested=None, *, explicit_fields=None, resume=False, stage=None, initialize=False, allow_loss_weight_override=False):
    payload = read_slot_artifacts(directory) if directory else None
    if payload is None:
        metadata_path = Path(directory) / "zr0_checkpoint_metadata.json" if directory else None
        metadata = json.loads(metadata_path.read_text()) if metadata_path and metadata_path.is_file() else {}
        saved = _slot_runtime_config(metadata)
        config = (requested or saved or SlotConfig()).validate()
        if resume and config.enabled:
            raise ValueError("Slot resume requires complete Slot checkpoint artifacts")
        if saved is not None and requested is not None:
            explicit = set(requested.to_dict()) if explicit_fields is None else set(explicit_fields)
            if resume and any(getattr(saved, key) != getattr(requested, key) for key in explicit):
                raise ValueError("explicit Slot config conflicts with checkpoint on resume")
            config = SlotConfig(**{**saved.to_dict(), **{key: getattr(requested, key) for key in explicit}}).validate()
        return config, None
    saved = SlotConfig(**payload["config"])
    if requested is None:
        return saved, payload
    explicit = set(requested.to_dict()) if explicit_fields is None else set(explicit_fields)
    metadata_path = Path(directory) / "zr0_checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    from utils.optical_flow_checkpoint import is_cross_stage_initialization
    removing = ("slot_aux_type" in explicit and not requested.enabled and
        is_cross_stage_initialization(metadata.get("training_stage"), stage, initialize=initialize, resume=resume))
    if removing and explicit_fields is None:
        # Config objects cannot distinguish omitted constructor defaults. An
        # explicit None type removes the head while inheriting those settings.
        defaults = SlotConfig()
        explicit = {key for key in explicit if getattr(requested, key) != getattr(defaults, key)} | {
            "slot_aux_type", "slot_loss_weight"}
    weight_override = (allow_loss_weight_override and initialize and not resume
        and metadata.get("training_stage") == "stage2_aux" and stage == "stage3_joint"
        and saved.enabled and requested.enabled)
    for key in explicit:
        if weight_override and key == "slot_loss_weight":
            continue
        if removing and key in {"slot_aux_type", "slot_loss_weight"}:
            continue
        if getattr(saved, key) != getattr(requested, key):
            raise ValueError(f"explicit Slot config conflicts with checkpoint: {key}")
    if removing:
        return SlotConfig(**{**saved.to_dict(), "slot_aux_type": "none",
            "slot_loss_weight": requested.slot_loss_weight if "slot_loss_weight" in explicit else 0.}).validate(), payload
    if weight_override:
        from dataclasses import replace
        saved = replace(saved, slot_loss_weight=requested.slot_loss_weight).validate()
    return saved, payload


def save_slot_artifacts(model, directory):
    from utils.optical_flow_checkpoint import reject_flow_zero3
    head = getattr(model, "slot_aux", None)
    if head is None:
        return
    reject_flow_zero3(model=model)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    stats = validate_slot_stats(checkpoint_metadata(model.slot_supervision_stats), model.slot_config)
    save_file(head.state_dict(), str(root / WEIGHTS_NAME))
    (root / STATS_NAME).write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    payload = {"version": 1, "config": model.slot_config.to_dict(), "query_layout": model.query_role_layout,
               "hidden_size": head.hidden_size, "weight_shapes": {k: list(v.shape) for k, v in head.state_dict().items()},
               "parameter_count": sum(p.numel() for p in head.parameters())}
    (root / CONFIG_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_slot_weights(head, directory, payload):
    if payload is not None:
        if head is None:
            raise ValueError("cannot discard saved Slot head during training")
        head.load_state_dict(load_file(str(Path(directory) / WEIGHTS_NAME)), strict=True)
