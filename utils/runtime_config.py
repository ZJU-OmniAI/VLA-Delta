"""Portable JSON/YAML configuration shared by training, preparation and serving."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = {
    "dataset_registry", "vlm_name_or_path", "action_expert_name_or_path",
    "action_expert_config_path", "init_from_checkpoint", "resume_from_checkpoint",
    "output_ckpt_dir", "tensorboard_log_dir", "wandb_dir", "FAST_tokenizer_path",
    "slot_supervision_dir", "optical_flow_data_root", "aux_dataset_config",
    "preparation_audit_cache",
    "output_dir", "ckpt_dir", "stats_file", "dataset_path", "ar_sidecar_path", "joint_sidecar_path", "sidecar_path",
    "frozen_stage_index", "flow_candidate_index", "annotation_root", "output_root", "flow_episode_map",
}


def expand(value, *, base: Path, key=""):
    if isinstance(value, dict):
        return {k: expand(v, base=base, key=k) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, base=base, key=key) for v in value]
    if not isinstance(value, str):
        return value
    value = os.path.expandvars(os.path.expanduser(value))
    if re.search(r"\$\{[A-Za-z_][A-Za-z_0-9]*\}|\$[A-Za-z_][A-Za-z_0-9]*", value):
        raise ValueError(f"unset environment variable in configuration field {key!r}")
    if key in PATH_KEYS and value:
        value = str((base / value).resolve())
    return value


def load_mapping(path, *, _parents=(), resolve=True):
    def read(current, parents):
        current = Path(current).expanduser().resolve()
        if current in parents:
            raise ValueError(f"configuration inheritance cycle: {current.name}")
        payload = yaml.safe_load(current.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"configuration must be a mapping: {current}")
        parent = payload.pop("extends", None)
        values = read(current.parent / parent, (*parents, current)) if parent else {}
        values.update({key: (value, current.parent) for key, value in payload.items()})
        return values
    return {key: expand(value, base=base, key=key) if resolve else value
            for key, (value, base) in read(path, _parents).items()}


def config_arguments(parser, args):
    """Convert config defaults to argparse tokens, retaining explicit-field checks.

    Only CLI-overridden config fields skip expansion, so a resume can replace an
    unavailable source path. Existing CLI-only invocations retain their semantics.
    """
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    selected, _ = probe.parse_known_args(args)
    if selected.config is None:
        return list(args)
    actions = {action.dest: action for action in parser._actions}
    provided = {actions_by_flag.dest for token in args
                if (actions_by_flag := parser._option_string_actions.get(token.split("=", 1)[0])) is not None}
    # Resolve inherited values relative to the file that declares them.
    def read(path, parents=()):
        path = Path(path).resolve()
        if path in parents:
            raise ValueError("configuration inheritance cycle")
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("configuration must be a mapping")
        parent = data.pop("extends", None)
        values = read(path.parent / parent, (*parents, path)) if parent else {}
        for key, value in data.items():
            if key not in actions or key == "config":
                raise ValueError(f"unknown configuration option: {key}")
            if key not in provided:
                values[key] = (value, path.parent)
        return values
    tokens = []
    for key, (value, base) in read(selected.config).items():
        if key in provided or value is None:
            continue
        value = expand(value, base=base, key=key)
        action = actions[key]
        flag = action.option_strings[0]
        if isinstance(action, getattr(argparse, "BooleanOptionalAction", ())):
            if type(value) is not bool:
                raise ValueError(f"{key} requires a boolean")
            tokens.append(flag if value else next(s for s in action.option_strings if s.startswith("--no-")))
        elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if type(value) is not bool:
                raise ValueError(f"{key} requires a boolean")
            if value != action.default:
                tokens.append(flag)
        else:
            tokens.append(flag)
            values = value if isinstance(value, list) else [value]
            tokens.extend(json.dumps(v) if isinstance(v, (dict, bool)) else str(v) for v in values)
    return tokens + list(args)


def resolve_pretraining_kind(entry):
    """Accept the public field and historical private registries consistently."""
    kind = entry.get("dataset_kind", entry.get("stage05_kind"))
    if "dataset_kind" in entry and "stage05_kind" in entry and entry["dataset_kind"] != entry["stage05_kind"]:
        raise ValueError("dataset_kind conflicts with legacy stage05_kind")
    if not isinstance(kind, str) or kind not in {"droid", "molmo", "rh20t"}:
        raise ValueError(f"dataset_kind must be droid, molmo or rh20t, got {kind!r}")
    return kind


def configure_datasets(path=None):
    from utils.constants import DATASET2FEATURE
    path = Path(path or ROOT / "dataset2feature.yaml").resolve()
    # Preserve object identity for modules with existing from-imports.
    DATASET2FEATURE.clear()
    DATASET2FEATURE.update(load_mapping(path))
    return DATASET2FEATURE
