"""Deterministic resolved dataset manifests and resume validation."""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from utils.checkpoint_metadata import checkpoint_metadata, saved_min_max_bounds

from utils.dataset_spec import (
    ResolvedDatasetSpec,
    resolve_objective_requirements,
)


RESOLVED_DATASET_MANIFEST_NAME = "resolved_dataset_manifest.json"
RESOLVED_DATASET_MANIFEST_VERSION = 4
DATASET_SEMANTIC_FIELDS = (
    "resolved_adapter",
    "camera_keys",
    "grounding_camera_keys",
    "task_key",
    "target_text_field",
    "state_key",
    "action_key",
    "state_dimension",
    "action_dimension",
    "action_horizon",
    "normalization",
    "stats_key",
    "state_q01",
    "state_q99",
    "action_q01",
    "action_q99",
    "state_min",
    "state_max",
    "action_min",
    "action_max",
    "observation_contract",
    "canonical_schema",
    "vision_input_contract",
    "episode_input_contract",
    "downstream_contract_version",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _stats_vector(spec: ResolvedDatasetSpec, key: str, statistic: str):
    if spec.normalization_stats is None:
        return None
    values = spec.normalization_stats.get(key, {}).get(statistic)
    return _jsonable(values) if values is not None else None


def _stats_identity(spec: ResolvedDatasetSpec) -> str | None:
    if spec.stats_sha256 is not None:
        return spec.stats_sha256
    if spec.normalization_stats is None:
        return None
    payload = json.dumps(
        _jsonable(spec.normalization_stats),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_spec_to_manifest(
    spec: ResolvedDatasetSpec,
    loss_type: str,
) -> dict:
    return {
        "dataset_entry": spec.dataset_entry,
        "dataset_path": spec.dataset_path,
        "dataset_type": spec.dataset_type,
        "resolved_adapter": spec.adapter,
        "task_key": spec.task_key,
        "target_text_field": spec.target_text_field,
        "loss_requirements": resolve_objective_requirements(
            loss_type,
            adapter=spec.adapter,
            target_text_field=spec.target_text_field,
            dataset_type=spec.dataset_type,
            dataset_entry=spec.dataset_entry,
        ).dependency_names(),
        "camera_keys": list(spec.camera_keys),
        "grounding_camera_keys": list(spec.grounding_camera_keys),
        "state_key": spec.state_key or None,
        "action_key": spec.action_key or None,
        "state_dimension": spec.state_dim,
        "action_dimension": spec.action_dim,
        "action_horizon": spec.action_horizon,
        "stats_file": spec.stats_path,
        "stats_key": spec.stats_key,
        "stats_sha256": _stats_identity(spec),
        "state_q01": _stats_vector(spec, "observation.state", "q01"),
        "state_q99": _stats_vector(spec, "observation.state", "q99"),
        "action_q01": _stats_vector(spec, "action", "q01"),
        "action_q99": _stats_vector(spec, "action", "q99"),
        "normalization": spec.normalization,
        **({f"{prefix}_{bound}": _stats_vector(spec, feature, bound)
            for feature, prefix in (("observation.state", "state"), ("action", "action"))
            for bound in ("min", "max")} if spec.normalization == "min_max" else {}),
        "sample_ratio": spec.sample_ratio,
        "training_eligibility": {
            "exists": spec.training_eligibility_exists,
            "used": spec.training_eligibility_used,
            "source": spec.training_eligibility_source,
        },
        "data_version": spec.data_version,
        "observation_contract": {
            "version": spec.observation_contract.version,
            "window_size": spec.observation_contract.window_size,
            "history_order": spec.observation_contract.history_order,
            "history_stride": spec.observation_contract.history_stride,
        },
        "vision_input_contract": _jsonable(spec.vision_input_contract),
        **({"downstream_contract_version": 2} if spec.downstream_contract_version == 2 else {}),
        "sidecar_sha256": spec.sidecar_sha256,
        "canonical_schema": _jsonable(spec.canonical_schema),
        **({"auxiliary_contract": _jsonable(spec.auxiliary_contract)} if spec.auxiliary_contract is not None else {}),
        **({"episode_input_contract": _jsonable(spec.episode_input_contract)}
           if spec.episode_input_contract is not None else {}),
    }


def build_resolved_dataset_manifest(
    specs: Iterable[ResolvedDatasetSpec],
    loss_type: str,
) -> dict:
    entries = [dataset_spec_to_manifest(spec, loss_type) for spec in specs]
    if not entries:
        raise ValueError("resolved dataset manifest requires at least one entry")
    return {
        "format_version": RESOLVED_DATASET_MANIFEST_VERSION,
        "loss_type": loss_type,
        "entries": entries,
    }


def resolved_manifest_json(manifest: dict) -> str:
    value = checkpoint_metadata(manifest)
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n"


def write_resolved_dataset_manifest(directory, manifest: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RESOLVED_DATASET_MANIFEST_NAME
    path.write_text(resolved_manifest_json(manifest), encoding="utf-8")
    return path


def load_resolved_dataset_manifest(directory) -> dict:
    path = Path(directory) / RESOLVED_DATASET_MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"failed to load resolved dataset manifest {path}: {error}") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("entries"), list) or not manifest["entries"]:
        raise ValueError(f"invalid resolved dataset manifest in {path}")
    return manifest


def validate_resume_manifest(
    current_manifest: dict,
    checkpoint_directory,
    *,
    allow_legacy_missing: bool = False,
    allow_legacy_missing_observation_contract: bool = False,
) -> None:
    path = Path(checkpoint_directory) / RESOLVED_DATASET_MANIFEST_NAME
    if not path.is_file():
        if allow_legacy_missing:
            print(
                f"[WARNING] legacy checkpoint has no {RESOLVED_DATASET_MANIFEST_NAME}; "
                "dataset semantic validation is unavailable"
            )
            return
        raise ValueError(f"checkpoint dataset manifest is missing: {path}")
    checkpoint_manifest = load_resolved_dataset_manifest(checkpoint_directory)
    from utils.downstream_contracts import checkpoint_version
    checkpoint_version(checkpoint_directory)
    current = checkpoint_metadata(current_manifest)
    checkpoint = checkpoint_metadata(checkpoint_manifest)
    for entry in checkpoint["entries"]:
        if entry.get("normalization") == "min_max" and "state_min" not in entry:
            entry.update(saved_min_max_bounds(checkpoint_directory, entry["dataset_entry"]))
    _resolve_legacy_observation_contracts(
        current,
        checkpoint,
        allow_legacy_missing=allow_legacy_missing_observation_contract,
    )
    # Historical LeRobot releases used both absent and explicit null. Never
    # normalize a nonempty contract or a v2 entry; their semantics must match.
    for manifest in (current, checkpoint):
        for entry in manifest.get("entries", []):
            if entry.get("resolved_adapter") == "lerobot_v2" and entry.get("downstream_contract_version", 1) == 1:
                entry.setdefault("vision_input_contract", None)
    if current != checkpoint:
        detail = _first_semantic_mismatch(current, checkpoint)
        raise ValueError(
            "dataset manifest mismatch between current resolved data and checkpoint: "
            + (f"; {detail}" if detail else "")
        )


def select_resolved_manifest_entry(manifest: dict, dataset_entry: str) -> dict:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("checkpoint dataset manifest entries must be a list")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("dataset_entry") == dataset_entry
    ]
    if not matches:
        raise ValueError(
            f"dataset entry {dataset_entry!r} was not found in checkpoint manifest"
        )
    if len(matches) != 1:
        raise ValueError(
            f"dataset entry {dataset_entry!r} has multiple matches in checkpoint manifest"
        )
    return matches[0]


def _first_semantic_mismatch(current: dict, checkpoint: dict) -> str | None:
    checkpoint_entries = checkpoint.get("entries")
    current_entries = current.get("entries")
    if not isinstance(checkpoint_entries, list) or not isinstance(current_entries, list):
        return None
    for current_entry in current_entries:
        if not isinstance(current_entry, dict):
            continue
        dataset_entry = current_entry.get("dataset_entry")
        matches = [
            entry
            for entry in checkpoint_entries
            if isinstance(entry, dict) and entry.get("dataset_entry") == dataset_entry
        ]
        if len(matches) != 1:
            continue
        checkpoint_entry = matches[0]
        for field in DATASET_SEMANTIC_FIELDS:
            if current_entry.get(field) != checkpoint_entry.get(field):
                detail_field = field
                current_value = current_entry.get(field)
                checkpoint_value = checkpoint_entry.get(field)
                if isinstance(current_value, dict) and isinstance(checkpoint_value, dict):
                    for nested_field in current_value:
                        if current_value.get(nested_field) != checkpoint_value.get(nested_field):
                            detail_field = f"{field}.{nested_field}"
                            current_value = current_value.get(nested_field)
                            checkpoint_value = checkpoint_value.get(nested_field)
                            break
                return (
                    f"field={detail_field!r}, checkpoint={checkpoint_value!r}, "
                    f"current={current_value!r}, "
                    f"dataset_entry={dataset_entry!r}"
                )
    return None


def validate_policy_dataset_manifest(
    spec: ResolvedDatasetSpec,
    checkpoint_directory,
    *,
    allow_legacy_missing: bool = False,
    allow_legacy_missing_observation_contract: bool = False,
    external_stats: bool = False,
) -> dict | None:
    path = Path(checkpoint_directory) / RESOLVED_DATASET_MANIFEST_NAME
    if not path.is_file():
        if not allow_legacy_missing:
            raise ValueError(f"checkpoint dataset manifest is missing: {path}")
        message = (
            "[WARNING] legacy checkpoint has no resolved dataset manifest; "
            "dataset and normalization semantics cannot be verified"
        )
        print(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return None

    from utils.downstream_contracts import checkpoint_version
    checkpoint_version(checkpoint_directory)
    checkpoint_manifest = load_resolved_dataset_manifest(checkpoint_directory)
    checkpoint_entry = select_resolved_manifest_entry(
        checkpoint_manifest, spec.dataset_entry
    )
    current_entry = dataset_spec_to_manifest(spec, "action")
    if "observation_contract" not in checkpoint_entry:
        if not allow_legacy_missing_observation_contract:
            raise ValueError(
                f"legacy checkpoint entry {spec.dataset_entry!r} has no observation "
                "contract; pass the explicit legacy observation-contract override"
            )
        message = (
            f"[WARNING] legacy checkpoint entry {spec.dataset_entry!r} has no "
            "observation contract; accepting the explicitly configured current contract"
        )
        print(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        checkpoint_entry = copy.deepcopy(checkpoint_entry)
        checkpoint_entry["observation_contract"] = current_entry[
            "observation_contract"
        ]
    current_entry = checkpoint_metadata(current_entry)
    checkpoint_entry = checkpoint_metadata(checkpoint_entry)
    if current_entry["normalization"] == "min_max" and "state_min" not in checkpoint_entry:
        checkpoint_entry.update(saved_min_max_bounds(checkpoint_directory, spec.dataset_entry))
    semantic_fields = DATASET_SEMANTIC_FIELDS
    if external_stats:
        # Compare effective normalization bounds directly, without file digests.
        names = ("min", "max") if current_entry["normalization"] == "min_max" else ("q01", "q99")
        bounds = tuple(f"{prefix}_{name}" for prefix in ("state", "action") for name in names)
        if any(checkpoint_entry.get(key) is None or current_entry.get(key) != checkpoint_entry[key] for key in bounds):
            raise ValueError("external statistics do not match checkpoint training normalization")
    for field in semantic_fields:
        if current_entry.get(field) != checkpoint_entry.get(field):
            detail_field = field
            current_value = current_entry.get(field)
            checkpoint_value = checkpoint_entry.get(field)
            if isinstance(current_value, dict) and isinstance(checkpoint_value, dict):
                for nested_field in current_value:
                    if current_value.get(nested_field) != checkpoint_value.get(nested_field):
                        detail_field = f"{field}.{nested_field}"
                        current_value = current_value.get(nested_field)
                        checkpoint_value = checkpoint_value.get(nested_field)
                        break
            raise ValueError(
                f"checkpoint dataset manifest field {detail_field!r} does not match: "
                f"checkpoint={checkpoint_value!r}, "
                f"current={current_value!r}, "
                f"dataset_entry={spec.dataset_entry!r}"
            )
    return checkpoint_entry


def _resolve_legacy_observation_contracts(
    current_manifest: dict,
    checkpoint_manifest: dict,
    *,
    allow_legacy_missing: bool,
) -> None:
    current_entries = current_manifest.get("entries", [])
    checkpoint_entries = checkpoint_manifest.get("entries", [])
    for current_entry in current_entries:
        dataset_entry = current_entry.get("dataset_entry")
        matches = [
            entry
            for entry in checkpoint_entries
            if entry.get("dataset_entry") == dataset_entry
        ]
        if len(matches) != 1 or "observation_contract" in matches[0]:
            continue
        if not allow_legacy_missing:
            raise ValueError(
                f"legacy checkpoint entry {dataset_entry!r} has no observation "
                "contract; pass the explicit legacy observation-contract override"
            )
        message = (
            f"[WARNING] legacy checkpoint entry {dataset_entry!r} has no observation "
            "contract; accepting the explicitly configured current contract"
        )
        print(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        matches[0]["observation_contract"] = copy.deepcopy(
            current_entry.get("observation_contract")
        )
    if allow_legacy_missing and checkpoint_manifest.get("format_version") == 2:
        checkpoint_manifest["format_version"] = current_manifest.get("format_version")
