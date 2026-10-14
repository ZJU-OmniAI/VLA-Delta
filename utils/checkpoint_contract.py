"""Checkpoint contracts for pretraining, downstream fine-tuning and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.checkpoint_metadata import checkpoint_metadata

from utils.dataset_manifest import (
    load_resolved_dataset_manifest,
)


STAGE05_AR_JOINT_CONTRACT_KEY = "stage05_ar_joint_contract"
STAGE05_AR_JOINT_CONTRACT_VERSION = 1
STAGE05_ACTION_EXPERT_CONFIG_SCHEMA_VERSION = 1
STAGE05_CHECKPOINT_METADATA_VERSION = 1
CHECKPOINT_METADATA_NAME = "zr0_checkpoint_metadata.json"
STAGE05_DATASET_ENTRIES = (
    "stage05_droid_mixed",
    "stage05_household_mixed",
    "stage05_tabletop_mixed",
    "stage05_rh20t_mixed",
)
PRETRAINING_DATASET_ENTRIES = tuple(name.removeprefix("stage05_") for name in STAGE05_DATASET_ENTRIES)

CHECKPOINT_LOAD_PURPOSES = (
    "stage05_ar_resume",
    "stage05_ar_to_joint",
    "stage05_joint_resume",
    "downstream_finetune",
    "inference",
)
STAGE05_AR_RESUME = "stage05_ar_resume"
STAGE05_AR_TO_JOINT = "stage05_ar_to_joint"
STAGE05_JOINT_RESUME = "stage05_joint_resume"
DOWNSTREAM_FINETUNE = "downstream_finetune"
INFERENCE = "inference"


def validate_checkpoint_load_purpose_arguments(
    purpose: str | None, *, resume_training: bool
) -> None:
    """Validate the purpose/state combination before any model allocation."""
    if purpose is not None and purpose not in CHECKPOINT_LOAD_PURPOSES:
        raise ValueError(f"unknown checkpoint load purpose: {purpose!r}")
    if purpose == STAGE05_AR_RESUME and not resume_training:
        raise ValueError("stage05_ar_resume must use resume_training=True")
    if purpose == STAGE05_AR_TO_JOINT and resume_training:
        raise ValueError(
            "stage05_ar_to_joint is a fresh Joint initialization and cannot use "
            "resume_training=True"
        )
    if purpose == STAGE05_JOINT_RESUME and not resume_training:
        raise ValueError(
            "stage05_joint_resume must use resume_training=True; use "
            "stage05_ar_to_joint for AR-to-Joint initialization"
        )
    if purpose == INFERENCE and resume_training:
        raise ValueError("inference checkpoint loading cannot use resume_training=True")


def _expected_action_expert_state_shapes(config) -> dict[str, tuple[int, ...]]:
    """Build only a meta-device Expert to obtain its complete state contract."""
    import torch
    import model.flow_matching_action_head as action_head_module

    # ``Beta`` validates scalar tensors during construction and is unrelated to
    # parameter structure.  Keep this header-only check allocation-free by
    # replacing that distribution only while constructing the meta module.
    original_beta = action_head_module.Beta
    action_head_module.Beta = lambda *_args, **_kwargs: None
    try:
        with torch.device("meta"):
            expert = action_head_module.FlowmatchingActionHead(config, True)
        return {name: tuple(value.shape) for name, value in expert.state_dict().items()}
    finally:
        action_head_module.Beta = original_beta


def validate_action_expert_weights(
    checkpoint_directory: Path | str,
    config,
    *,
    label: str = "Action Expert checkpoint",
    expected_shapes: dict[str, tuple[int, ...]] | None = None,
) -> None:
    """Validate a safetensors file's complete key/shape contract without GPU load."""
    path = Path(checkpoint_directory).resolve() / "action_expert.safetensors"
    try:
        from safetensors import safe_open

        if not path.is_file():
            raise ValueError(f"{label} weights are missing: {path}")
        expected = expected_shapes or _expected_action_expert_state_shapes(config)
        with safe_open(str(path), framework="pt", device="cpu") as source:
            actual_keys = list(source.keys())
            if set(actual_keys) != set(expected):
                missing = sorted(set(expected) - set(actual_keys))
                unexpected = sorted(set(actual_keys) - set(expected))
                raise ValueError(
                    f"{label} state_dict keys do not match: missing={missing[:5]}, "
                    f"unexpected={unexpected[:5]}"
                )
            mismatches = [
                name
                for name, shape in expected.items()
                if tuple(source.get_slice(name).get_shape()) != shape
            ]
            if mismatches:
                raise ValueError(
                    f"{label} parameter shapes do not match: {mismatches[:5]}"
                )
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"{label} safetensors cannot be parsed: {path}") from error


def _checkpoint_has_stage05_identity(
    checkpoint: Path, metadata: dict[str, Any] | None = None
) -> bool:
    """Detect Stage05 artifacts even when their contract field was removed."""
    metadata = metadata or {}
    if metadata.get("experiment") in {
        "stage05_four_dataset_pretraining_20260904",
        "stage05_four_dataset_pretraining",
    }:
        return True
    # The contract key itself is also an identity marker.  A checkpoint with
    # a removed or malformed manifest must not silently fall through to the
    # permissive legacy loader.
    if STAGE05_AR_JOINT_CONTRACT_KEY in metadata:
        return True
    manifest_path = checkpoint / "resolved_dataset_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = load_resolved_dataset_manifest(checkpoint)
    except Exception as error:
        raise ValueError(
            f"checkpoint has a Stage05-looking dataset manifest that cannot be read: {manifest_path}"
        ) from error
    return is_stage05_four_dataset_manifest(manifest)


def is_stage05_four_dataset_manifest(
    manifest: dict[str, Any] | None,
    *,
    expected_loss_type: str | None = None,
) -> bool:
    """Identify the four-dataset Stage05 manifest for a specific phase.

    AR and Joint manifests share the dataset identity but intentionally carry
    different objective contracts.  Callers that know the checkpoint phase
    should pass ``expected_loss_type`` so an AR manifest cannot masquerade as a
    Joint manifest (or vice versa).
    """
    if not isinstance(manifest, dict):
        return False
    if expected_loss_type is not None and manifest.get("loss_type") != expected_loss_type:
        return False
    if expected_loss_type is None and manifest.get("loss_type") not in {
        "vlm",
        "vlm_and_action",
    }:
        return False
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return False
    names = tuple(
        entry.get("dataset_entry") for entry in entries if isinstance(entry, dict)
    )
    return names in (PRETRAINING_DATASET_ENTRIES, STAGE05_DATASET_ENTRIES) and all(
        entry.get("resolved_adapter") in {"mixed_pretraining", "stage05_mixed_pretraining"}
        for entry in entries
    )


def validate_action_expert_config_provenance(checkpoint_directory, metadata):
    """Validate an explicit-stage runtime config against its original source."""
    from utils.action_expert_config import load_action_expert_config

    provenance = metadata.get("action_expert_config_provenance")
    if provenance is None:
        return None
    if not isinstance(provenance, dict) or provenance.get("version") != 1:
        raise ValueError("Action Expert config provenance is invalid")
    if provenance.get("source_file") != "action_expert_source_config.json":
        raise ValueError("Action Expert source config filename is invalid")
    checkpoint = Path(checkpoint_directory)
    source_path = checkpoint / provenance["source_file"]
    runtime_path = checkpoint / "action_expert_config.json"
    runtime = load_action_expert_config(runtime_path)
    source = load_action_expert_config(source_path, action_horizon_override=runtime.config.action_horizon)
    if source.payload != runtime.payload:
        raise ValueError("Action Expert source/runtime configs differ beyond the horizon")
    if (provenance.get("source_action_horizon") != source.source_action_horizon
            or provenance.get("runtime_action_horizon") != runtime.config.action_horizon):
        raise ValueError("Action Expert source/runtime horizon provenance mismatch")
    return provenance


def build_stage05_ar_joint_contract(
    *, checkpoint_directory: Path | str, resolved_action_expert_config: dict[str, Any],
    resolved_dataset_manifest: dict[str, Any], vlm_hidden_size: int,
    num_difference_queries: int, checkpoint_kind: str = "ar_only",
    runtime_purpose: str = STAGE05_AR_TO_JOINT,
) -> dict[str, Any]:
    """Save explicit architecture and stage semantics without content hashes."""
    from utils.action_expert_config import load_action_expert_config
    if checkpoint_kind not in {"ar_only", "joint"}:
        raise ValueError("Stage05 contract checkpoint_kind must be ar_only or joint")
    expected_loss = "vlm" if checkpoint_kind == "ar_only" else "vlm_and_action"
    if not is_stage05_four_dataset_manifest(resolved_dataset_manifest, expected_loss_type=expected_loss):
        raise ValueError("Stage05 contract dataset manifest does not match checkpoint phase")
    config_path = Path(checkpoint_directory) / "action_expert_config.json"
    if load_action_expert_config(config_path).payload != resolved_action_expert_config:
        raise ValueError("saved Stage05 Action Expert config differs from the constructed config")
    return {
        "version": STAGE05_AR_JOINT_CONTRACT_VERSION,
        "checkpoint_metadata_version": STAGE05_CHECKPOINT_METADATA_VERSION,
        "config_schema_version": STAGE05_ACTION_EXPERT_CONFIG_SCHEMA_VERSION,
        "checkpoint_kind": checkpoint_kind,
        "action_expert_config": {"file": "action_expert_config.json",
                                 "resolved_config": resolved_action_expert_config},
        "vlm_hidden_size": int(vlm_hidden_size),
        "num_difference_queries": int(num_difference_queries),
        "action_dim": int(resolved_action_expert_config["action_dim"]),
        "state_dim": int(resolved_action_expert_config["state_dim"]),
        "action_horizon": int(resolved_action_expert_config["action_horizon"]),
        "runtime_contract": {
            "purpose": runtime_purpose, "checkpoint_kind": checkpoint_kind,
            "training_stage": "ar_only" if checkpoint_kind == "ar_only" else "joint",
            "loss_type": expected_loss,
            "action_horizon": int(resolved_action_expert_config["action_horizon"]),
        },
    }


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{description} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"failed to read {description}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def validate_stage05_joint_warm_start(
    checkpoint_directory: Path | str,
    external_config_path: Path | str,
    *,
    expected_action_dim: int = 64,
    expected_state_dim: int = 64,
    expected_action_horizon: int | None = None,
    expected_num_difference_queries: int = 32,
):
    """Validate the complete Stage05 contract before Joint model construction."""
    from utils.action_expert_config import (
        load_action_expert_config,
        read_vlm_hidden_size,
        validate_difference_query_config,
    )

    checkpoint = Path(checkpoint_directory).resolve()
    metadata_path = checkpoint / "zr0_checkpoint_metadata.json"
    metadata = _read_json_object(metadata_path, "checkpoint metadata")
    validate_action_expert_config_provenance(checkpoint, metadata)
    if metadata.get("version") != STAGE05_CHECKPOINT_METADATA_VERSION:
        raise ValueError(
            "checkpoint is not a compatible Stage05 AR-only checkpoint: "
            f"metadata version must be {STAGE05_CHECKPOINT_METADATA_VERSION}"
        )
    if metadata.get("checkpoint_kind") != "ar_only":
        raise ValueError(
            "checkpoint is not a compatible Stage05 AR-only checkpoint: "
            f"checkpoint_kind={metadata.get('checkpoint_kind')!r}"
        )
    contract = metadata.get(STAGE05_AR_JOINT_CONTRACT_KEY)
    if not isinstance(contract, dict):
        raise ValueError(
            "checkpoint is not a compatible Stage05 Joint warm-start: missing "
            "Stage05 AR-to-Joint configuration contract (legacy Tabletop AR "
            "checkpoints are not accepted)"
        )
    if contract.get("version") != STAGE05_AR_JOINT_CONTRACT_VERSION:
        raise ValueError("Stage05 AR-to-Joint contract version mismatch")
    if contract.get("checkpoint_kind") != "ar_only":
        raise ValueError("Stage05 AR-to-Joint contract checkpoint kind mismatch")
    if (
        contract.get("checkpoint_metadata_version")
        != STAGE05_CHECKPOINT_METADATA_VERSION
    ):
        raise ValueError("Stage05 contract checkpoint metadata version mismatch")
    if (
        contract.get("config_schema_version")
        != STAGE05_ACTION_EXPERT_CONFIG_SCHEMA_VERSION
    ):
        raise ValueError("Stage05 Action Expert config schema version mismatch")
    runtime_contract = contract.get("runtime_contract")
    if not isinstance(runtime_contract, dict):
        raise ValueError("Stage05 AR-to-Joint runtime contract is missing")
    if runtime_contract.get("purpose") != STAGE05_AR_TO_JOINT:
        raise ValueError("Stage05 AR-to-Joint runtime contract purpose mismatch")
    if runtime_contract.get("checkpoint_kind") != "ar_only":
        raise ValueError("Stage05 AR-to-Joint runtime checkpoint kind mismatch")
    if runtime_contract.get("training_stage") != "ar_only":
        raise ValueError("Stage05 AR-to-Joint runtime training stage mismatch")
    if runtime_contract.get("loss_type") != "vlm":
        raise ValueError("Stage05 AR-to-Joint runtime loss type mismatch")
    contract_horizon = runtime_contract.get("action_horizon")
    if expected_action_horizon is None:
        expected_action_horizon = contract_horizon
    if runtime_contract.get("action_horizon") != expected_action_horizon:
        raise ValueError("Stage05 AR-to-Joint runtime action horizon mismatch")

    config_metadata = contract.get("action_expert_config")
    required_config_metadata = {
        "file",
        "resolved_config",
    }
    if not isinstance(config_metadata, dict) or not required_config_metadata.issubset(
        config_metadata
    ):
        raise ValueError("Stage05 AR checkpoint Action Expert config metadata is incomplete")
    if config_metadata["file"] != "action_expert_config.json":
        raise ValueError("Stage05 Action Expert config filename is invalid")
    checkpoint_config_path = checkpoint / config_metadata["file"]
    if not checkpoint_config_path.is_file():
        raise ValueError(
            f"Stage05 AR checkpoint Action Expert config is missing: {checkpoint_config_path}"
        )

    required_scalar_fields = {
        "vlm_hidden_size": None,
        "num_difference_queries": expected_num_difference_queries,
        "action_dim": expected_action_dim,
        "state_dim": expected_state_dim,
        "action_horizon": expected_action_horizon,
    }
    for field, expected in required_scalar_fields.items():
        if field not in contract:
            raise ValueError(f"Stage05 AR-to-Joint contract is missing {field}")
        if expected is not None and contract[field] != expected:
            raise ValueError(
                f"Stage05 AR-to-Joint contract {field} mismatch: "
                f"expected {expected}, got {contract[field]!r}"
            )

    actual_hidden_size = read_vlm_hidden_size(checkpoint)
    if contract["vlm_hidden_size"] != actual_hidden_size:
        raise ValueError("Stage05 AR checkpoint VLM hidden size metadata mismatch")
    checkpoint_resolved = load_action_expert_config(
        checkpoint_config_path,
        expected_action_dim=expected_action_dim,
        expected_state_dim=expected_state_dim,
        expected_action_horizon=expected_action_horizon,
        expected_vlm_hidden_size=actual_hidden_size,
    )
    if (
        config_metadata["resolved_config"] != checkpoint_resolved.payload
    ):
        raise ValueError("Stage05 AR checkpoint embedded Action Expert config mismatch")
    action_metadata = metadata.get("action_expert")
    if (
        not isinstance(action_metadata, dict)
        or action_metadata.get("status")
        != "not_constructed_future_joint_config_reference"
        or action_metadata.get("weights_file") is not None
    ):
        raise ValueError("Stage05 AR checkpoint Expert-free metadata is inconsistent")
    if (checkpoint / "action_expert.safetensors").exists():
        raise ValueError("Stage05 AR-only checkpoint unexpectedly contains Expert weights")

    difference_query_path = checkpoint / "difference_query_config.json"
    validate_difference_query_config(
        difference_query_path,
        expected_num_queries=expected_num_difference_queries,
        expected_hidden_size=actual_hidden_size,
    )
    manifest = load_resolved_dataset_manifest(checkpoint)
    if not is_stage05_four_dataset_manifest(manifest, expected_loss_type="vlm"):
        raise ValueError(
            "checkpoint dataset manifest is not the four-dataset Stage05 AR-only experiment"
        )

    external = load_action_expert_config(
        external_config_path,
        expected_action_dim=expected_action_dim,
        expected_state_dim=expected_state_dim,
        expected_action_horizon=expected_action_horizon,
        expected_vlm_hidden_size=actual_hidden_size,
    )
    if external.payload != checkpoint_resolved.payload:
        raise ValueError(
            "external Action Expert config does not match the "
            "Stage05 AR checkpoint contract"
        )
    return external


def _validate_contract_file_and_config(
    checkpoint: Path,
    metadata: dict[str, Any],
    *,
    expected_checkpoint_kind: str,
    expected_action_horizon: int | None,
    expected_action_dim: int,
    expected_state_dim: int,
    expected_num_difference_queries: int | None,
    require_stage05: bool,
) -> tuple[dict[str, Any], Any]:
    """Validate the shared Expert contract used by resume and fine-tune loads."""
    validate_action_expert_config_provenance(checkpoint, metadata)
    from utils.action_expert_config import (
        load_action_expert_config,
        read_vlm_hidden_size,
        validate_difference_query_config,
    )
    if require_stage05 and metadata.get("version") != STAGE05_CHECKPOINT_METADATA_VERSION:
        raise ValueError("checkpoint metadata version is not compatible with Stage05")
    if require_stage05 and metadata.get("checkpoint_kind") != expected_checkpoint_kind:
        raise ValueError("checkpoint metadata kind does not match Stage05 contract")
    contract = metadata.get(STAGE05_AR_JOINT_CONTRACT_KEY)
    if not isinstance(contract, dict):
        if require_stage05:
            raise ValueError(
                "checkpoint is not a compatible Stage05 checkpoint: missing "
                "the Stage05 Action Expert configuration contract"
            )
        raise ValueError("checkpoint is missing the Action Expert configuration contract")
    if contract.get("version") != STAGE05_AR_JOINT_CONTRACT_VERSION:
        raise ValueError("Stage05 Action Expert contract version mismatch")
    if contract.get("checkpoint_metadata_version") != STAGE05_CHECKPOINT_METADATA_VERSION:
        raise ValueError("Stage05 checkpoint metadata version mismatch")
    if contract.get("config_schema_version") != STAGE05_ACTION_EXPERT_CONFIG_SCHEMA_VERSION:
        raise ValueError("Stage05 Action Expert config schema version mismatch")
    if contract.get("checkpoint_kind") != expected_checkpoint_kind:
        raise ValueError("Action Expert contract checkpoint kind mismatch")
    runtime_contract = contract.get("runtime_contract")
    if not isinstance(runtime_contract, dict):
        raise ValueError("Action Expert runtime contract is missing")
    if runtime_contract.get("checkpoint_kind") != expected_checkpoint_kind:
        raise ValueError("Action Expert runtime checkpoint kind mismatch")
    expected_stage = "ar_only" if expected_checkpoint_kind == "ar_only" else "joint"
    expected_loss = "vlm" if expected_checkpoint_kind == "ar_only" else "vlm_and_action"
    if runtime_contract.get("training_stage") != expected_stage:
        raise ValueError("Action Expert runtime training stage mismatch")
    if runtime_contract.get("loss_type") != expected_loss:
        raise ValueError("Action Expert runtime loss type mismatch")
    if expected_action_horizon is not None and runtime_contract.get(
        "action_horizon"
    ) != expected_action_horizon:
        raise ValueError("Action Expert runtime action horizon mismatch")

    config_metadata = contract.get("action_expert_config")
    required = {
        "file",
        "resolved_config",
    }
    if not isinstance(config_metadata, dict) or not required.issubset(config_metadata):
        raise ValueError("Action Expert contract configuration metadata is incomplete")
    config_path = checkpoint / config_metadata["file"]
    if config_metadata["file"] != "action_expert_config.json" or not config_path.is_file():
        raise ValueError("Action Expert contract configuration file is missing")
    hidden_size = read_vlm_hidden_size(checkpoint)
    if contract.get("vlm_hidden_size") != hidden_size:
        raise ValueError("Action Expert VLM hidden-size contract mismatch")
    resolved = load_action_expert_config(
        config_path,
        expected_action_dim=expected_action_dim,
        expected_state_dim=expected_state_dim,
        expected_action_horizon=expected_action_horizon,
        expected_vlm_hidden_size=hidden_size,
    )
    if config_metadata["resolved_config"] != resolved.payload:
        raise ValueError("Action Expert checkpoint embedded config mismatch")
    if contract.get("action_dim") != expected_action_dim or contract.get(
        "state_dim"
    ) != expected_state_dim:
        raise ValueError("Action Expert dimension contract mismatch")
    if expected_num_difference_queries is not None and contract.get(
        "num_difference_queries"
    ) != expected_num_difference_queries:
        raise ValueError("Difference Query count contract mismatch")
    if require_stage05:
        manifest = load_resolved_dataset_manifest(checkpoint)
        expected_manifest_loss = (
            "vlm" if expected_checkpoint_kind == "ar_only" else "vlm_and_action"
        )
        if not is_stage05_four_dataset_manifest(
            manifest, expected_loss_type=expected_manifest_loss
        ):
            raise ValueError("checkpoint manifest is not the four-dataset Stage05 experiment")
        if expected_num_difference_queries is not None:
            validate_difference_query_config(
                checkpoint / "difference_query_config.json",
                expected_num_queries=expected_num_difference_queries,
                expected_hidden_size=hidden_size,
            )
    return contract, resolved


def validate_generic_action_expert_contract(
    checkpoint_directory: Path | str,
    *,
    expected_action_horizon: int | None = None,
    expected_action_dim: int | None = None,
    expected_state_dim: int | None = None,
    expected_vlm_hidden_size: int | None = None,
    validate_weights: bool = True,
    action_horizon_override: int | None = None,
) -> Any:
    """Validate the generic contract emitted by all new action checkpoints.

    This path is intentionally separate from the four-dataset Stage05 contract:
    old Tabletop checkpoints remain legacy-compatible, while any checkpoint that
    carries this newer contract must be checked instead of silently falling back
    to the legacy loader.
    """
    from utils.action_expert_config import (
        load_action_expert_config, read_vlm_hidden_size,
    )

    checkpoint = Path(checkpoint_directory).resolve()
    metadata = checkpoint_metadata(_read_json_object(checkpoint / CHECKPOINT_METADATA_NAME, "checkpoint metadata"))
    provenance = validate_action_expert_config_provenance(checkpoint, metadata)
    if metadata.get("version") != STAGE05_CHECKPOINT_METADATA_VERSION:
        raise ValueError("generic Action Expert checkpoint metadata version mismatch")
    kind = metadata.get("checkpoint_kind")
    if kind not in {"joint", "action_only"}:
        raise ValueError("generic Action Expert contract requires a Joint or action-only checkpoint")
    if _checkpoint_has_stage05_identity(checkpoint, metadata) and not isinstance(
        metadata.get(STAGE05_AR_JOINT_CONTRACT_KEY), dict
    ):
        raise ValueError(
            "checkpoint has Stage05 dataset identity but is missing its Stage05 Action Expert contract"
        )
    contract = metadata.get("action_expert_contract")
    if not isinstance(contract, dict):
        raise ValueError("checkpoint is missing the Action Expert configuration contract")
    required = {
        "version",
        "runtime_contract",
        "config_file",
        "state_dict_shapes",
    }
    if not required.issubset(contract):
        raise ValueError("Action Expert contract is incomplete")
    if contract.get("version") != 1:
        raise ValueError("Action Expert contract version mismatch")
    runtime = contract["runtime_contract"]
    if not isinstance(runtime, dict):
        raise ValueError("Action Expert runtime contract is missing")
    if runtime.get("checkpoint_kind") != kind:
        raise ValueError("Action Expert runtime checkpoint kind mismatch")
    if runtime.get("action_horizon") is None:
        raise ValueError("Action Expert runtime contract horizon is missing")
    target_horizon = runtime.get("target_action_horizon", runtime.get("action_horizon"))
    source_horizon = runtime.get("source_action_horizon", target_horizon)
    if provenance is not None and source_horizon != provenance["source_action_horizon"]:
        raise ValueError("Action Expert runtime source horizon disagrees with provenance")
    for label, value in (("source", source_horizon), ("target", target_horizon)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Action Expert runtime {label} horizon is invalid")
    if runtime.get("action_horizon") != target_horizon:
        raise ValueError("Action Expert runtime target horizon disagrees with action_horizon")
    config_file = contract.get("config_file")
    if config_file != "action_expert_config.json":
        raise ValueError("Action Expert contract configuration filename is invalid")
    config_path = checkpoint / config_file
    if not config_path.is_file():
        raise ValueError("Action Expert contract configuration file is missing")
    checkpoint_hidden_size = read_vlm_hidden_size(checkpoint)
    if expected_vlm_hidden_size is not None and checkpoint_hidden_size != expected_vlm_hidden_size:
        raise ValueError("Action Expert checkpoint VLM hidden size mismatch")
    resolved = load_action_expert_config(
        config_path,
        expected_action_dim=expected_action_dim,
        expected_state_dim=expected_state_dim,
        expected_action_horizon=expected_action_horizon,
        expected_vlm_hidden_size=checkpoint_hidden_size,
    )
    if runtime["action_horizon"] != resolved.config.action_horizon:
        raise ValueError("Action Expert runtime horizon does not match config")
    if "resolved_config" in contract and contract["resolved_config"] != resolved.payload:
        raise ValueError("Action Expert embedded config differs from saved config")
    state_shapes = contract.get("state_dict_shapes")
    if validate_weights:
        if not isinstance(state_shapes, dict) or not state_shapes:
            raise ValueError("Action Expert state_dict_shapes contract is invalid")
        expected_shapes = _expected_action_expert_state_shapes(resolved.config)
        if state_shapes != {name: list(shape) for name, shape in expected_shapes.items()}:
            raise ValueError("Action Expert state_dict_shapes disagree with the architecture")
        validate_action_expert_weights(
            checkpoint,
            resolved.config,
            label="Action Expert contract weights",
            expected_shapes=expected_shapes,
        )
    if action_horizon_override is not None:
        if expected_action_horizon is not None:
            raise ValueError("expected horizon and horizon override are mutually exclusive")
        return load_action_expert_config(
            config_path,
            expected_action_dim=expected_action_dim,
            expected_state_dim=expected_state_dim,
            expected_vlm_hidden_size=checkpoint_hidden_size,
            action_horizon_override=action_horizon_override,
        )
    return resolved


def _validate_generic_contract_consistency(
    contract: dict[str, Any],
    resolved,
    *,
    expected_runtime_purpose: str | None = None,
    expected_source_horizon: int | None = None,
) -> None:
    """Ensure a second generic contract cannot disagree with Stage05 metadata."""

    if "resolved_config" in contract and contract["resolved_config"] != resolved.payload:
        raise ValueError("Action Expert contracts disagree on configuration")
    runtime = contract.get("runtime_contract")
    if not isinstance(runtime, dict) or runtime.get("action_horizon") != resolved.config.action_horizon:
        raise ValueError("Action Expert contracts disagree on runtime horizon")
    target_horizon = runtime.get("target_action_horizon", runtime.get("action_horizon"))
    if target_horizon != resolved.config.action_horizon:
        raise ValueError("Action Expert contracts disagree on target horizon")
    if (
        expected_source_horizon is not None
        and runtime.get("source_action_horizon", target_horizon)
        != expected_source_horizon
    ):
        raise ValueError("Action Expert contracts disagree on source horizon")
    if (
        expected_runtime_purpose is not None
        and runtime.get("purpose") != expected_runtime_purpose
    ):
        raise ValueError("Action Expert contracts disagree on runtime purpose")


def validate_legacy_action_checkpoint(
    checkpoint_directory: Path | str,
    *,
    expected_action_dim: int | None = None,
    expected_state_dim: int | None = None,
    expected_action_horizon: int | None = None,
    expected_vlm_hidden_size: int | None = None,
) -> Any:
    """Strictly validate a genuinely old generic Joint/action checkpoint."""
    from utils.action_expert_config import load_action_expert_config, read_vlm_hidden_size

    checkpoint = Path(checkpoint_directory).resolve()
    metadata_path = checkpoint / CHECKPOINT_METADATA_NAME
    metadata = (
        _read_json_object(metadata_path, "checkpoint metadata")
        if metadata_path.is_file()
        else {}
    )
    if metadata and metadata.get("checkpoint_kind") not in {"joint", "action_only"}:
        raise ValueError("legacy Action Expert checkpoint must be Joint or action-only")
    if _checkpoint_has_stage05_identity(checkpoint, metadata):
        raise ValueError(
            "checkpoint carries Stage05 identity but lacks its contract; it is not a legacy checkpoint"
        )
    if "action_expert_contract" in metadata:
        raise ValueError(
            "checkpoint has a generic Action Expert contract; use the strict generic validator"
        )
    config_path = checkpoint / "action_expert_config.json"
    hidden = expected_vlm_hidden_size
    if hidden is None:
        try:
            hidden = read_vlm_hidden_size(checkpoint)
        except ValueError:
            hidden = None
    resolved = load_action_expert_config(
        config_path,
        expected_action_dim=expected_action_dim,
        expected_state_dim=expected_state_dim,
        expected_action_horizon=expected_action_horizon,
        expected_vlm_hidden_size=hidden,
    )
    validate_action_expert_weights(checkpoint, resolved.config, label="legacy Action Expert checkpoint")
    return resolved


def _validate_stage05_ar_resume_contract(
    checkpoint, *, external_config_path, requested_action_horizon,
    expected_action_dim, expected_state_dim, expected_num_difference_queries,
):
    from model.difference_query import resolve_difference_query_config
    from utils.action_expert_config import load_action_expert_config

    metadata = checkpoint_metadata(_read_json_object(checkpoint / CHECKPOINT_METADATA_NAME, "checkpoint metadata"))
    contract, resolved = _validate_contract_file_and_config(
        checkpoint, metadata,
        expected_checkpoint_kind="ar_only",
        expected_action_horizon=requested_action_horizon,
        expected_action_dim=expected_action_dim,
        expected_state_dim=expected_state_dim,
        expected_num_difference_queries=expected_num_difference_queries,
        require_stage05=True,
    )
    runtime = contract["runtime_contract"]
    # AR artifacts keep their existing future-Joint contract when resumed.
    if runtime.get("purpose") != STAGE05_AR_TO_JOINT:
        raise ValueError("Stage05 AR resume artifact runtime purpose mismatch")
    if any(value != resolved.config.action_horizon for value in (
        contract.get("action_horizon"), runtime.get("action_horizon"),
    )):
        raise ValueError("Stage05 AR resume horizon contract mismatch")
    action = metadata.get("action_expert")
    if (
        not isinstance(action, dict)
        or action.get("status") != "not_constructed_future_joint_config_reference"
        or action.get("config_file") != "action_expert_config.json"
        or action.get("weights_file") is not None
        or (checkpoint / "action_expert.safetensors").exists()
        or "action_expert_contract" in metadata
    ):
        raise ValueError("Stage05 AR resume requires an Expert-free checkpoint")
    resolve_difference_query_config(
        checkpoint, None, use_difference_query=True,
        num_difference_queries=expected_num_difference_queries,
        vlm_attention_backend="sdpa",
    )
    if external_config_path is not None:
        external = load_action_expert_config(external_config_path)
        if external.payload != resolved.payload:
            raise ValueError("Stage05 AR resume external Expert config mismatch")
    return resolved


def validate_stage05_resume_artifacts(
    checkpoint_directory: Path | str,
    *,
    external_config_path: Path | str | None = None,
    requested_action_horizon: int | None = None,
    expected_action_dim: int = 64,
    expected_state_dim: int = 64,
    expected_num_difference_queries: int = 32,
    purpose: str = STAGE05_JOINT_RESUME,
) -> Any:
    """Validate a Stage05 resume, including CPU-readable training state."""
    checkpoint = Path(checkpoint_directory).resolve()
    if purpose not in {STAGE05_AR_RESUME, STAGE05_JOINT_RESUME}:
        raise ValueError("Stage05 resume artifacts require an AR or Joint resume purpose")
    ar_resume = purpose == STAGE05_AR_RESUME
    label = "Stage05 AR resume" if ar_resume else "Stage05 Joint resume"
    validator = (
        _validate_stage05_ar_resume_contract if ar_resume
        else validate_stage05_checkpoint_for_purpose
    )
    resolved = validator(
        checkpoint,
        **({} if ar_resume else {"purpose": purpose, "resume_training": True}),
        external_config_path=external_config_path,
        requested_action_horizon=requested_action_horizon,
        expected_action_dim=expected_action_dim,
        expected_state_dim=expected_state_dim,
        expected_num_difference_queries=expected_num_difference_queries,
    )
    if not ar_resume:
        validate_action_expert_weights(
            checkpoint, resolved.config, label="Stage05 Joint resume Action Expert"
        )

    scheduler_path = checkpoint / "scheduler.pt"
    if not scheduler_path.is_file():
        raise ValueError(f"{label} scheduler state is missing: {scheduler_path}")
    try:
        import torch

        scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=True)
    except TypeError:
        scheduler_state = torch.load(scheduler_path, map_location="cpu")
    except Exception as error:
        raise ValueError(f"{label} scheduler state cannot be loaded: {scheduler_path}") from error
    if not isinstance(scheduler_state, dict) or not isinstance(
        scheduler_state.get("last_epoch"), int
    ) or not isinstance(scheduler_state.get("_step_count"), int):
        raise ValueError(f"{label} scheduler state is incomplete")

    model_state_paths = sorted(checkpoint.glob("mp_rank_*_model_states.pt"))
    optimizer_state_paths = sorted(checkpoint.glob("*optim_states.pt"))
    if not model_state_paths:
        raise ValueError(f"{label} DeepSpeed model/client state is missing")
    if not optimizer_state_paths:
        raise ValueError(f"{label} DeepSpeed optimizer state is missing")
    global_steps: list[int] = []
    scheduler_steps_per_update = 1
    for path in model_state_paths:
        try:
            try:
                state = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                state = torch.load(path, map_location="cpu")
        except Exception as error:
            raise ValueError(f"{label} client state cannot be loaded: {path}") from error
        if not isinstance(state, dict) or not isinstance(state.get("last_global_step"), int):
            raise ValueError(f"{label} client state lacks last_global_step: {path}")
        if ar_resume:
            module = state.get("module")
            if not isinstance(module, dict) or not module or not state.get("param_shapes"):
                raise ValueError(f"{label} DeepSpeed model state is incomplete: {path}")
            if any(name.startswith("action_expert.") for name in module):
                raise ValueError(f"{label} DeepSpeed model state contains Action Expert weights")
        saved_dp_world_size = state.get("dp_world_size")
        if type(saved_dp_world_size) is not int or saved_dp_world_size <= 0:
            raise ValueError(f"{label} cannot verify scheduler steps: missing or invalid saved dp_world_size: {path}")
        if saved_dp_world_size != len(optimizer_state_paths):
            raise ValueError(f"{label} saved dp_world_size disagrees with optimizer partitions: {path}")
        # Both Stage05 stages use AcceleratedScheduler with split_batches=False
        # and step_with_optimizer=True, independently of the resume topology.
        scheduler_steps_per_update = saved_dp_world_size
        global_steps.append(int(state["last_global_step"]))
        if isinstance(state.get("global_steps"), int) and state["global_steps"] != state["last_global_step"]:
            raise ValueError(f"{label} global step fields disagree: {path}")
    if len(set(global_steps)) != 1 or global_steps[0] < 0:
        raise ValueError(f"{label} client states have inconsistent global steps")
    expected_step = global_steps[0]
    expected_scheduler_step = expected_step * scheduler_steps_per_update
    if (
        scheduler_state["last_epoch"] != expected_scheduler_step
        or scheduler_state["_step_count"] != expected_scheduler_step + 1
    ):
        raise ValueError(
            f"{label} scheduler/global step mismatch: "
            f"scheduler last_epoch={scheduler_state['last_epoch']}, "
            f"_step_count={scheduler_state['_step_count']}, global_step={expected_step}"
        )
    for path in optimizer_state_paths:
        try:
            try:
                state = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                state = torch.load(path, map_location="cpu")
        except Exception as error:
            raise ValueError(f"{label} optimizer state cannot be loaded: {path}") from error
        if not isinstance(state, dict) or not state:
            raise ValueError(f"{label} optimizer state is empty: {path}")
        if ar_resume:
            zero_state = state.get("optimizer_state_dict")
            if not isinstance(zero_state, dict) or zero_state.get("zero_stage") != 2:
                raise ValueError(f"{label} requires a ZeRO-2 optimizer state: {path}")
            optimizer = zero_state.get("base_optimizer_state")
            if (
                not isinstance(optimizer, dict)
                or not isinstance(optimizer.get("state"), dict)
                or not optimizer.get("param_groups")
                or not zero_state.get("single_partition_of_fp32_groups")
                or not zero_state.get("partition_count")
            ):
                raise ValueError(f"{label} optimizer state is incomplete: {path}")
            if any(count != len(optimizer_state_paths) for count in zero_state["partition_count"]):
                raise ValueError(f"{label} optimizer partitions are missing: {path}")
            for group in optimizer["param_groups"]:
                if not group.get("params"):
                    raise ValueError(f"{label} optimizer parameter group is empty: {path}")
                for parameter in group["params"]:
                    moment = optimizer["state"].get(parameter, {})
                    if expected_step > 0 and (
                        not {"step", "exp_avg", "exp_avg_sq"}.issubset(moment)
                        or moment["step"] != expected_step
                    ):
                        raise ValueError(f"{label} optimizer/global step state is inconsistent: {path}")
    return resolved


def validate_stage05_checkpoint_for_purpose(
    checkpoint_directory: Path | str,
    *,
    purpose: str,
    external_config_path: Path | str | None = None,
    requested_action_horizon: int | None = None,
    resume_training: bool = False,
    expected_action_dim: int | None = 64,
    expected_state_dim: int | None = 64,
    expected_num_difference_queries: int = 32,
) -> Any:
    """Validate a checkpoint before model/GPU construction for an explicit purpose."""
    from utils.action_expert_config import architecture_payload, load_action_expert_config

    validate_checkpoint_load_purpose_arguments(
        purpose, resume_training=resume_training
    )
    if purpose == STAGE05_AR_RESUME:
        return validate_stage05_resume_artifacts(
            checkpoint_directory, purpose=purpose,
            external_config_path=external_config_path,
            requested_action_horizon=requested_action_horizon,
            expected_action_dim=expected_action_dim,
            expected_state_dim=expected_state_dim,
            expected_num_difference_queries=expected_num_difference_queries,
        )
    checkpoint = Path(checkpoint_directory).resolve()
    metadata_path = checkpoint / CHECKPOINT_METADATA_NAME
    metadata = (
        _read_json_object(metadata_path, "checkpoint metadata")
        if metadata_path.is_file()
        else {}
    )
    kind = metadata.get("checkpoint_kind")
    if kind is None and (checkpoint / "action_expert.safetensors").is_file():
        kind = "legacy_full"
    stage05_identity = _checkpoint_has_stage05_identity(checkpoint, metadata)
    provenance = validate_action_expert_config_provenance(checkpoint, metadata)
    stage05_contract = metadata.get(STAGE05_AR_JOINT_CONTRACT_KEY)
    if stage05_identity and not isinstance(stage05_contract, dict):
        raise ValueError(
            "checkpoint has Stage05 dataset identity but is missing its Stage05 Action Expert contract"
        )
    if purpose == STAGE05_AR_TO_JOINT:
        if external_config_path is None:
            raise ValueError("Stage05 AR-to-Joint requires an explicit Expert config")
        if not isinstance(stage05_contract, dict):
            raise ValueError("Stage05 AR-to-Joint requires a Stage05 AR-only contract")
        contract_horizon = stage05_contract.get("action_horizon")
        if requested_action_horizon is not None and requested_action_horizon != contract_horizon:
            raise ValueError(
                "Stage05 AR-to-Joint action_horizon must match the AR checkpoint"
            )
        expected_horizon = (
            requested_action_horizon
            if requested_action_horizon is not None
            else contract_horizon
        )
        if not isinstance(expected_horizon, int) or expected_horizon <= 0:
            raise ValueError("Stage05 AR-to-Joint checkpoint horizon is invalid")
        return validate_stage05_joint_warm_start(
            checkpoint,
            external_config_path,
            expected_action_dim=expected_action_dim,
            expected_state_dim=expected_state_dim,
            expected_action_horizon=expected_horizon,
            expected_num_difference_queries=expected_num_difference_queries,
        )

    if purpose == STAGE05_JOINT_RESUME:
        if kind != "joint":
            raise ValueError("Stage05 Joint resume requires a Joint checkpoint")
        if not stage05_identity:
            raise ValueError("Stage05 Joint resume requires a four-dataset Stage05 checkpoint")
        checkpoint_horizon = stage05_contract.get("action_horizon")
        if requested_action_horizon is not None and requested_action_horizon != checkpoint_horizon:
            raise ValueError("Stage05 Joint resume requires the checkpoint action_horizon")
        contract, resolved = _validate_contract_file_and_config(
            checkpoint,
            metadata,
            expected_checkpoint_kind="joint",
            expected_action_horizon=checkpoint_horizon,
            expected_action_dim=expected_action_dim,
            expected_state_dim=expected_state_dim,
            expected_num_difference_queries=expected_num_difference_queries,
            require_stage05=True,
        )
        if contract["runtime_contract"].get("purpose") != STAGE05_JOINT_RESUME:
            raise ValueError("Stage05 Joint resume runtime purpose mismatch")
        if not (checkpoint / "action_expert.safetensors").is_file():
            raise ValueError("Stage05 Joint checkpoint is missing Action Expert weights")
        validate_action_expert_weights(
            checkpoint, resolved.config, label="Stage05 Joint Action Expert"
        )
        if isinstance(metadata.get("action_expert_contract"), dict):
            generic = validate_generic_action_expert_contract(
                checkpoint,
                expected_action_horizon=checkpoint_horizon,
                expected_action_dim=expected_action_dim,
                expected_state_dim=expected_state_dim,
                expected_vlm_hidden_size=resolved.config.vlm_output_embedding_dim,
                validate_weights=True,
            )
            _validate_generic_contract_consistency(
                metadata["action_expert_contract"],
                generic,
                expected_runtime_purpose=STAGE05_JOINT_RESUME,
                expected_source_horizon=(provenance["source_action_horizon"] if provenance else checkpoint_horizon),
            )
        if external_config_path is not None:
            external = load_action_expert_config(
                external_config_path,
                expected_action_dim=expected_action_dim,
                expected_state_dim=expected_state_dim,
                expected_action_horizon=checkpoint_horizon,
                expected_vlm_hidden_size=resolved.config.vlm_output_embedding_dim,
            )
            if external.payload != resolved.payload:
                raise ValueError("resume Expert config does not match checkpoint architecture")
        return resolved

    if purpose == DOWNSTREAM_FINETUNE:
        if requested_action_horizon is None and not resume_training:
            raise ValueError("downstream_finetune fresh initialization requires an explicit action_horizon")
        if not resume_training and (
            (stage05_identity and kind != "joint")
            or (not stage05_identity and kind not in {"joint", "action_only", "legacy_full"})
        ):
            raise ValueError(
                "downstream_finetune requires a compatible Joint or action-only checkpoint"
            )
        if resume_training and kind not in {"joint", "action_only"}:
            raise ValueError("downstream fine-tune resume requires a Joint or action-only checkpoint")
        if resume_training:
            if stage05_identity:
                raise ValueError(
                    "a Stage05 pretraining checkpoint cannot be used as downstream resume; "
                    "use fresh downstream_finetune initialization"
                )
            resolved = validate_generic_action_expert_contract(
                checkpoint,
                expected_action_horizon=requested_action_horizon,
                expected_action_dim=expected_action_dim,
                expected_state_dim=expected_state_dim,
                validate_weights=True,
            )
            if requested_action_horizon is not None and resolved.config.action_horizon != requested_action_horizon:
                raise ValueError("downstream resume action_horizon must match the checkpoint")
            return resolved
        if stage05_identity:
            source_contract, source = _validate_contract_file_and_config(
                checkpoint,
                metadata,
                expected_checkpoint_kind="joint",
                expected_action_horizon=stage05_contract.get("action_horizon"),
                expected_action_dim=expected_action_dim,
                expected_state_dim=expected_state_dim,
                expected_num_difference_queries=expected_num_difference_queries,
                require_stage05=True,
            )
            if source_contract["runtime_contract"].get("purpose") != STAGE05_JOINT_RESUME:
                raise ValueError("Stage05 Joint source runtime purpose mismatch")
            if isinstance(metadata.get("action_expert_contract"), dict):
                generic = validate_generic_action_expert_contract(
                    checkpoint,
                    expected_action_horizon=source.config.action_horizon,
                    expected_action_dim=expected_action_dim,
                    expected_state_dim=expected_state_dim,
                    expected_vlm_hidden_size=source.config.vlm_output_embedding_dim,
                    validate_weights=True,
                )
                _validate_generic_contract_consistency(
                    metadata["action_expert_contract"],
                    generic,
                    expected_runtime_purpose=STAGE05_JOINT_RESUME,
                    expected_source_horizon=(provenance["source_action_horizon"] if provenance else source.config.action_horizon),
                )
            else:
                validate_action_expert_weights(
                    checkpoint,
                    source.config,
                    label="Stage05 downstream source Action Expert",
                )
        else:
            # New generic-contract artifacts are never allowed to fall
            # through the no-contract legacy compatibility path.
            if isinstance(metadata.get("action_expert_contract"), dict):
                source = validate_generic_action_expert_contract(
                    checkpoint,
                    expected_action_dim=expected_action_dim,
                    expected_state_dim=expected_state_dim,
                    validate_weights=True,
                )
            else:
                # A genuine old Tabletop/action checkpoint may be used as a
                # new downstream initialization, but it must pass the same
                # strict config and safetensors key/shape checks.
                source = validate_legacy_action_checkpoint(
                    checkpoint,
                    expected_action_dim=expected_action_dim,
                    expected_state_dim=expected_state_dim,
                )
            source_contract = None
        if external_config_path is not None:
            external = load_action_expert_config(
                external_config_path,
                expected_action_dim=expected_action_dim,
                expected_state_dim=expected_state_dim,
                expected_action_horizon=source.config.action_horizon,
                expected_vlm_hidden_size=source.config.vlm_output_embedding_dim,
            )
            if architecture_payload(external.payload) != architecture_payload(source.payload):
                raise ValueError("downstream Expert config architecture does not match checkpoint")
        if requested_action_horizon is None:
            raise ValueError("downstream_finetune fresh initialization requires an explicit action_horizon")
        # Resolve only the explicit runtime override.  All structure remains
        # bound to the source checkpoint config and strict state loading.
        return load_action_expert_config(
            checkpoint / "action_expert_config.json",
            expected_action_dim=expected_action_dim,
            expected_state_dim=expected_state_dim,
            action_horizon_override=requested_action_horizon,
            expected_vlm_hidden_size=source.config.vlm_output_embedding_dim,
        )

    # Inference is a strict artifact load, but it does not alter the runtime H.
    if kind == "ar_only":
        raise ValueError("checkpoint kind 'ar_only' cannot be used for action inference")
    if kind not in {"joint", "action_only", "legacy_full"}:
        raise ValueError("inference requires a checkpoint containing Action Expert weights")
    if stage05_identity:
        contract, resolved = _validate_contract_file_and_config(
            checkpoint,
            metadata,
            expected_checkpoint_kind=kind,
            expected_action_horizon=None,
            expected_action_dim=expected_action_dim,
            expected_state_dim=expected_state_dim,
            expected_num_difference_queries=expected_num_difference_queries,
            require_stage05=True,
        )
        validate_action_expert_weights(checkpoint, resolved.config, label="Stage05 inference Action Expert")
        if (
            requested_action_horizon is not None
            and requested_action_horizon != resolved.config.action_horizon
        ):
            raise ValueError(
                "inference action_horizon conflicts with the checkpoint horizon"
            )
        generic = metadata.get("action_expert_contract")
        if isinstance(generic, dict):
            generic_resolved = validate_generic_action_expert_contract(
                checkpoint,
                expected_action_horizon=resolved.config.action_horizon,
                expected_action_dim=expected_action_dim,
                expected_state_dim=expected_state_dim,
                expected_vlm_hidden_size=resolved.config.vlm_output_embedding_dim,
                validate_weights=True,
            )
            _validate_generic_contract_consistency(
                generic,
                generic_resolved,
                expected_runtime_purpose=STAGE05_JOINT_RESUME,
                expected_source_horizon=(provenance["source_action_horizon"] if provenance else resolved.config.action_horizon),
            )
        return resolved
    if isinstance(metadata.get("action_expert_contract"), dict):
        resolved = validate_generic_action_expert_contract(
            checkpoint,
            expected_action_horizon=requested_action_horizon,
            expected_action_dim=expected_action_dim,
            expected_state_dim=expected_state_dim,
            validate_weights=True,
        )
        return resolved
    resolved = validate_legacy_action_checkpoint(
        checkpoint,
        expected_action_dim=expected_action_dim,
        expected_state_dim=expected_state_dim,
        expected_action_horizon=requested_action_horizon,
    )
    return resolved


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--external-config", type=Path)
    parser.add_argument(
        "--purpose",
        choices=CHECKPOINT_LOAD_PURPOSES,
        default=STAGE05_AR_TO_JOINT,
    )
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument(
        "--validate-resume-artifacts",
        action="store_true",
        help="Also load scheduler, DeepSpeed client state and optimizer state on CPU.",
    )
    parser.add_argument("--action-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="Optional expected horizon; omitted means use the checkpoint contract.",
    )
    parser.add_argument("--num-difference-queries", type=int, default=32)
    args = parser.parse_args()
    if args.validate_resume_artifacts:
        validate_checkpoint_load_purpose_arguments(
            args.purpose, resume_training=args.resume_training
        )
        resolved = validate_stage05_resume_artifacts(
            args.checkpoint,
            purpose=args.purpose,
            external_config_path=args.external_config,
            requested_action_horizon=args.action_horizon,
            expected_action_dim=args.action_dim,
            expected_state_dim=args.state_dim,
            expected_num_difference_queries=args.num_difference_queries,
        )
    else:
        resolved = validate_stage05_checkpoint_for_purpose(
            args.checkpoint,
            purpose=args.purpose,
            external_config_path=args.external_config,
            requested_action_horizon=args.action_horizon,
            resume_training=args.resume_training,
            expected_action_dim=args.action_dim,
            expected_state_dim=args.state_dim,
            expected_num_difference_queries=args.num_difference_queries,
        )
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "external_config": str(resolved.path),
                "action_horizon": resolved.config.action_horizon,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
