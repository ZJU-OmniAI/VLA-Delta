"""Strict loading and validation for externally sourced Action Expert configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model.flow_matching_action_head import FlowmatchingActionHeadConfig


REQUIRED_TOP_LEVEL_FIELDS = {
    "add_pos_embed": bool,
    "vlm_output_embedding_dim": int,
    "action_or_state_token_embedding_dim": int,
    "mlp_hidden_size": int,
    "max_seq_len": int,
    "action_dim": int,
    "state_dim": int,
    "action_horizon": int,
    "noise_beta_alpha": (int, float),
    "noise_beta_beta": (int, float),
    "noise_s": (int, float),
    "num_timestep_buckets": int,
    "diffusion_transformer_cfg": dict,
}
REQUIRED_DIT_FIELDS = {
    "num_attention_heads": int,
    "attention_head_dim": int,
    "output_dim": int,
    "num_layers": int,
    "dropout": (int, float),
    "attention_bias": bool,
    "activation_fn": str,
    "upcast_attention": bool,
    "norm_type": str,
    "norm_elementwise_affine": bool,
    "norm_eps": (int, float),
    "max_num_positional_embeddings": int,
    "positional_embeddings": (str, type(None)),
    "final_dropout": bool,
    "interleave_self_attention": bool,
    "causal_mask_in_self_attn": bool,
}


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def architecture_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Return the complete Expert structure with the runtime horizon removed.

    The implementation uses a fixed-length position table, while the number of
    action tokens is selected at runtime.  Keeping every other parsed field in
    this identity makes structural drift (including dropout, heads and norms)
    fail closed instead of falling back to class defaults.
    """
    if not isinstance(value, dict):
        raise ValueError("Action Expert config must be a mapping")
    result = dict(value)
    result.pop("action_horizon", None)
    return result


def architecture_config_hash(value: dict[str, Any]) -> str:
    return canonical_json_hash(architecture_payload(value))


@dataclass(frozen=True)
class ResolvedActionExpertConfig:
    path: Path
    source_sha256: str
    parsed_sha256: str
    source_action_horizon: int
    action_horizon_overridden: bool
    payload: dict[str, Any]
    config: FlowmatchingActionHeadConfig


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Action Expert config does not exist: {path}")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except Exception as error:
        raise ValueError(f"Action Expert config is not valid JSON: {path}") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Action Expert config must be a non-empty JSON object: {path}")
    return payload


def _require_fields(payload: dict[str, Any], requirements, *, prefix: str) -> None:
    for key, expected_type in requirements.items():
        if key not in payload:
            raise ValueError(f"Action Expert config is missing {prefix}{key}")
        value = payload[key]
        if isinstance(value, bool) and expected_type is not bool:
            valid = False
        else:
            valid = isinstance(value, expected_type)
        if not valid:
            raise ValueError(f"Action Expert config field {prefix}{key} has invalid type")


def load_action_expert_config(
    path: Path | str,
    *,
    expected_action_dim: int | None = None,
    expected_state_dim: int | None = None,
    expected_action_horizon: int | None = None,
    action_horizon_override: int | None = None,
    expected_vlm_hidden_size: int | None = None,
) -> ResolvedActionExpertConfig:
    if expected_action_horizon is not None and action_horizon_override is not None:
        raise ValueError(
            "expected_action_horizon and action_horizon_override are mutually exclusive"
        )
    path = Path(path).resolve()
    payload = _read_json_object(path)
    _require_fields(payload, REQUIRED_TOP_LEVEL_FIELDS, prefix="")
    dit = payload["diffusion_transformer_cfg"]
    _require_fields(dit, REQUIRED_DIT_FIELDS, prefix="diffusion_transformer_cfg.")

    positive_top_level = (
        "vlm_output_embedding_dim",
        "action_or_state_token_embedding_dim",
        "mlp_hidden_size",
        "max_seq_len",
        "action_dim",
        "state_dim",
        "action_horizon",
        "num_timestep_buckets",
    )
    if any(payload[key] <= 0 for key in positive_top_level):
        raise ValueError("Action Expert config integer dimensions must be positive")
    for key in (
        "num_attention_heads",
        "attention_head_dim",
        "output_dim",
        "num_layers",
        "max_num_positional_embeddings",
    ):
        if dit[key] <= 0:
            raise ValueError(f"Action Expert config {key} must be positive")
    if payload["action_horizon"] + 1 > payload["max_seq_len"]:
        raise ValueError("Action Expert config max_seq_len cannot hold state plus action horizon")
    if payload["action_horizon"] + 1 > dit["max_num_positional_embeddings"]:
        raise ValueError(
            "Action Expert DiT positional capacity cannot hold state plus action horizon"
        )
    if (
        dit["num_attention_heads"] * dit["attention_head_dim"]
        != payload["action_or_state_token_embedding_dim"]
    ):
        raise ValueError("Action Expert attention head dimensions do not equal hidden size")
    if payload["vlm_output_embedding_dim"] != payload["action_or_state_token_embedding_dim"]:
        raise ValueError("Action Expert VLM and action/state hidden sizes must match")

    expectations = {
        "action_dim": expected_action_dim,
        "state_dim": expected_state_dim,
        "action_horizon": expected_action_horizon,
        "vlm_output_embedding_dim": expected_vlm_hidden_size,
    }
    for key, expected in expectations.items():
        if expected is not None and payload[key] != expected:
            raise ValueError(
                f"Action Expert config {key} mismatch: expected {expected}, got {payload[key]}"
            )

    source_action_horizon = payload["action_horizon"]
    resolved_payload = dict(payload)
    if action_horizon_override is not None:
        if (
            isinstance(action_horizon_override, bool)
            or not isinstance(action_horizon_override, int)
            or action_horizon_override <= 0
        ):
            raise ValueError("Action Expert action_horizon override must be a positive integer")
        if action_horizon_override + 1 > payload["max_seq_len"]:
            raise ValueError(
                "Action Expert action_horizon override exceeds state/action sequence capacity"
            )
        if action_horizon_override + 1 > dit["max_num_positional_embeddings"]:
            raise ValueError(
                "Action Expert action_horizon override exceeds DiT positional capacity"
            )
        resolved_payload["action_horizon"] = action_horizon_override

    config = FlowmatchingActionHeadConfig(**resolved_payload)
    parsed = config.to_dict()
    return ResolvedActionExpertConfig(
        path=path,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parsed_sha256=canonical_json_hash(parsed),
        source_action_horizon=source_action_horizon,
        action_horizon_overridden=(
            source_action_horizon != config.action_horizon
        ),
        payload=parsed,
        config=config,
    )


def read_vlm_hidden_size(model_path: Path | str) -> int:
    config_path = Path(model_path).resolve() / "config.json"
    payload = _read_json_object(config_path)
    text_config = payload.get("text_config")
    hidden_size = (
        text_config.get("hidden_size") if isinstance(text_config, dict) else None
    )
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError(f"VLM config has no valid text hidden_size: {config_path}")
    return hidden_size


def validate_difference_query_config(
    path: Path | str, *, expected_num_queries: int, expected_hidden_size: int
) -> None:
    path = Path(path).resolve()
    payload = _read_json_object(path)
    if payload.get("enabled") is not True:
        raise ValueError(f"Difference Query config is not enabled: {path}")
    if payload.get("num_difference_queries") != expected_num_queries:
        raise ValueError(
            f"Difference Query count mismatch in {path}: "
            f"expected {expected_num_queries}, got {payload.get('num_difference_queries')!r}"
        )
    if payload.get("hidden_size") != expected_hidden_size:
        raise ValueError(
            f"Difference Query hidden size mismatch in {path}: "
            f"expected {expected_hidden_size}, got {payload.get('hidden_size')!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--state-dim", type=int, required=True)
    parser.add_argument("--action-horizon", type=int, required=True)
    parser.add_argument("--difference-query-config", type=Path)
    parser.add_argument("--num-difference-queries", type=int, default=32)
    args = parser.parse_args()
    hidden_size = read_vlm_hidden_size(args.vlm)
    resolved = load_action_expert_config(
        args.config,
        expected_action_dim=args.action_dim,
        expected_state_dim=args.state_dim,
        expected_action_horizon=args.action_horizon,
        expected_vlm_hidden_size=hidden_size,
    )
    if args.difference_query_config is not None:
        validate_difference_query_config(
            args.difference_query_config,
            expected_num_queries=args.num_difference_queries,
            expected_hidden_size=hidden_size,
        )
    print(
        json.dumps(
            {
                "path": str(resolved.path),
                "source_sha256": resolved.source_sha256,
                "parsed_sha256": resolved.parsed_sha256,
                "vlm_hidden_size": hidden_size,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
