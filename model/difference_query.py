from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from safetensors.torch import load_file, save_file


DIFFERENCE_QUERY_CONFIG_NAME = "difference_query_config.json"
DIFFERENCE_QUERY_WEIGHT_NAME = "difference_query.safetensors"
DIFFERENCE_QUERY_WEIGHT_KEY = "difference_query"
DIFFERENCE_QUERY_CONFIG_VERSION = 1
SUPPORTED_ATTENTION_BACKENDS = {"eager", "flash_attention_2", "sdpa"}


@dataclass(frozen=True)
class ResolvedDifferenceQueryConfig:
    enabled: bool
    num_difference_queries: Optional[int]
    attention_backend: Optional[str]
    expected_hidden_size: Optional[int]
    checkpoint_tensor: Optional[torch.Tensor]
    random_initialization: bool
    checkpoint_directories: tuple[Path, ...]


@dataclass(frozen=True)
class DifferenceQuerySequence:
    auxiliary_input_ids: torch.Tensor
    valid_attention_mask: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor]
    query_positions: torch.Tensor
    context_lengths: torch.Tensor
    target_lengths: torch.Tensor


class DifferenceQuery(nn.Module):
    def __init__(self, num_queries: int, hidden_size: int, initializer_std: float):
        super().__init__()
        if num_queries <= 0 or hidden_size <= 0:
            raise ValueError("Difference Query dimensions must be positive")
        if initializer_std < 0:
            raise ValueError("Difference Query initializer_std must be non-negative")

        weight = torch.empty(num_queries, hidden_size, dtype=torch.float32)
        with torch.random.fork_rng(devices=[]):
            nn.init.normal_(weight, mean=0.0, std=initializer_std)
        self.weight = nn.Parameter(weight)

    @property
    def num_queries(self) -> int:
        return self.weight.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.weight.shape[1]

    def for_batch(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if reference.shape[-1] != self.hidden_size:
            raise ValueError(
                f"reference hidden size {reference.shape[-1]} does not match "
                f"Difference Query hidden size {self.hidden_size}"
            )
        values = self.weight.to(device=reference.device, dtype=reference.dtype)
        return values.unsqueeze(0).expand(batch_size, -1, -1)


@dataclass(frozen=True)
class _CheckpointQueryData:
    directory: Path
    enabled: bool
    num_difference_queries: Optional[int]
    hidden_size: int
    attention_backend: Optional[str]
    tensor: Optional[torch.Tensor]

    @property
    def canonical_config(self) -> tuple[object, ...]:
        return (
            DIFFERENCE_QUERY_CONFIG_VERSION,
            self.enabled,
            self.num_difference_queries,
            self.hidden_size,
            self.attention_backend,
        )


def _validate_sequence_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor],
    num_queries: int,
) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, L]")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same [B, L] shape as input_ids")
    if labels is not None and labels.shape != input_ids.shape:
        raise ValueError("labels must have the same [B, L] shape as input_ids")
    if num_queries <= 0:
        raise ValueError("num_queries must be positive")

    valid_mask = attention_mask.to(dtype=torch.bool)
    if valid_mask.shape[1] > 1 and (valid_mask[:, 1:] & ~valid_mask[:, :-1]).any():
        raise ValueError("Difference Query requires right padding")
    if labels is not None and (labels.masked_select(~valid_mask) != -100).any():
        raise ValueError("labels must not supervise padding positions")
    return valid_mask


def _build_block_attention_mask(
    context_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    num_queries: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size = context_lengths.shape[0]
    positions = torch.arange(sequence_length, device=device).unsqueeze(0)
    context_ends = context_lengths.unsqueeze(1)
    query_ends = context_ends + num_queries
    target_ends = query_ends + target_lengths.unsqueeze(1)

    regions = torch.full(
        (batch_size, sequence_length), 3, dtype=torch.uint8, device=device
    )
    regions.masked_fill_(positions < context_ends, 0)
    regions.masked_fill_(
        (positions >= context_ends) & (positions < query_ends), 1
    )
    regions.masked_fill_(
        (positions >= query_ends) & (positions < target_ends), 2
    )

    row_regions = regions.unsqueeze(2)
    key_regions = regions.unsqueeze(1)
    causal = positions.unsqueeze(2) >= positions.unsqueeze(1)
    mask = torch.zeros(
        batch_size,
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=device,
    )
    scratch = torch.empty_like(mask)

    torch.logical_and(row_regions == 0, key_regions == 0, out=scratch)
    scratch.logical_and_(causal)
    mask.logical_or_(scratch)

    torch.logical_and(row_regions == 1, key_regions < 2, out=scratch)
    mask.logical_or_(scratch)

    torch.logical_and(row_regions == 2, key_regions == 1, out=scratch)
    mask.logical_or_(scratch)

    torch.logical_and(row_regions == 2, key_regions == 2, out=scratch)
    scratch.logical_and_(causal)
    mask.logical_or_(scratch)
    return mask.unsqueeze(1)


def build_difference_query_sequence(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    labels: Optional[torch.Tensor],
    num_queries: int,
    placeholder_token_id: int,
) -> DifferenceQuerySequence:
    valid_mask = _validate_sequence_inputs(
        input_ids, attention_mask, labels, num_queries
    )
    batch_size, original_length = input_ids.shape
    extended_length = original_length + num_queries
    valid_lengths = valid_mask.sum(dim=1, dtype=torch.long)
    context_lengths = valid_lengths.clone()
    target_lengths = torch.zeros_like(valid_lengths)

    if labels is not None:
        supervised = (labels != -100) & valid_mask
        has_supervision = supervised.any(dim=1)
        first_supervised = supervised.to(dtype=torch.long).argmax(dim=1)
        positions = torch.arange(original_length, device=input_ids.device).unsqueeze(0)
        expected_supervision = (
            (positions >= first_supervised.unsqueeze(1))
            & (positions < valid_lengths.unsqueeze(1))
            & has_supervision.unsqueeze(1)
        )
        if (supervised != expected_supervision).any():
            raise ValueError(
                "supervised assistant target must be contiguous through the final valid token"
            )
        context_lengths = torch.where(
            has_supervision, first_supervised, valid_lengths
        )
        target_lengths = supervised.sum(dim=1, dtype=torch.long)

    positions = torch.arange(
        extended_length, dtype=torch.long, device=input_ids.device
    ).unsqueeze(0).expand(batch_size, -1)
    context_ends = context_lengths.unsqueeze(1)
    query_ends = context_ends + num_queries
    valid_ends = valid_lengths.unsqueeze(1) + num_queries
    is_context = positions < context_ends
    is_query = (positions >= context_ends) & (positions < query_ends)

    source_positions = torch.where(is_context, positions, positions - num_queries)
    source_positions.clamp_(min=0, max=original_length - 1)
    auxiliary_ids = input_ids.gather(1, source_positions)
    auxiliary_ids.masked_fill_(is_query, placeholder_token_id)
    extended_valid_mask = positions < valid_ends
    query_positions = context_ends + torch.arange(
        num_queries, dtype=torch.long, device=input_ids.device
    ).unsqueeze(0)

    if labels is not None:
        extended_labels = labels.gather(1, source_positions)
        is_target = (positions >= query_ends) & (positions < valid_ends)
        extended_labels.masked_fill_(~is_target, -100)
    else:
        extended_labels = None

    block_mask = _build_block_attention_mask(
        context_lengths=context_lengths,
        target_lengths=target_lengths,
        num_queries=num_queries,
        sequence_length=extended_length,
        device=input_ids.device,
    )
    return DifferenceQuerySequence(
        auxiliary_input_ids=auxiliary_ids,
        valid_attention_mask=extended_valid_mask,
        attention_mask=block_mask,
        labels=extended_labels,
        query_positions=query_positions,
        context_lengths=context_lengths,
        target_lengths=target_lengths,
    )


def save_difference_query_artifacts(
    save_directory: object,
    *,
    enabled: bool,
    hidden_size: int,
    difference_query: Optional[torch.Tensor],
) -> None:
    directory = Path(save_directory)
    directory.mkdir(parents=True, exist_ok=True)
    hidden_size = _positive_int(
        hidden_size, "hidden_size", directory / DIFFERENCE_QUERY_CONFIG_NAME
    )
    weight_path = directory / DIFFERENCE_QUERY_WEIGHT_NAME

    if enabled:
        if difference_query is None or difference_query.ndim != 2:
            raise ValueError("enabled Difference Query must provide a rank-2 weight tensor")
        if difference_query.shape[1] != hidden_size:
            raise ValueError(
                "Difference Query weight hidden size does not match the VLM: "
                f"{difference_query.shape[1]} != {hidden_size}"
            )
        num_queries = _positive_int(
            difference_query.shape[0],
            "num_difference_queries",
            directory / DIFFERENCE_QUERY_CONFIG_NAME,
        )
        if not difference_query.dtype.is_floating_point:
            raise ValueError("Difference Query weights must be floating point")
        if not torch.isfinite(difference_query.detach()).all():
            raise ValueError("Difference Query weights must contain only finite values")
        save_file(
            {
                DIFFERENCE_QUERY_WEIGHT_KEY: difference_query.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            },
            str(weight_path),
        )
        attention_backend = "sdpa"
    else:
        if difference_query is not None:
            raise ValueError("disabled Difference Query must not provide weights")
        num_queries = None
        attention_backend = None
        if weight_path.exists():
            weight_path.unlink()

    config = {
        "version": DIFFERENCE_QUERY_CONFIG_VERSION,
        "enabled": enabled,
        "num_difference_queries": num_queries,
        "hidden_size": hidden_size,
        "attention_backend": attention_backend,
    }
    (directory / DIFFERENCE_QUERY_CONFIG_NAME).write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


def _positive_int(value: object, field_name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}: {field_name} must be a positive integer")
    return value


def _read_checkpoint_query_data(directory: Path) -> Optional[_CheckpointQueryData]:
    config_path = directory / DIFFERENCE_QUERY_CONFIG_NAME
    weight_path = directory / DIFFERENCE_QUERY_WEIGHT_NAME
    has_config = config_path.is_file()
    has_weights = weight_path.is_file()

    flow_config = directory / "optical_flow_aux_config.json"
    flow_weights = directory / "optical_flow_aux.safetensors"
    if (flow_config.exists() or flow_weights.exists()) and not has_config:
        raise ValueError(f"{directory}: OF checkpoint requires complete Difference Query artifacts; missing config")

    if not has_config and not has_weights:
        return None
    if has_weights and not has_config:
        raise ValueError(
            f"{directory}: found {DIFFERENCE_QUERY_WEIGHT_NAME} without its config"
        )

    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{config_path}: invalid Difference Query config") from exc
    if not isinstance(raw_config, dict):
        raise ValueError(f"{config_path}: Difference Query config must be a JSON object")
    if raw_config.get("version") != DIFFERENCE_QUERY_CONFIG_VERSION:
        raise ValueError(
            f"{config_path}: unsupported Difference Query config version "
            f"{raw_config.get('version')!r}"
        )

    enabled = raw_config.get("enabled")
    if flow_weights.exists() and not enabled:
        raise ValueError(f"{directory}: OF weights require enabled Difference Query artifacts")
    if not isinstance(enabled, bool):
        raise ValueError(f"{config_path}: enabled must be a boolean")
    hidden_size = _positive_int(raw_config.get("hidden_size"), "hidden_size", config_path)
    backend = raw_config.get("attention_backend")

    if enabled:
        if not has_weights:
            raise ValueError(
                f"{directory}: enabled Difference Query config is missing "
                f"{DIFFERENCE_QUERY_WEIGHT_NAME}"
            )
        num_queries = _positive_int(
            raw_config.get("num_difference_queries"),
            "num_difference_queries",
            config_path,
        )
        if backend != "sdpa":
            raise ValueError(f"{config_path}: enabled Difference Query requires SDPA")
        tensors = load_file(str(weight_path), device="cpu")
        if set(tensors) != {DIFFERENCE_QUERY_WEIGHT_KEY}:
            raise ValueError(
                f"{weight_path}: expected only tensor key {DIFFERENCE_QUERY_WEIGHT_KEY!r}"
            )
        tensor = tensors[DIFFERENCE_QUERY_WEIGHT_KEY]
        expected_shape = (num_queries, hidden_size)
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{weight_path}: Difference Query shape {tuple(tensor.shape)} does not "
                f"match config shape {expected_shape}"
            )
        if not tensor.dtype.is_floating_point:
            raise ValueError(f"{weight_path}: Difference Query weights must be floating point")
        tensor = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{weight_path}: Difference Query weights must contain only finite values"
            )
    else:
        if has_weights:
            raise ValueError(
                f"{directory}: disabled Difference Query config must not have weights"
            )
        if raw_config.get("num_difference_queries") is not None:
            raise ValueError(
                f"{config_path}: disabled Difference Query must set num_difference_queries to null"
            )
        if backend is not None:
            raise ValueError(
                f"{config_path}: disabled Difference Query must set attention_backend to null"
            )
        num_queries = None
        tensor = None

    return _CheckpointQueryData(
        directory=directory,
        enabled=enabled,
        num_difference_queries=num_queries,
        hidden_size=hidden_size,
        attention_backend=backend,
        tensor=tensor,
    )


def _normalize_local_directories(*paths: object) -> tuple[Path, ...]:
    directories: list[Path] = []
    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(raw_path).expanduser()
        if not path.is_dir():
            continue
        path = path.resolve()
        if path not in directories:
            directories.append(path)
    return tuple(directories)


def resolve_difference_query_config(
    vlm_name_or_path: object,
    action_expert_name_or_path: object,
    *,
    use_difference_query: Optional[bool] = None,
    num_difference_queries: Optional[int] = None,
    vlm_attention_backend: Optional[str] = None,
) -> ResolvedDifferenceQueryConfig:
    if use_difference_query is not None and not isinstance(use_difference_query, bool):
        raise TypeError("use_difference_query must be True, False, or None")
    if num_difference_queries is not None:
        _positive_int(num_difference_queries, "num_difference_queries", Path("CLI"))
    if (
        vlm_attention_backend is not None
        and vlm_attention_backend not in SUPPORTED_ATTENTION_BACKENDS
    ):
        raise ValueError(
            f"vlm_attention_backend must be one of {sorted(SUPPORTED_ATTENTION_BACKENDS)}"
        )

    directories = _normalize_local_directories(
        vlm_name_or_path, action_expert_name_or_path
    )
    directories_with_declarations = tuple(
        directory
        for directory in directories
        if (directory / DIFFERENCE_QUERY_CONFIG_NAME).is_file()
        or (directory / DIFFERENCE_QUERY_WEIGHT_NAME).is_file()
    )
    legacy_directories = tuple(
        directory
        for directory in directories
        if directory not in directories_with_declarations
    )
    if directories_with_declarations and legacy_directories:
        declared_locations = ", ".join(
            str(directory) for directory in directories_with_declarations
        )
        missing_locations = ", ".join(
            str(directory) for directory in legacy_directories
        )
        raise ValueError(
            "cannot mix Difference Query checkpoint with legacy checkpoint across "
            "loading directories: Difference Query declaration found in "
            f"{declared_locations}; missing Difference Query declaration in "
            f"{missing_locations}"
        )
    declarations = tuple(
        data
        for directory in directories
        if (data := _read_checkpoint_query_data(directory)) is not None
    )
    enabled_declarations = tuple(data for data in declarations if data.enabled)

    if enabled_declarations and len(enabled_declarations) != len(declarations):
        locations = ", ".join(str(data.directory) for data in declarations)
        raise ValueError(
            "Difference Query enabled state conflicts across loading directories: "
            f"{locations}"
        )

    if len(declarations) > 1:
        first = declarations[0]
        for other in declarations[1:]:
            if first.canonical_config != other.canonical_config:
                raise ValueError(
                    "loading directories contain different Difference Query configs: "
                    f"{first.directory} and {other.directory}"
                )
            if first.enabled and not torch.equal(first.tensor, other.tensor):
                raise ValueError(
                    "loading directories contain different Difference Query weights: "
                    f"{first.directory} and {other.directory}"
                )

    if enabled_declarations:
        checkpoint = enabled_declarations[0]
        if use_difference_query is False:
            raise ValueError(
                "Difference Query checkpoint was found but the feature was explicitly disabled"
            )
        if (
            num_difference_queries is not None
            and num_difference_queries != checkpoint.num_difference_queries
        ):
            raise ValueError(
                "explicit num_difference_queries conflicts with the checkpoint: "
                f"{num_difference_queries} != {checkpoint.num_difference_queries}"
            )
        if vlm_attention_backend not in (None, "sdpa"):
            raise ValueError("Difference Query requires the SDPA attention backend")
        return ResolvedDifferenceQueryConfig(
            enabled=True,
            num_difference_queries=checkpoint.num_difference_queries,
            attention_backend="sdpa",
            expected_hidden_size=checkpoint.hidden_size,
            checkpoint_tensor=checkpoint.tensor,
            random_initialization=False,
            checkpoint_directories=tuple(data.directory for data in enabled_declarations),
        )

    if declarations:
        checkpoint = declarations[0]
        if use_difference_query is True:
            raise ValueError(
                "Difference Query checkpoint explicitly disables the feature, but "
                "the current configuration explicitly enables it"
            )
        if num_difference_queries is not None:
            raise ValueError(
                "num_difference_queries conflicts with a checkpoint that explicitly "
                "disables Difference Query"
            )
        return ResolvedDifferenceQueryConfig(
            enabled=False,
            num_difference_queries=None,
            attention_backend=vlm_attention_backend,
            expected_hidden_size=checkpoint.hidden_size,
            checkpoint_tensor=None,
            random_initialization=False,
            checkpoint_directories=tuple(data.directory for data in declarations),
        )

    if use_difference_query is not True:
        if num_difference_queries is not None:
            raise ValueError(
                "random Difference Query initialization requires use_difference_query=True"
            )
        return ResolvedDifferenceQueryConfig(
            enabled=False,
            num_difference_queries=None,
            attention_backend=vlm_attention_backend,
            expected_hidden_size=None,
            checkpoint_tensor=None,
            random_initialization=False,
            checkpoint_directories=(),
        )

    if vlm_attention_backend not in (None, "sdpa"):
        raise ValueError("Difference Query requires the SDPA attention backend")
    return ResolvedDifferenceQueryConfig(
        enabled=True,
        num_difference_queries=(
            num_difference_queries if num_difference_queries is not None else 32
        ),
        attention_backend="sdpa",
        expected_hidden_size=None,
        checkpoint_tensor=None,
        random_initialization=True,
        checkpoint_directories=(),
    )
