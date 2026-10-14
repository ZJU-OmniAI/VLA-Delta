"""Shared target-boundary, token-metric, and dataset-integrity helpers."""

from __future__ import annotations

import errno
import logging
from dataclasses import dataclass
from typing import Any, Callable

import torch


class DatasetIntegrityError(ValueError):
    """Permanent sample corruption or exhausted same-sample I/O retries."""


@dataclass(frozen=True)
class TokenMetricSpec:
    dtype: torch.dtype
    requires_target: bool = False
    aggregation: str = "count"


TOKENIZATION_METRIC_SCHEMA = {
    "context_token_count": TokenMetricSpec(torch.long),
    "json_content_token_count": TokenMetricSpec(torch.long, requires_target=True),
    "chat_termination_token_count": TokenMetricSpec(torch.long, requires_target=True),
    "target_region_token_count": TokenMetricSpec(torch.long, requires_target=True),
    "padding_token_count": TokenMetricSpec(torch.long),
    "original_target_token_count": TokenMetricSpec(torch.long, requires_target=True),
    "kept_target_token_count": TokenMetricSpec(torch.long, requires_target=True),
    "supervised_token_count": TokenMetricSpec(torch.long, requires_target=True),
    "truncated_token_count": TokenMetricSpec(torch.long),
    "input_truncated": TokenMetricSpec(torch.bool, aggregation="ratio"),
    "target_truncated": TokenMetricSpec(
        torch.bool, requires_target=True, aggregation="ratio"
    ),
}


def token_metric_validity_key(metric_key: str) -> str:
    if metric_key not in TOKENIZATION_METRIC_SCHEMA:
        raise KeyError(f"unknown tokenization metric {metric_key!r}")
    return f"{metric_key}_valid"


def attach_token_metric_validity(
    result: dict[str, torch.Tensor], *, has_target: bool
) -> dict[str, torch.Tensor]:
    for key, spec in TOKENIZATION_METRIC_SCHEMA.items():
        if key in result:
            result[token_metric_validity_key(key)] = torch.tensor(
                has_target or not spec.requires_target, dtype=torch.bool
            )
    return result


def encode_target_text_tokens(tokenizer, target: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(target, add_special_tokens=False))
    encoded = tokenizer(target, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return list(ids)


def _find_last_subsequence(sequence: torch.Tensor, subsequence: torch.Tensor) -> int:
    if subsequence.numel() == 0 or subsequence.numel() > sequence.numel():
        return -1
    matches = sequence.unfold(0, subsequence.numel(), 1).eq(subsequence).all(dim=1)
    indices = torch.where(matches)[0]
    return int(indices[-1]) if indices.numel() else -1


def assistant_termination_token_ids(processor) -> list[int]:
    cached = getattr(processor, "_zr0_assistant_termination_token_ids", None)
    if cached is not None:
        return list(cached)
    marker = "__ZR0_ASSISTANT_BOUNDARY__"
    probe = [
        {"role": "user", "content": "boundary"},
        {"role": "assistant", "content": marker},
    ]
    rendered = processor.apply_chat_template(
        probe, tokenize=False, add_generation_prompt=False
    )
    rendered_ids = torch.tensor(
        encode_target_text_tokens(processor.tokenizer, rendered), dtype=torch.long
    )
    marker_ids = torch.tensor(
        encode_target_text_tokens(processor.tokenizer, marker), dtype=torch.long
    )
    marker_start = _find_last_subsequence(rendered_ids, marker_ids)
    if marker_start < 0:
        raise DatasetIntegrityError(
            "assistant boundary marker is absent from the chat template"
        )
    suffix = rendered_ids[marker_start + marker_ids.numel() :].tolist()
    if not suffix:
        raise DatasetIntegrityError(
            "assistant chat template does not define termination tokens"
        )
    processor._zr0_assistant_termination_token_ids = tuple(suffix)
    return suffix


def extract_final_assistant_target(messages: list[dict[str, Any]]) -> str:
    if not messages or messages[-1].get("role") != "assistant":
        raise DatasetIntegrityError(
            "teacher-forced chat sequence must end with an assistant response"
        )
    content = messages[-1].get("content")
    if isinstance(content, str):
        target = content
    elif isinstance(content, list) and all(
        isinstance(item, dict) and item.get("type") == "text"
        for item in content
    ):
        target = "".join(str(item.get("text", "")) for item in content)
    else:
        raise DatasetIntegrityError(
            "assistant response must be a string or text-only content list"
        )
    if not target.strip():
        raise DatasetIntegrityError("assistant target must be a non-empty string")
    return target


def measure_assistant_response_boundary(
    messages: list[dict[str, Any]],
    processor,
    *,
    image_inputs,
    video_inputs,
    target: str | None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run one untruncated processor call and resolve the exact assistant span."""
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=target is None,
    )
    encoded = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        truncation=False,
        do_resize=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    original_total = int(attention_mask.sum().item())
    input_ids = input_ids[:original_total]

    target_ids = torch.empty(0, dtype=input_ids.dtype, device=input_ids.device)
    termination_ids = torch.empty(0, dtype=input_ids.dtype, device=input_ids.device)
    assistant_start = original_total
    assistant_end = original_total
    if target is not None:
        if not isinstance(target, str) or not target.strip():
            raise DatasetIntegrityError("assistant target must be a non-empty string")
        target_ids = torch.tensor(
            encode_target_text_tokens(processor.tokenizer, target),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        if target_ids.numel() == 0:
            raise DatasetIntegrityError("empty target token sequence")
        assistant_start = _find_last_subsequence(input_ids, target_ids)
        if assistant_start < 0:
            raise DatasetIntegrityError(
                "assistant content tokens are not present in the chat sequence"
            )
        termination_ids = torch.tensor(
            assistant_termination_token_ids(processor),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        content_end = assistant_start + target_ids.numel()
        assistant_end = content_end + termination_ids.numel()
        if assistant_end != original_total or not torch.equal(
            input_ids[content_end:assistant_end], termination_ids
        ):
            raise DatasetIntegrityError(
                "assistant response boundary does not match the standard chat "
                "termination sequence"
            )

    return encoded, {
        "original_total": original_total,
        "assistant_start": assistant_start,
        "assistant_end": assistant_end,
        "content_end": assistant_start + target_ids.numel(),
        "target_ids": target_ids,
        "termination_ids": termination_ids,
    }


def tokenize_chat_with_complete_assistant(
    messages: list[dict[str, Any]],
    processor,
    *,
    image_inputs,
    video_inputs,
    max_length: int,
    dataset_entry: str,
    sample_id: str,
    target: str | None,
) -> dict[str, torch.Tensor]:
    """Tokenize without truncation, reject overflow, then pad on the right."""
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    identity = f"dataset_entry={dataset_entry} {sample_id}".strip()
    try:
        encoded, boundary = measure_assistant_response_boundary(
            messages,
            processor,
            image_inputs=image_inputs,
            video_inputs=video_inputs,
            target=target,
        )
    except DatasetIntegrityError as error:
        raise DatasetIntegrityError(f"{identity}: {error}") from error

    original_total = boundary["original_total"]
    assistant_start = boundary["assistant_start"]
    assistant_end = boundary["assistant_end"]
    content_end = boundary["content_end"]
    target_ids = boundary["target_ids"]
    termination_ids = boundary["termination_ids"]
    if original_total > max_length:
        kept_target_region = max(
            0, min(max_length, assistant_end) - assistant_start
        )
        kind = "target_truncated" if target is not None else "input_truncated"
        raise DatasetIntegrityError(
            f"{identity}: {kind}; context_tokens={assistant_start}, "
            f"original_target_tokens={target_ids.numel()}, "
            f"original_target_region={assistant_end - assistant_start}, "
            f"termination={termination_ids.numel()}, "
            f"termination_tokens={termination_ids.numel()}, "
            f"total_tokens={original_total}, max_length={max_length}, "
            f"kept_target_region={kept_target_region}"
        )

    input_ids = encoded["input_ids"][0][:original_total]
    attention_mask = encoded["attention_mask"][0][:original_total]
    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise DatasetIntegrityError(
            f"{identity}: processor tokenizer must define pad_token_id"
        )
    pad_count = max_length - original_total
    if pad_count:
        input_ids = torch.cat(
            [
                input_ids,
                torch.full(
                    (pad_count,),
                    pad_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ),
            ]
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.zeros(
                    pad_count,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ]
        )

    result = {
        key: value
        for key, value in dict(encoded).items()
        if key not in {"input_ids", "attention_mask"}
    }
    result.update(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "context_token_count": torch.tensor(assistant_start, dtype=torch.long),
            "json_content_token_count": torch.tensor(
                target_ids.numel(), dtype=torch.long
            ),
            "chat_termination_token_count": torch.tensor(
                termination_ids.numel(), dtype=torch.long
            ),
            "target_region_token_count": torch.tensor(
                assistant_end - assistant_start, dtype=torch.long
            ),
            "padding_token_count": torch.tensor(pad_count, dtype=torch.long),
            "original_target_token_count": torch.tensor(
                target_ids.numel(), dtype=torch.long
            ),
            "kept_target_token_count": torch.tensor(
                content_end - assistant_start, dtype=torch.long
            ),
            "supervised_token_count": torch.tensor(
                assistant_end - assistant_start, dtype=torch.long
            ),
            "truncated_token_count": torch.tensor(0, dtype=torch.long),
            "input_truncated": torch.tensor(False),
            "target_truncated": torch.tensor(False),
        }
    )
    if target is not None:
        labels = torch.full_like(input_ids, -100)
        labels[assistant_start:assistant_end] = input_ids[
            assistant_start:assistant_end
        ]
        if not labels.ne(-100).any():
            raise DatasetIntegrityError(
                f"{identity}: no supervised target tokens remain"
            )
        result["labels"] = labels
    return attach_token_metric_validity(result, has_target=target is not None)


TRANSIENT_IO_ERRNOS = frozenset(
    {errno.EINTR, errno.EAGAIN, errno.ESTALE, errno.ETIMEDOUT}
)


def is_transient_io_error(error: BaseException) -> bool:
    if isinstance(error, (TimeoutError, InterruptedError, BlockingIOError)):
        return True
    return isinstance(error, OSError) and error.errno in TRANSIENT_IO_ERRNOS


def run_with_same_sample_retries(
    operation: Callable[[], Any],
    *,
    dataset_entry: str,
    sample_id: str,
    max_transient_retries: int,
    return_retry_count: bool = False,
):
    if (
        isinstance(max_transient_retries, bool)
        or not isinstance(max_transient_retries, int)
        or max_transient_retries < 0
    ):
        raise ValueError("max_transient_retries must be a non-negative integer")
    attempts = max_transient_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            result = operation()
            if return_retry_count:
                return result, attempt - 1
            return result
        except DatasetIntegrityError:
            raise
        except Exception as error:
            identity = f"dataset_entry={dataset_entry} {sample_id}".strip()
            if not is_transient_io_error(error):
                raise DatasetIntegrityError(f"{identity}: {error}") from error
            if attempt == attempts:
                raise DatasetIntegrityError(
                    f"{identity}: transient I/O failed after {attempts} attempts: {error}"
                ) from error
            logging.getLogger(__name__).warning(
                "%s: transient I/O error on attempt %d/%d; retrying the same sample: %s",
                identity,
                attempt,
                attempts,
                error,
            )
