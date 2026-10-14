"""Shared vision/text inputs and future-difference target formatting."""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any
import torch
from PIL import Image
from utils.training_tokenization import (
    encode_target_text_tokens,
    measure_assistant_response_boundary,
    tokenize_chat_with_complete_assistant,
)

try:
    from qwen_vl_utils import process_vision_info as _qwen_process_vision_info
except ImportError:
    _qwen_process_vision_info = None

FUTURE_DIFFERENCE_KEYS = (
    "Task_temporal",
    "Spatial_motion",
    "Contact_interaction",
    "Object_constraints",
)
SUPPORTED_LOSS_TYPES = {"vlm", "action", "vlm_and_action", "aux"}
FUTURE_DIFFERENCE_IMAGE_SIZE = 224


def process_vision_info(messages, image_patch_size=16):
    if _qwen_process_vision_info is not None:
        return _qwen_process_vision_info(messages, image_patch_size=image_patch_size)
    images = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        images.extend(
            item["image"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "image"
            and "image" in item
        )
    return images, None


def canonicalize_future_difference_target(raw_target: Any, sample_id: str) -> str:
    if not isinstance(raw_target, str) or not raw_target.strip():
        raise ValueError(f"{sample_id}: target must be a non-empty string")
    try:
        parsed = json.loads(raw_target)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{sample_id}: target must be a valid JSON object") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{sample_id}: target must be a JSON object")
    if set(parsed) != set(FUTURE_DIFFERENCE_KEYS):
        raise ValueError(
            f"{sample_id}: target keys must be exactly {list(FUTURE_DIFFERENCE_KEYS)}"
        )
    canonical = {}
    for key in FUTURE_DIFFERENCE_KEYS:
        value = parsed[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{sample_id}: target field {key!r} must be a non-empty string"
            )
        canonical[key] = value
    return json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))


def build_future_difference_message(
    task: str, images: list[tuple[str, Image.Image]], target: str | None = None
) -> list[dict[str, Any]]:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    if len(images) != 3:
        raise ValueError(
            f"future-difference messages require exactly 3 images, got {len(images)}"
        )
    content = []
    for camera_key, image in images:
        content.append({"type": "text", "text": camera_key})
        content.append(
            {
                "type": "image",
                "image": image,
                "resized_height": FUTURE_DIFFERENCE_IMAGE_SIZE,
                "resized_width": FUTURE_DIFFERENCE_IMAGE_SIZE,
            }
        )
    content.append({"type": "text", "text": "<TASK> " + task.strip() + " </TASK>"})
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def encode_future_difference_target_tokens(tokenizer, target: str) -> list[int]:
    return encode_target_text_tokens(tokenizer, target)


def measure_future_difference_token_boundary(
    messages: list[dict[str, Any]],
    processor,
    target: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return the untruncated processor output and exact semantic token span."""
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    return measure_assistant_response_boundary(
        messages,
        processor,
        image_inputs=image_inputs,
        video_inputs=video_inputs,
        target=target,
    )


def tokenize_future_difference_message(
    messages: list[dict[str, Any]],
    processor,
    max_length: int,
    sample_id: str,
    target: str | None = None,
    dataset_entry: str = "lerobot_v3_future_difference",
) -> dict[str, torch.Tensor]:
    """Tokenize once without truncation, enforce the boundary, then pad safely."""
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    return tokenize_chat_with_complete_assistant(
        messages,
        processor,
        image_inputs=image_inputs,
        video_inputs=video_inputs,
        max_length=max_length,
        dataset_entry=dataset_entry,
        sample_id=sample_id,
        target=target,
    )


def prepare_future_difference_direct_inputs(
    *,
    task: str,
    images: list[tuple[str, Image.Image]],
    processor,
) -> dict[str, torch.Tensor]:
    """Build the v3 direct-action context with the training prompt contract."""
    messages = build_future_difference_message(task, images, target=None)
    encoded, boundary = measure_future_difference_token_boundary(
        messages, processor, target=None
    )
    context_length = boundary["original_total"]
    result = {
        "input_ids": encoded["input_ids"][0][:context_length],
        "attention_mask": encoded["attention_mask"][0][:context_length],
        "context_token_count": torch.tensor(context_length, dtype=torch.long),
        "sub_task_flag": torch.tensor(0),
    }
    for key in ("pixel_values", "image_grid_thw"):
        if key in encoded:
            result[key] = encoded[key]
    return result


def resolve_future_difference_vision_contract(
    root: Path, camera_keys: list[str], processor
) -> dict[str, Any] | None:
    image_processor = getattr(processor, "image_processor", None)
    required_processor_fields = (
        "merge_size",
        "rescale_factor",
        "image_mean",
        "image_std",
        "patch_size",
        "temporal_patch_size",
    )
    if image_processor is None or any(
        not hasattr(image_processor, field) for field in required_processor_fields
    ):
        return None
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    source_images = []
    dummy_images = []
    for camera_key in camera_keys:
        shape = info["features"][camera_key]["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or shape[2] != 3
            or not all(isinstance(value, int) and value > 0 for value in shape)
        ):
            raise ValueError(
                f"camera {camera_key!r} must have a positive [height,width,3] shape"
            )
        height, width, channels = shape
        source_images.append(
            {
                "camera_key": camera_key,
                "shape_hwc": [height, width, channels],
            }
        )
        dummy_images.append((camera_key, Image.new("RGB", (width, height))))

    messages = build_future_difference_message(
        "vision input contract probe", dummy_images, target=None
    )
    encoded, _ = measure_future_difference_token_boundary(messages, processor)
    grid = encoded.get("image_grid_thw")
    pixels = encoded.get("pixel_values")
    if not isinstance(grid, torch.Tensor) or grid.shape != (len(camera_keys), 3):
        raise ValueError(
            "processor must return one image_grid_thw row for every configured camera"
        )
    if not isinstance(pixels, torch.Tensor):
        raise ValueError("processor must return pixel_values for the vision contract")
    merge_size = int(image_processor.merge_size)
    grid_rows = grid.detach().cpu().to(torch.long).tolist()
    visual_tokens = [math.prod(row) // (merge_size**2) for row in grid_rows]
    return {
        "version": 1,
        "camera_order": list(camera_keys),
        "source_images": source_images,
        "color_mode": "RGB",
        "resize": {
            "height": FUTURE_DIFFERENCE_IMAGE_SIZE,
            "width": FUTURE_DIFFERENCE_IMAGE_SIZE,
            "preserve_aspect_ratio": False,
            "interpolation": "bicubic",
        },
        "crop": False,
        "pad": False,
        "letterbox": False,
        "augmentation": False,
        "normalization": {
            "rescale_factor": float(image_processor.rescale_factor),
            "mean": list(image_processor.image_mean),
            "std": list(image_processor.image_std),
        },
        "patch_size": int(image_processor.patch_size),
        "temporal_patch_size": int(image_processor.temporal_patch_size),
        "spatial_merge_size": merge_size,
        "image_grid_thw_by_camera": {
            key: row for key, row in zip(camera_keys, grid_rows)
        },
        "visual_tokens_by_camera": {
            key: count for key, count in zip(camera_keys, visual_tokens)
        },
        "total_visual_tokens": sum(visual_tokens),
        "pixel_values_shape_per_sample": list(pixels.shape),
    }
