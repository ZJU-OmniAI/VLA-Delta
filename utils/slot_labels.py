"""Supervision-only normalization. Presence is annotated position availability."""

import json
import math
import torch

VOCABULARIES = {
    "Q2": ["align", "approach", "contact", "manipulate", "release", "retract", "transport"],
    "Q7": ["closed_to_open", "maintain_closed", "maintain_open", "open_to_closed"],
    "Q8": ["break_contact", "establish_contact", "maintain_contact", "remain_separated"],
}
SHAPES = {"Q1": (2,), "Q2": (2,), "Q3": (2, 4), "Q4": (2, 4), "Q5": (3, 3),
          "Q6": (2, 2), "Q7": (), "Q8": (), "Q9_bbox": (3, 4), "Q9_risk": (3,), "Q9_presence": (3,)}


def empty_slot_labels():
    result = {}
    for key, shape in SHAPES.items():
        result[f"slot_{key}"] = torch.zeros(shape, dtype=torch.long if key in VOCABULARIES else torch.float32)
        mask_shape = shape[:-1] if key in {"Q3", "Q4", "Q5", "Q6", "Q9_bbox"} else shape
        result[f"slot_{key}_mask"] = torch.zeros(mask_shape, dtype=torch.bool)
    return result


def _at(value, index):
    return value[index] if isinstance(value, (list, tuple)) and index < len(value) else None


def _binary(value):
    return type(value) in (int, float, bool) and value in (0, 1)


def _numeric(value, width=None, bounded=False, bbox=False):
    values = [value] if width is None else value
    if not isinstance(values, (list, tuple)) or (width is not None and len(values) != width):
        return False
    if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
        return False
    if bounded and not all(0 <= v <= 1 for v in values):
        return False
    return not bbox or (values[0] < values[2] and values[1] < values[3])


def normalize_slot_labels(raw, *, is_anchor, identity="unknown sample"):
    result = empty_slot_labels()
    if not is_anchor or raw is None or raw == "":
        return result
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{identity}: invalid slot_data JSON") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{identity}: slot_data must be an object")

    def put(q, index, value, valid, *, path, **checks):
        if not valid or value is None:
            return
        if not _numeric(value, **checks):
            raise ValueError(f"{identity}: {path} invalid numeric value/shape/geometry")
        if valid:
            result[f"slot_{q}"][index] = torch.as_tensor(value)
            result[f"slot_{q}_mask"][index] = True

    for number in range(1, 10):
        q = f"Q{number}"
        data = raw.get(f"query_{number}")
        data = {} if data is None else data
        if not isinstance(data, dict):
            raise ValueError(f"{identity}: query_{number} must be an object")
        mask = data.get("valid_mask")
        mask_name = "valid_mask_t" if q == "Q6" else "valid_mask"
        required = data.get(mask_name)
        if q not in {"Q1", "Q9"} and any(v is not None for k, v in data.items() if k != mask_name) and required is None:
            raise ValueError(f"{identity}: query_{number}.{mask_name} is required")
        if required is not None:
            length = {"Q3": 2, "Q4": 2, "Q5": 3, "Q6": 2, "Q9": 3}.get(q)
            legal = (isinstance(required, (list, tuple)) and len(required) == length and all(_binary(v) for v in required)) if length else _binary(required)
            if not legal:
                raise ValueError(f"{identity}: query_{number}.{mask_name} invalid mask/shape")
        if q in VOCABULARIES:
            names = {"Q2": ("phase_t_id", "phase_tK_id"), "Q7": ("gripper_transition_id",), "Q8": ("contact_transition_id",)}[q]
            for i, name in enumerate(names):
                value = data.get(name)
                if value is not None and mask == 1 and not isinstance(value, str):
                    raise ValueError(f"{identity}: query_{number}.{name} must be a class name")
                if isinstance(value, str) and value.strip() and value not in VOCABULARIES[q]:
                    raise ValueError(f"{identity}: unknown {q} class {value!r}")
                idx = i if q == "Q2" else ()
                if value in VOCABULARIES[q] and _binary(mask) and mask == 1:
                    result[f"slot_{q}"][idx] = VOCABULARIES[q].index(value)
                    result[f"slot_{q}_mask"][idx] = True
        elif q == "Q1":
            for i, key in enumerate(("progress_t", "progress_tK")):
                put(q, i, data.get(key), True, path=f"query_{number}.{key}", bounded=True)
        elif q in {"Q3", "Q4", "Q5"}:
            keys = {"Q3": ("target_bbox_t", "target_bbox_tK"), "Q4": ("gripper_bbox_t", "gripper_bbox_tK"),
                    "Q5": ("target_displacement_m", "gripper_displacement_m", "relative_displacement_m")}[q]
            for i, key in enumerate(keys):
                valid = _at(mask, i)
                put(q, i, data.get(key), _binary(valid) and valid == 1,
                    path=f"query_{number}.{key}",
                    width=3 if q == "Q5" else 4, bounded=q != "Q5", bbox=q != "Q5")
        elif q == "Q6":
            points = data.get("affordance_contact_points")
            if points is not None and required is not None and any(required) and (not isinstance(points, (list, tuple)) or len(points) != 2):
                raise ValueError(f"{identity}: query_6.affordance_contact_points invalid shape")
            for i in range(2):
                valid = _at(data.get("valid_mask_t"), i)
                put(q, i, _at(points, i), _binary(valid) and valid == 1,
                    path=f"query_6.affordance_contact_points[{i}]", width=2, bounded=True)
        elif q == "Q9":
            obstacles = data.get("obstacles")
            if obstacles is not None and mask is not None and any(mask) and (not isinstance(obstacles, (list, tuple)) or len(obstacles) > 3):
                raise ValueError(f"{identity}: query_9.obstacles invalid shape")
            for i in range(3):
                valid = _at(mask, i)
                if not _binary(valid):
                    continue
                result["slot_Q9_presence"][i] = float(valid)
                result["slot_Q9_presence_mask"][i] = True
                obstacle = _at(obstacles, i)
                if valid == 1 and obstacle is not None and not isinstance(obstacle, dict):
                    raise ValueError(f"{identity}: query_9.obstacles[{i}] must be an object")
                if not isinstance(obstacle, dict):
                    continue
                put("Q9_bbox", i, obstacle.get("bbox"), valid == 1, path=f"query_9.obstacles[{i}].bbox", width=4, bounded=True, bbox=True)
                put("Q9_risk", i, obstacle.get("risk_score"), valid == 1, path=f"query_9.obstacles[{i}].risk_score", bounded=True)
    return result


def task_validity(batch, *, device=None, batch_size=None):
    if batch_size is None:
        batch_size = batch["input_ids"].shape[0]
    masks = {}
    for q in (f"Q{i}" for i in range(1, 10)):
        keys = ("Q9_presence", "Q9_bbox", "Q9_risk") if q == "Q9" else (q,)
        valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for key in keys:
            value = batch.get(f"slot_{key}_mask")
            if value is not None:
                valid |= value.to(device=device, dtype=torch.bool).reshape(batch_size, -1).any(-1)
        masks[q] = valid
    return masks
