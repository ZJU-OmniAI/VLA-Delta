"""Versioned downstream data contracts. Legacy callers remain version 1."""
from dataclasses import replace
import json
from pathlib import Path

import torch

ACTIVE_GR1 = tuple(range(13)) + tuple(range(22, 35)) + tuple(range(41, 44))
PLACEHOLDER_GR1 = tuple(range(13, 22)) + tuple(range(35, 41))
PROVENANCE_LIMITATION = (
    "Uses the existing released videos; there is not yet complete evidence that upstream "
    "cropping/padding of training videos matches the current evaluation pipeline. "
    "This uncertainty has been accepted and may affect interpretation of evaluation performance."
)
IDENTITY_KEYS = {"downstream_dataset_entry", "downstream_episode_id", "downstream_frame_id",
                 "downstream_time_valid", "downstream_camera_keys"}


def vision_contract(info, cameras):
    actual = [key for key, value in info["features"].items() if value["dtype"] == "video"]
    if list(cameras) != actual:
        raise ValueError(f"metadata camera order differs: {cameras} != {actual}")
    return dict(version=1, camera_keys=list(cameras), window_size=1, color_mode="RGB",
        source_images=[dict(camera_key=k, shape_hwc=list(info["features"][k]["shape"])) for k in cameras],
        resize=dict(width=224, height=224, interpolation="bicubic", preserve_aspect_ratio=False),
        crop=False, pad=False, letterbox=False, augmentation=False,
        normalization=dict(rescale_factor=1/255, mean=[.5]*3, std=[.5]*3),
        processor_do_resize=False, image_grid_thw_per_view=[1, 14, 14])


def bind_spec(spec, info, version=1):
    if version == 1:
        return spec
    if version != 2 or spec.adapter != "lerobot_v2" or spec.observation_contract.window_size != 1:
        raise ValueError("downstream contract v2 requires current-frame LeRobot v2")
    if not 0 < spec.action_dim <= 64 or not 0 < spec.state_dim <= 64:
        raise ValueError("invalid downstream native dimensions")
    return replace(spec, downstream_contract_version=2,
                   vision_input_contract=vision_contract(info, spec.camera_keys))


def checkpoint_version(directory):
    """Do not infer legacy solely from a missing manifest field."""
    root = Path(directory)
    markers = []
    manifest = {}
    # Completion/index records are independent v2 evidence. Removing all
    # primary markers must not make an exported model enter the legacy path.
    for receipt_path in (root / "model_export_complete.json", root / "dataset_artifacts/index.json"):
        if receipt_path.exists() and json.loads(receipt_path.read_text()).get("version") == 2:
            markers.append(2)
    recovery_receipt = root.parent / "checkpoint_complete.json"
    if root.name == "latest-model-optimizer-lr" and recovery_receipt.exists():
        receipt = json.loads(recovery_receipt.read_text())
        if receipt.get("version") == 2 and "experiment_id" in receipt:
            markers.append(2)
    for name in ("config.json", "run_identity.json", "downstream_contract.json"):
        path = root / name
        if path.exists():
            value = json.loads(path.read_text()).get("downstream_contract_version")
            if value is not None:
                markers.append(value)
    path = root / "resolved_dataset_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        markers += [e["downstream_contract_version"] for e in manifest.get("entries", [])
                    if "downstream_contract_version" in e]
    if not markers:
        return 1
    if any(value != 2 for value in markers):
        raise ValueError("conflicting downstream checkpoint contract versions")
    for name in ("config.json", "run_identity.json", "downstream_contract.json"):
        if not (root / name).exists() or json.loads((root / name).read_text()).get("downstream_contract_version") != 2:
            raise ValueError(f"v2 checkpoint missing version marker: {name}")
    if not manifest.get("entries") or any(e.get("downstream_contract_version") != 2 or
            not isinstance(e.get("vision_input_contract"), dict) for e in manifest["entries"]):
        raise ValueError("v2 checkpoint has missing/empty dataset vision contract")
    return 2


def validate_entry(entry, info):
    name = entry["dataset_entry"]
    expected = vision_contract(info, entry["camera_keys"])
    if entry.get("downstream_contract_version") != 2 or entry.get("vision_input_contract") != expected:
        raise ValueError(f"{name}: missing or conflicting v2 vision contract")
    for field, key in (("action_dimension", "action"), ("state_dimension", "observation.state")):
        if info["features"][key]["shape"] != [entry[field]]:
            raise ValueError(f"{name}: metadata {field} differs")
    if entry["action_horizon"] != 16 or entry["observation_contract"]["window_size"] != 1:
        raise ValueError(f"{name}: H16/current-frame contract differs")


def validate_batch(batch, entries):
    missing = IDENTITY_KEYS - batch.keys()
    if missing:
        raise ValueError(f"missing downstream sample identities: {sorted(missing)}")
    by_name = {e["dataset_entry"]: e for e in entries}
    batch_size = batch["action"].shape[0]
    if batch["action"].shape != (batch_size, 16, 64) or batch["action_mask"].dtype != torch.bool:
        raise ValueError("H16/64 action shape or mask dtype changed")
    expected_grids = []
    for row, name in enumerate(batch["downstream_dataset_entry"]):
        entry = by_name[name]
        identity = (name, int(batch["downstream_episode_id"][row]), int(batch["downstream_frame_id"][row]))
        valid = batch["downstream_time_valid"][row]
        if valid.shape != (16,) or valid.dtype != torch.bool or not bool(valid[0]) or bool((valid[1:] & ~valid[:-1]).any()):
            raise ValueError(f"{identity}: invalid time mask")
        expected = valid[:, None] & (torch.arange(64, device=valid.device) < entry["action_dimension"])
        if not torch.equal(batch["action_mask"][row], expected):
            raise ValueError(f"{identity}: native action/time padding mask differs")
        if list(batch["downstream_camera_keys"][row]) != entry["camera_keys"]:
            raise ValueError(f"{identity}: production camera order differs")
        expected_grids.extend([entry["vision_input_contract"]["image_grid_thw_per_view"]]*len(entry["camera_keys"]))
    grid = torch.tensor(expected_grids, device=batch["image_grid_thw"].device)
    if not torch.equal(batch["image_grid_thw"], grid):
        raise ValueError("production processor view count/resolution differs")
    for key in ("action", "observation.state", "pixel_values"):
        if not torch.isfinite(batch[key]).all():
            raise ValueError(f"nonfinite production {key}")


def validate_source_images(data, spec):
    """Validate decoded frames before resizing can conceal a source mismatch."""
    for source in spec.vision_input_contract["source_images"]:
        key = source["camera_key"]
        height, width, channels = source["shape_hwc"]
        image = data[key]
        if image.shape != (channels, height, width) or image.dtype != torch.float32:
            raise ValueError(f"{spec.dataset_entry}: decoded {key} shape/dtype differs from manifest")
        if not bool(torch.isfinite(image).all()) or image.min() < 0 or image.max() > 1:
            raise ValueError(f"{spec.dataset_entry}: decoded RGB must be finite float32 in [0,1]")
