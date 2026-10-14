"""Stage05 auxiliary provenance and strict lazy Flow file validation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from utils.annotation_io import _load_episode_mapping


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flow_contract(entry):
    audit_cache = None
    digest = digest_file
    if entry.get("preparation_audit_cache"):
        from utils.preparation_audit_cache import load_preparation_audit_cache
        audit_cache = load_preparation_audit_cache(entry["preparation_audit_cache"])
        digest = audit_cache.check
    root = Path(entry["dataset_path"]).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    mapping = _load_episode_mapping(root)
    manifest = Path(entry["optical_flow_manifest"])
    if not manifest.is_absolute():
        manifest = Path(entry["optical_flow_data_root"]) / manifest
    if not entry.get("aux_dataset_identity"):
        raise ValueError("Flow registration requires aux_dataset_identity")
    exclusions = entry.get("flow_excluded_frames", {})
    excluded_frames = {}
    for episode, frames in exclusions.items():
        if (not str(episode).isdigit() or str(int(episode)) != str(episode)
                or not isinstance(frames, list) or not frames
                or any(type(frame) is not int or frame < 0 for frame in frames)
                or len(frames) != len(set(frames)) or int(episode) not in mapping):
            raise ValueError("invalid explicit Flow frame exclusion")
        excluded_frames[str(episode)] = sorted(frames)
    result = {"version": 1, "dataset_root": str(root), "dataset_id": entry["aux_dataset_identity"],
            "fps": float(info["fps"]), "camera": entry["camera_keys"][0],
            "nominal_delta_frames": int(entry["flow_delta_frames"]),
            "units": "normalized_source_image_extent", "geometry": "resize_224x224_no_crop",
            "manifest_sha256": digest(manifest),
            "mapping_sha256": digest(root / "meta/stage05_episode_mapping.jsonl"),
            "mapping": {str(ep): row for ep, row in mapping.items()}}
    if excluded_frames:
        result["excluded_frames"] = excluded_frames
    if entry.get("flow_episode_map"):
        path = Path(entry["flow_episode_map"])
        remap = json.loads(path.read_text())
        if (remap.get("version") != 1 or remap["source_manifest_sha256"] != result["manifest_sha256"]
                or remap["target_mapping_sha256"] != result["mapping_sha256"]
                or remap["dataset_root"] != str(root)):
            raise ValueError("Flow episode remapping source identity mismatch")
        pairs = remap["matches"]
        old_to_full = {str(row["flow_episode"]): row["full_episode"] for row in pairs}
        if len(old_to_full) != len(pairs) or len(set(old_to_full.values())) != len(pairs):
            raise ValueError("Flow episode remapping is not one-to-one")
        for row in pairs:
            if mapping[row["full_episode"]]["old_episode_index"] != row["source_episode"]:
                raise ValueError("Flow episode remapping source episode mismatch")
        result.update(flow_episode_map_sha256=digest(path), flow_to_dataset_episode=old_to_full)
    if audit_cache is not None:
        result["audit_cache"] = audit_cache
    return result


def public_flow_contract(contract):
    return {key: value for key, value in contract.items() if key not in {"mapping", "flow_to_dataset_episode", "audit_cache"}}


def validate_flow_file(handle, entry, contract, frames, targets, *, verified_sha256=None):
    expected = {"schema_version": "stage06_flow_v2", "dataset_id": contract["dataset_id"],
                "camera_key": contract["camera"], "output_height": 224, "output_width": 224}
    for key in ("artifact_identity", "generation_identity", "label_identity", "source_fingerprint",
                "checkpoint_sha256", "model_revision"):
        expected[key] = entry[key]
    for key, value in expected.items():
        if handle.attrs.get(key) != value:
            raise ValueError(f"flow metadata mismatch: {key}")
    actual_sha256 = verified_sha256 if verified_sha256 is not None else digest_file(handle.filename)
    if actual_sha256 != entry["sha256"]:
        raise ValueError("Flow file content differs from manifest sha256")
    ep = contract.get("flow_to_dataset_episode", {}).get(str(entry["merged_episode_index"]), entry["merged_episode_index"])
    path = Path(contract["dataset_root"]) / contract["mapping"][str(ep)]["source_data_uri"]
    table = pq.read_table(path, columns=["episode_index", "frame_index", "timestamp"], filters=[("episode_index", "=", ep)])
    rows = table.to_pylist()
    times = {row["frame_index"]: row["timestamp"] for row in rows}
    if len(times) != len(rows):
        raise ValueError("duplicate source frame timestamp")
    source = handle["source_timestamp_s"][:]
    target = handle["target_timestamp_s"][:]
    actual = handle["actual_delta_s"][:]
    for values in (source, target, actual):
        if values.shape != frames.shape or not np.isfinite(values).all():
            raise ValueError("invalid Flow timestamp shape/value")
    excluded = np.isin(frames, contract.get("excluded_frames", {}).get(str(ep), []))
    if int(excluded.sum()) != len(contract.get("excluded_frames", {}).get(str(ep), [])):
        raise ValueError("explicit Flow exclusion names an absent frame")
    active = ~excluded
    if (not np.allclose(source, np.asarray([times[int(f)] for f in frames]), rtol=0, atol=2e-6)
            or not np.allclose(target, np.asarray([times[int(f)] for f in targets]), rtol=0, atol=2e-6)
            or not np.allclose(actual, target - source, rtol=0, atol=2e-6)
            or not np.allclose(actual[active], ((targets - frames) / contract["fps"])[active], rtol=0, atol=2e-5)):
        raise ValueError("Flow timestamp/source/FPS interval mismatch")
