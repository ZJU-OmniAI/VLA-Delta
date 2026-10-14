"""Content-addressed stage admission independent of enabled auxiliary readers."""

import json
from pathlib import Path

import numpy as np

from utils.sidecar import sha256_file, canonical_json_hash


def load_frozen_stage_index(path, *, dataset_root, sidecar_root, phase, audit_cache=None):
    path = Path(path).resolve()
    digest = audit_cache.check if audit_cache is not None else sha256_file
    if audit_cache is not None:
        audit_cache.check(path)
    report = json.loads(path.read_text())
    content_hash = report.pop("content_hash")
    if report.get("version") != 1 or canonical_json_hash(report) != content_hash:
        raise ValueError("frozen stage index integrity mismatch")
    if report["dataset_root"] != str(Path(dataset_root).resolve()):
        raise ValueError("frozen stage index dataset identity mismatch")
    if digest(Path(sidecar_root) / "manifest.json") != report["sidecar_manifest_sha256"][phase]:
        raise ValueError("frozen stage index sidecar identity mismatch")
    item = report["stage3" if phase == "joint" else "stage12"]
    index_path = path.parent / item["file"]
    if not index_path.resolve().is_relative_to(path.parent) or digest(index_path) != item["sha256"]:
        raise ValueError("frozen stage index file integrity mismatch")
    indices = np.load(index_path, mmap_mode="r", allow_pickle=False)
    if (indices.ndim != 1 or indices.dtype != np.int64 or len(indices) != item["count"]
            or not len(indices) or indices[0] < 0 or indices[-1] >= report["source_frames"]
            or (audit_cache is None and (indices[1:] <= indices[:-1]).any())):
        raise ValueError("invalid frozen stage index")
    identity = {"manifest_sha256": digest(path), "index_sha256": item["sha256"],
                     "count": len(indices), "phase": "stage3" if phase == "joint" else "stage12",
                     "statistics": report["statistics"], "source_audit": report.get("source_audit")}
    if audit_cache is not None:
        identity["preparation_audit_sha256"] = audit_cache.sha256
    return indices, identity
