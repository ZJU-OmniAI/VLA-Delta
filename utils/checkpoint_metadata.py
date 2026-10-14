"""Checkpoint metadata without content digests; data/cache hashing is separate."""

from __future__ import annotations

import re


_DIGEST = re.compile(r"[0-9a-fA-F]{64}\Z")


def is_checkpoint_hash_field(key: str, value=None) -> bool:
    name = str(key).lower()
    return (
        "sha256" in name or "checksum" in name or "fingerprint" in name
        or name in {"hash", "hashes", "digest", "digests", "digest_protocol"}
        or name.endswith(("_hash", "_hashes", "_digest", "_digests"))
        or (isinstance(value, str) and _DIGEST.fullmatch(value) is not None)
    )


def checkpoint_metadata(value):
    """Copy metadata, omitting digest fields from old or new checkpoint records.

    Apply only at checkpoint boundaries. Source dataset/cache manifests keep
    their own hash protocols and must not be passed through this function.
    """
    if isinstance(value, dict):
        return {key: checkpoint_metadata(item) for key, item in value.items()
                if not is_checkpoint_hash_field(key, item)}
    if isinstance(value, list):
        return [checkpoint_metadata(item) for item in value]
    return value


def validate_normalization_snapshot(stats, entry):
    """Check the effective bounds used by a policy against its training record."""
    mode = entry.get("normalization")
    if mode in {"quantile", "quantile_min_max_q01_q99"}:
        bounds = ("q01", "q99")
    elif mode == "min_max":
        bounds = ("min", "max")
        # Older manifests did not record min/max; their own snapshot supplies
        # these values when comparing an explicitly selected external file.
        if all(f"{prefix}_{bound}" not in entry for prefix in ("state", "action") for bound in bounds):
            return
    else:
        return
    for feature, prefix in (("observation.state", "state"), ("action", "action")):
        for bound in bounds:
            expected = entry.get(f"{prefix}_{bound}")
            actual = stats.get(feature, {}).get(bound)
            if expected is None or actual != expected:
                raise ValueError(f"checkpoint normalization differs from training: {feature}.{bound}")


def saved_min_max_bounds(checkpoint_directory, dataset_entry):
    """Read legacy min/max values from an indexed checkpoint stats snapshot."""
    import json
    from pathlib import Path
    root = Path(checkpoint_directory).resolve()
    candidates = []
    for filename in ("inference_metadata.json", "dataset_artifacts/index.json"):
        index = root / filename
        if not index.is_file():
            continue
        rows = json.loads(index.read_text())["entries"]
        matches = [row for row in rows if row["dataset_entry"] == dataset_entry]
        if len(matches) != 1:
            raise ValueError("checkpoint metadata has no unique dataset entry")
        candidates.append(root / matches[0]["directory"] / "meta/stats.json")
    # Single-dataset recovery snapshots predate the indexed export format.
    candidates.append(root / "dataset_artifacts/meta/stats.json")
    for candidate in candidates:
        path = candidate.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            continue
        stats = json.loads(path.read_text())
        return {f"{prefix}_{bound}": stats[feature][bound]
                for feature, prefix in (("observation.state", "state"), ("action", "action"))
                for bound in ("min", "max")}
    raise ValueError("legacy min/max normalization needs its checkpoint stats snapshot to compare external statistics")
