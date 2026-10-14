"""Read pretraining episode mappings and locate source annotations."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import pyarrow.parquet as pq


def _read_parquet_columns(path: Path, columns: Iterable[str]):
    return pq.read_table(path, columns=list(columns))


def _data_paths(dataset_root: Path, info: Mapping[str, Any]) -> list[Path]:
    metadata_paths = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not metadata_paths:
        raise FileNotFoundError(
            f"no episode metadata under {dataset_root / 'meta' / 'episodes'}"
        )
    locations: set[tuple[int, int]] = set()
    for metadata_path in metadata_paths:
        table = _read_parquet_columns(
            metadata_path, ("data/chunk_index", "data/file_index")
        )
        for row in table.to_pylist():
            locations.add((int(row["data/chunk_index"]), int(row["data/file_index"])))
    template = info.get("data_path")
    if not isinstance(template, str) or not template:
        raise ValueError("meta/info.json must define data_path")
    return [
        dataset_root / template.format(chunk_index=chunk, file_index=file_index)
        for chunk, file_index in sorted(locations)
    ]


def _load_episode_mapping(dataset_root: Path) -> dict[int, dict[str, Any]]:
    path = dataset_root / "meta" / "stage05_episode_mapping.jsonl"
    mapping = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            new_episode = int(row["new_episode_index"])
            if new_episode in mapping:
                raise ValueError(
                    f"{path}:{line_number}: duplicate new_episode_index {new_episode}"
                )
            mapping[new_episode] = row
    return mapping


def _resolve_annotation_root(
    dataset_root: Path, annotation_root: str | Path | None
) -> tuple[Path | None, str]:
    if annotation_root is not None:
        candidate = Path(annotation_root).expanduser().resolve()
        return (candidate if candidate.is_dir() else None, "command_line")
    manifest_path = dataset_root / "meta" / "stage05_merge.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        configured = manifest.get("stage05_dir")
        if isinstance(configured, str) and configured:
            candidate = Path(configured).expanduser()
            if not candidate.is_absolute():
                candidate = dataset_root / candidate
            candidate = candidate.resolve()
            if candidate.is_dir():
                return candidate, "meta/stage05_merge.json:stage05_dir"
    return None, "unavailable"


def _active_training_samples(annotation_root: Path, old_episode: int) -> Path | None:
    episode_root = annotation_root / f"episode_{old_episode:06d}"
    status_paths = sorted(episode_root.glob("shard_*/rank_*/status.json"))
    for status_path in status_paths:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            artifact_path = Path(status["artifacts"]["training_samples"]["path"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        candidates = [
            artifact_path,
            status_path.parent / artifact_path.name,
            annotation_root.parents[1] / artifact_path
            if len(annotation_root.parents) > 1
            else artifact_path,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    candidates = sorted(episode_root.glob("shard_*/rank_*/training_samples*.jsonl"))
    return candidates[0].resolve() if len(candidates) == 1 else None
