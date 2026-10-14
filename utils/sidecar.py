"""Build and validate lightweight mixed-pretraining eligibility/statistics sidecars."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from utils.vision_text import canonicalize_future_difference_target
from utils.canonical import (
    CANONICAL_SCHEMA,
    canonical_droid_arrays,
    canonical_molmo_arrays,
    canonical_rh20t_arrays,
)


SIDECAR_FORMAT_VERSION = 2
MANIFEST_NAME = "manifest.json"
INCOMPLETE_MARKER_NAME = ".stage05_sidecar_incomplete"
TEMP_DIRECTORY_PREFIX = ".zr0-stage05-sidecar-incomplete-"
RH20T_NUMERIC_ANOMALY_EPISODES = {403, 1248, 3176, 4350}
DATASET_KINDS = {"droid", "molmo", "rh20t"}
GENERATOR_COMMON_DEPENDENCY_PATHS = (
    "utils/sidecar.py",
    "utils/vision_text.py",
    "utils/pretraining_preparation.py",
)
GENERATOR_JOINT_DEPENDENCY_PATHS = ("utils/canonical.py",)
GENERATOR_DEPENDENCY_PATHS = (
    *GENERATOR_COMMON_DEPENDENCY_PATHS,
    *GENERATOR_JOINT_DEPENDENCY_PATHS,
)


def generator_identity(
    repository_root: Path | str | None = None, *, build_joint: bool = True
) -> dict[str, Any]:
    """Fingerprint every repository dependency that changes sidecar contents."""
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    dependencies = []
    paths = GENERATOR_COMMON_DEPENDENCY_PATHS + (
        GENERATOR_JOINT_DEPENDENCY_PATHS if build_joint else ()
    )
    for relative_path in paths:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing Stage05 generator dependency: {path}")
        digest = sha256_file(path)
        dependencies.append({"relative_path": relative_path, "sha256": digest})
    return {
        "scope": "joint" if build_joint else "ar_only",
        "dependencies": dependencies,
        "sha256": canonical_json_hash(
            {"build_joint": build_joint, "dependencies": dependencies}
        ),
    }


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_inventory(root: Path, *, embedded_images: bool) -> dict[str, Any]:
    """Fingerprint metadata by content and large payloads by immutable inventory."""
    metadata = []
    for path in sorted((root / "meta").glob("**/*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
            record = {"path": relative, "size": stat.st_size}
            if relative.startswith("meta/episodes/") or relative.endswith("steps_data_index.pkl"):
                record["mtime_ns"] = stat.st_mtime_ns
            else:
                record["sha256"] = sha256_file(path)
            metadata.append(record)
    payload = []
    patterns = ["data/**/*.parquet"]
    if not embedded_images:
        patterns.extend(
            [
                "videos/observation.images.exterior_1_left/**/*.mp4",
                "videos/observation.images.wrist_left/**/*.mp4",
            ]
        )
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            stat = path.stat()
            payload.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    inventory = {
        "version": 2,
        "payload_camera_policy": (
            "embedded_images_in_selected_parquet_columns"
            if embedded_images
            else "exterior_1_left_and_wrist_left_only"
        ),
        "metadata": metadata,
        "payload": payload,
    }
    return {"sha256": canonical_json_hash(inventory), **inventory}


def _read_episodes(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no episode metadata under {root}")
    rows = []
    for path in paths:
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        requested = [
            "episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index",
            "data/chunk_index", "data/file_index",
        ]
        for key in (
            "observation.images.exterior_1_left", "observation.images.wrist_left"
        ):
            requested.extend(
                [f"videos/{key}/chunk_index", f"videos/{key}/file_index",
                 f"videos/{key}/from_timestamp", f"videos/{key}/to_timestamp"]
            )
        table = pq.read_table(path, columns=[name for name in requested if name in schema_names])
        for row in table.to_pylist():
            rows.append(row)
    rows.sort(key=lambda row: int(row["dataset_from_index"]))
    previous_to = 0
    seen = set()
    for row in rows:
        episode = int(row["episode_index"])
        start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        if episode in seen or start != previous_to or stop - start != int(row["length"]):
            raise ValueError(f"invalid episode boundary metadata at episode {episode}")
        seen.add(episode)
        previous_to = stop
    return rows


def _task_lookup(root: Path) -> dict[int, str]:
    table = pq.read_table(root / "meta" / "tasks.parquet")
    task_column = "task" if "task" in table.column_names else "__index_level_0__"
    return {
        int(row["task_index"]): str(row.get(task_column) or "").strip()
        for row in table.to_pylist()
    }


def _episode_task(row: dict[str, Any]) -> str:
    tasks = row.get("tasks")
    if isinstance(tasks, list):
        return next((str(item).strip() for item in tasks if str(item).strip()), "")
    return str(tasks or "").strip()


def _data_path(root: Path, episode: dict[str, Any]) -> Path:
    return root / "data" / f"chunk-{int(episode['data/chunk_index']):03d}" / f"file-{int(episode['data/file_index']):03d}.parquet"


def _video_path(root: Path, episode: dict[str, Any], key: str) -> Path:
    return root / "videos" / key / f"chunk-{int(episode[f'videos/{key}/chunk_index']):03d}" / f"file-{int(episode[f'videos/{key}/file_index']):03d}.mp4"


def _valid_target(value: Any, identity: str) -> bool:
    try:
        canonicalize_future_difference_target(value, identity)
    except (TypeError, ValueError):
        return False
    return True


def _row_image_present(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get("bytes") or value.get("path"))


def _parquet_column_has_no_nulls(parquet_file: pq.ParquetFile, column_path: str) -> bool:
    found = False
    for row_group_index in range(parquet_file.num_row_groups):
        row_group = parquet_file.metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            if column.path_in_schema != column_path:
                continue
            found = True
            if column.statistics is None or column.statistics.null_count != 0:
                return False
    return found


def _embedded_validity_by_episode(
    parquet_file: pq.ParquetFile, camera_key: str
) -> dict[int, np.ndarray] | None:
    if _parquet_column_has_no_nulls(parquet_file, f"{camera_key}.bytes"):
        return None
    table = parquet_file.read(columns=["episode_index", "frame_index", camera_key])
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in table.to_pylist():
        grouped.setdefault(int(row["episode_index"]), []).append(row)
    result = {}
    for episode, rows in grouped.items():
        rows.sort(key=lambda row: int(row["frame_index"]))
        result[episode] = np.asarray(
            [_row_image_present(row[camera_key]) for row in rows], dtype=bool
        )
    return result


def _stack(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def _canonical_episode(kind: str, rows: list[dict[str, Any]]):
    if kind == "molmo":
        states, actions = _stack(rows, "state"), _stack(rows, "actions")
        return canonical_molmo_arrays(states, actions)
    elif kind == "droid":
        states = _stack(rows, "observation.state.cartesian_position")
        state_gripper = _stack(rows, "observation.state.gripper_position")
        targets = _stack(rows, "action.cartesian_position")
        target_gripper = _stack(rows, "action.gripper_position")
        return canonical_droid_arrays(states, state_gripper, targets, target_gripper)
    elif kind == "rh20t":
        states, actions = _stack(rows, "observation.state"), _stack(rows, "action")
        valid = np.asarray([bool(row["action.valid"]) for row in rows])
        return canonical_rh20t_arrays(states, actions, valid)
    else:
        raise ValueError(f"unknown Stage05 kind {kind!r}")


def _statistics(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.ndim != 2 or matrix.shape[1] != 7 or not len(matrix):
        raise ValueError("statistics require non-empty [N,7] canonical values")
    if not np.isfinite(matrix).all():
        raise ValueError("statistics input contains NaN/Inf")
    return {
        "count": int(len(matrix)),
        "q01": np.quantile(matrix, 0.01, axis=0).tolist(),
        "q99": np.quantile(matrix, 0.99, axis=0).tolist(),
        "min": matrix.min(axis=0).tolist(),
        "max": matrix.max(axis=0).tolist(),
        "mean": matrix.mean(axis=0).tolist(),
        "std": matrix.std(axis=0).tolist(),
    }


def _new_pair_diagnostic(dimensions: int) -> dict[str, np.ndarray]:
    return {
        "count": np.zeros(dimensions, dtype=np.int64),
        "sum_x": np.zeros(dimensions, dtype=np.float64),
        "sum_y": np.zeros(dimensions, dtype=np.float64),
        "sum_x2": np.zeros(dimensions, dtype=np.float64),
        "sum_y2": np.zeros(dimensions, dtype=np.float64),
        "sum_xy": np.zeros(dimensions, dtype=np.float64),
        "squared_error": np.zeros(dimensions, dtype=np.float64),
        "direction_count": np.zeros(dimensions, dtype=np.int64),
        "direction_match": np.zeros(dimensions, dtype=np.int64),
    }


def _update_pair_diagnostic(stats: dict[str, np.ndarray], left, right) -> None:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != len(stats["count"]):
        raise ValueError("DROID diagnostic pair shapes differ")
    finite = np.isfinite(left) & np.isfinite(right)
    x, y = np.where(finite, left, 0.0), np.where(finite, right, 0.0)
    stats["count"] += finite.sum(axis=0)
    stats["sum_x"] += x.sum(axis=0)
    stats["sum_y"] += y.sum(axis=0)
    stats["sum_x2"] += (x * x).sum(axis=0)
    stats["sum_y2"] += (y * y).sum(axis=0)
    stats["sum_xy"] += (x * y).sum(axis=0)
    stats["squared_error"] += ((x - y) ** 2 * finite).sum(axis=0)
    directional = finite & (np.abs(left) > 1e-9) & (np.abs(right) > 1e-9)
    stats["direction_count"] += directional.sum(axis=0)
    stats["direction_match"] += (directional & (np.sign(left) == np.sign(right))).sum(axis=0)


def _finalize_pair_diagnostic(stats: dict[str, np.ndarray]) -> dict[str, Any]:
    count = stats["count"].astype(np.float64)
    covariance = stats["sum_xy"] - stats["sum_x"] * stats["sum_y"] / np.maximum(count, 1)
    variance_x = stats["sum_x2"] - stats["sum_x"] ** 2 / np.maximum(count, 1)
    variance_y = stats["sum_y2"] - stats["sum_y"] ** 2 / np.maximum(count, 1)
    denominator = np.sqrt(np.maximum(variance_x, 0) * np.maximum(variance_y, 0))
    pearson = np.divide(
        covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator > 0
    )
    direction_count = stats["direction_count"]
    direction_rate = np.divide(
        stats["direction_match"], direction_count,
        out=np.full_like(count, np.nan), where=direction_count > 0,
    )
    rmse = np.sqrt(
        np.divide(
            stats["squared_error"], count,
            out=np.full_like(count, np.nan), where=count > 0,
        )
    )
    def finite_or_none(values: np.ndarray) -> list[float | None]:
        return [float(value) if np.isfinite(value) else None for value in values]

    return {
        "note": "Pearson and direction agreement are diagnostics, not hard eligibility thresholds.",
        "count_by_dimension": stats["count"].tolist(),
        "pearson_by_dimension": finite_or_none(pearson),
        "rmse_by_dimension": finite_or_none(rmse),
        "direction_count_by_dimension": direction_count.tolist(),
        "direction_agreement_by_dimension": finite_or_none(direction_rate),
    }


def _episode_columns(kind: str, embedded_images: bool, build_joint: bool) -> list[str]:
    columns = ["index", "episode_index", "frame_index", "task_index", "train_data", "slot_data"]
    if not build_joint:
        if kind == "rh20t":
            columns += [
                "is_episode_successful", "is_episode_successful_valid",
                "observation.camera_sync.wrist_left.timestamp_offset_ms", "timestamp",
            ]
        elif not embedded_images:
            columns += ["timestamp"]
            if kind == "droid":
                columns += ["language_instruction", "language_instruction_2", "language_instruction_3"]
        return list(dict.fromkeys(columns))
    if kind == "molmo":
        columns += ["state", "actions"]
    elif kind == "droid":
        columns += [
            "observation.state.cartesian_position",
            "observation.state.gripper_position",
            "action.cartesian_position",
            "action.cartesian_velocity",
            "action.original",
            "action.gripper_position",
            "action.gripper_velocity",
            "timestamp",
            "language_instruction", "language_instruction_2", "language_instruction_3",
        ]
    else:
        columns += [
            "observation.state", "action", "action.valid",
            "is_episode_successful", "is_episode_successful_valid",
            "observation.camera_sync.wrist_left.timestamp_offset_ms", "timestamp",
        ]
    return list(dict.fromkeys(columns))


def _build_stage05_sidecar_contents(
    *,
    root: Path | str,
    output: Path | str,
    dataset_id: str,
    kind: str,
    embedded_images: bool,
    horizon: int = 32,
    video_backend: str = "pyav",
    build_joint: bool = True,
    failure_injector: Callable[[str, Path], None] | None = None,
) -> dict[str, Any]:
    """Scan annotations and write one complete unpublished sidecar directory."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if kind not in DATASET_KINDS:
        raise ValueError("Stage05 sidecars require a known dataset kind")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Stage05 sidecars require a positive integer horizon")
    output.mkdir(parents=True, exist_ok=True)
    episodes = _read_episodes(root)
    total_frames = int(sum(int(row["length"]) for row in episodes))
    validity = np.zeros((total_frames, 5), dtype=bool)
    filter_reason = np.zeros(total_frames, dtype=np.uint8)
    tasks = _task_lookup(root)
    filters = Counter()
    target_valid_cache: dict[str, bool] = {}
    ar_indices: list[np.ndarray] = []
    joint_indices: list[np.ndarray] = []
    state_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    episode_records = []
    droid_diagnostics = {
        "original_equals_position_and_gripper": 0,
        "candidate_rows": 0,
        "next_state_translation_errors_m": [],
        "next_state_rotation_errors_rad": [],
        "next_state_pairs_total": 0,
        "diagnostic_sample_cap": 1000000,
        "absolute_target_vs_next_state": _new_pair_diagnostic(6),
        "command_delta_vs_next_state_delta": _new_pair_diagnostic(6),
        "cartesian_velocity_vs_next_state_velocity": _new_pair_diagnostic(6),
        "target_gripper_vs_next_state_gripper": _new_pair_diagnostic(1),
        "gripper_velocity_vs_next_state_velocity": _new_pair_diagnostic(1),
    }
    molmo_diagnostics = {
        "stride1_pairs": 0,
        "stride1_pose_matches_at_1e-4": 0,
        "stride2_pairs": 0,
        "stride2_pose_matches_at_1e-4": 0,
        "action_vs_stride1_state_delta": _new_pair_diagnostic(6),
        "action_vs_stride2_state_delta": _new_pair_diagnostic(6),
        "note": "The 1e-4 match rate is trajectory evidence, not a robot tracking-error eligibility gate.",
    }
    wrist_omitted = 0
    main_valid_total = 0

    by_file: dict[Path, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_file.setdefault(_data_path(root, episode), []).append(episode)
    ordered_files = sorted(by_file.items(), key=lambda item: str(item[0]))
    for file_number, (data_path, file_episodes) in enumerate(ordered_files, start=1):
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        progress_interval = max(1, len(ordered_files) // 100)
        if file_number in {1, len(ordered_files)} or file_number % progress_interval == 0:
            print(
                f"[{dataset_id}] scanning parquet {file_number}/{len(ordered_files)}: "
                f"{data_path.relative_to(root)}",
                flush=True,
            )
        parquet_file = pq.ParquetFile(data_path)
        table = parquet_file.read(columns=_episode_columns(kind, embedded_images, build_joint))
        embedded_main_validity = (
            _embedded_validity_by_episode(parquet_file, "first_view")
            if embedded_images
            else None
        )
        embedded_wrist_validity = (
            _embedded_validity_by_episode(parquet_file, "wrist_image")
            if embedded_images
            else None
        )
        file_rows = table.to_pylist()
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in file_rows:
            grouped.setdefault(int(row["episode_index"]), []).append(row)
        for episode_meta in file_episodes:
            episode = int(episode_meta["episode_index"])
            rows = sorted(grouped.get(episode, []), key=lambda row: int(row["frame_index"]))
            if len(rows) != int(episode_meta["length"]):
                raise ValueError(f"episode {episode} row count differs from metadata")
            task = _episode_task(episode_meta)
            if not task and rows:
                task = tasks.get(int(rows[0]["task_index"]), "")
            if kind == "droid" and not task:
                task = next(
                    (
                        str(row.get(key) or "").strip()
                        for row in rows
                        for key in ("language_instruction", "language_instruction_2", "language_instruction_3")
                        if str(row.get(key) or "").strip()
                    ),
                    "",
                )
            trusted_task = bool(task)
            episode_allowed = True
            exclusion = ""
            if kind == "rh20t":
                success_valid = all(bool(row["is_episode_successful_valid"]) for row in rows)
                success = all(bool(row["is_episode_successful"]) for row in rows)
                if episode in RH20T_NUMERIC_ANOMALY_EPISODES:
                    episode_allowed, exclusion = False, "numeric_anomaly_episode"
                elif not success_valid:
                    episode_allowed, exclusion = False, "unrated_episode"
                elif not success:
                    episode_allowed, exclusion = False, "failed_episode"
            if not trusted_task:
                episode_allowed, exclusion = False, "missing_trusted_task"

            if embedded_images:
                main_valid = (
                    np.ones(len(rows), dtype=bool)
                    if embedded_main_validity is None
                    else embedded_main_validity[episode]
                )
                wrist_valid = (
                    np.ones(len(rows), dtype=bool)
                    if embedded_wrist_validity is None
                    else embedded_wrist_validity[episode]
                )
            else:
                main_path = _video_path(root, episode_meta, "observation.images.exterior_1_left")
                wrist_path = _video_path(root, episode_meta, "observation.images.wrist_left")
                main_valid = np.full(len(rows), main_path.is_file(), dtype=bool)
                wrist_valid = np.full(len(rows), wrist_path.is_file(), dtype=bool)
                if kind == "rh20t":
                    synchronized = np.asarray(
                        [abs(int(row["observation.camera_sync.wrist_left.timestamp_offset_ms"])) <= 100 for row in rows]
                    )
                    wrist_valid &= synchronized
            main_valid_total += int(main_valid.sum())
            wrist_omitted += int((main_valid & ~wrist_valid).sum())

            global_indices = np.asarray([int(row["index"]) for row in rows], dtype=np.uint32)
            if len(global_indices) and (
                int(global_indices.min()) < 0 or int(global_indices.max()) >= total_frames
            ):
                raise ValueError("source global indices are outside metadata frame range")
            target_flags = []
            for local_index, row in enumerate(rows):
                raw_target = row.get("train_data")
                cached_target_valid = (
                    target_valid_cache.get(raw_target) if isinstance(raw_target, str) else None
                )
                if cached_target_valid is None:
                    cached_target_valid = _valid_target(
                        raw_target, f"episode={episode} frame={local_index}"
                    )
                    if isinstance(raw_target, str):
                        target_valid_cache[raw_target] = cached_target_valid
                target_flags.append(cached_target_valid)
            target_valid = np.asarray(target_flags, dtype=bool)
            ar_valid = main_valid & target_valid & episode_allowed
            validity[global_indices, 0] = main_valid
            validity[global_indices, 1] = wrist_valid
            validity[global_indices, 2] = target_valid
            ar_indices.append(global_indices[ar_valid])
            filters["ar_eligible"] += int(ar_valid.sum())
            filters["missing_main_view"] += int((~main_valid).sum())
            filters["invalid_or_empty_train_data"] += int((main_valid & ~target_valid).sum())

            joint_valid = np.zeros(len(rows), dtype=bool)
            ar_in_joint = 0
            action_valid_rows = 0
            if build_joint and episode_allowed:
                canonical_states, canonical_actions, state_valid, action_valid = _canonical_episode(kind, rows)
                # Reverse twice so every count covers the forward chunk [t:t+H].
                future_valid = np.convolve(
                    action_valid[::-1].astype(np.int16),
                    np.ones(horizon, dtype=np.int16), mode="full"
                )[: len(rows)][::-1]
                joint_valid = main_valid & state_valid & (future_valid > 0)
                validity[global_indices, 3] = action_valid
                validity[global_indices, 4] = joint_valid
                filters["invalid_canonical_state_or_action"] += int((main_valid & ~state_valid).sum())
                filters["zero_fm_count_chunk"] += int((main_valid & state_valid & (future_valid == 0)).sum())
                state_values.append(canonical_states[joint_valid])
                action_referenced = np.convolve(
                    joint_valid.astype(np.int16),
                    np.ones(horizon, dtype=np.int16),
                    mode="full",
                )[: len(rows)] > 0
                stats_action_valid = action_valid & action_referenced
                action_values.append(canonical_actions[stats_action_valid])
                action_valid_rows = int(stats_action_valid.sum())
                ar_in_joint = int((joint_valid & target_valid).sum())
                joint_indices.append(global_indices[joint_valid])
                filters["joint_eligible"] += int(joint_valid.sum())
                filters["joint_ar_eligible"] += ar_in_joint

            episode_reason = 4 if not episode_allowed else 0
            for local, global_index in enumerate(global_indices):
                code = episode_reason
                if not main_valid[local]:
                    code = 1
                elif not target_valid[local] and code == 0:
                    code = 3
                if build_joint and episode_allowed and not joint_valid[local]:
                    code = 5 if code == 0 else code
                filter_reason[int(global_index)] = code

            if build_joint and kind == "droid" and rows:
                original = _stack(rows, "action.original")
                positions = _stack(rows, "action.cartesian_position")
                grippers = _stack(rows, "action.gripper_position").reshape(-1)
                original_matches = np.all(np.isclose(original[:, :6], positions, atol=1e-7), axis=1) & np.isclose(
                    original[:, 6], grippers, atol=1e-7
                )
                droid_diagnostics["original_equals_position_and_gripper"] += int(original_matches.sum())
                droid_diagnostics["candidate_rows"] += len(rows)
                if len(rows) > 1:
                    current_pose = _stack(rows[:-1], "observation.state.cartesian_position")
                    next_pose = _stack(rows[1:], "observation.state.cartesian_position")
                    command = positions[:-1]
                    command_delta = command - current_pose
                    observed_delta = next_pose - current_pose
                    command_delta[:, 3:6] = (command_delta[:, 3:6] + np.pi) % (2*np.pi) - np.pi
                    observed_delta[:, 3:6] = (observed_delta[:, 3:6] + np.pi) % (2*np.pi) - np.pi
                    timestamps = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
                    delta_t = np.diff(timestamps)
                    positive_dt = np.isfinite(delta_t) & (delta_t > 0)
                    observed_velocity = np.full_like(observed_delta, np.nan)
                    observed_velocity[positive_dt] = observed_delta[positive_dt] / delta_t[positive_dt, None]
                    velocity_command = _stack(rows[:-1], "action.cartesian_velocity")
                    _update_pair_diagnostic(
                        droid_diagnostics["absolute_target_vs_next_state"], command, next_pose
                    )
                    _update_pair_diagnostic(
                        droid_diagnostics["command_delta_vs_next_state_delta"],
                        command_delta, observed_delta,
                    )
                    _update_pair_diagnostic(
                        droid_diagnostics["cartesian_velocity_vs_next_state_velocity"],
                        velocity_command, observed_velocity,
                    )
                    current_gripper = _stack(rows[:-1], "observation.state.gripper_position").reshape(-1, 1)
                    next_gripper = _stack(rows[1:], "observation.state.gripper_position").reshape(-1, 1)
                    target_gripper = _stack(rows[:-1], "action.gripper_position").reshape(-1, 1)
                    observed_gripper_velocity = np.full_like(next_gripper, np.nan)
                    observed_gripper_velocity[positive_dt] = (
                        (next_gripper - current_gripper)[positive_dt] / delta_t[positive_dt, None]
                    )
                    _update_pair_diagnostic(
                        droid_diagnostics["target_gripper_vs_next_state_gripper"],
                        target_gripper, next_gripper,
                    )
                    _update_pair_diagnostic(
                        droid_diagnostics["gripper_velocity_vs_next_state_velocity"],
                        _stack(rows[:-1], "action.gripper_velocity").reshape(-1, 1),
                        observed_gripper_velocity,
                    )
                    droid_diagnostics["next_state_pairs_total"] += len(next_pose)
                    remaining = max(
                        0,
                        droid_diagnostics["diagnostic_sample_cap"]
                        - len(droid_diagnostics["next_state_translation_errors_m"]),
                    )
                    translation_errors = np.linalg.norm(
                        next_pose[:, :3] - command[:, :3], axis=1
                    )[:remaining]
                    droid_diagnostics["next_state_translation_errors_m"].extend(
                        translation_errors.tolist()
                    )
                    wrapped = (next_pose[:, 3:6] - command[:, 3:6] + np.pi) % (2*np.pi) - np.pi
                    droid_diagnostics["next_state_rotation_errors_rad"].extend(
                        np.linalg.norm(wrapped, axis=1)[:remaining].tolist()
                    )
            if build_joint and kind == "molmo" and len(rows) > 1:
                native_states = _stack(rows, "state")
                native_actions = _stack(rows, "actions")
                for stride in (1, 2):
                    if len(rows) <= stride:
                        continue
                    observed = native_states[stride:, :6] - native_states[:-stride, :6]
                    observed[:, 3:6] = (observed[:, 3:6] + np.pi) % (2*np.pi) - np.pi
                    commanded = native_actions[:-stride, :6]
                    errors = np.max(np.abs(commanded - observed), axis=1)
                    prefix = f"stride{stride}"
                    molmo_diagnostics[f"{prefix}_pairs"] += len(errors)
                    molmo_diagnostics[f"{prefix}_pose_matches_at_1e-4"] += int(
                        (errors <= 1e-4).sum()
                    )
                    _update_pair_diagnostic(
                        molmo_diagnostics[f"action_vs_{prefix}_state_delta"],
                        commanded,
                        observed,
                    )
            episode_records.append(
                {
                    "episode_index": episode,
                    "dataset_from_index": int(episode_meta["dataset_from_index"]),
                    "dataset_to_index": int(episode_meta["dataset_to_index"]),
                    "length": len(rows),
                    "task": task,
                    "trusted_task": trusted_task,
                    "episode_allowed": episode_allowed,
                    "exclusion_reason": exclusion,
                    "main_valid_frames": int(main_valid.sum()),
                    "wrist_valid_frames": int((main_valid & wrist_valid).sum()),
                    "ar_eligible_frames": int(ar_valid.sum()),
                    "joint_action_eligible_frames": int(joint_valid.sum()),
                    "joint_ar_eligible_frames": int(ar_in_joint),
                    "valid_action_steps_referenced": int(action_valid_rows),
                }
            )

    ar = np.concatenate(ar_indices) if ar_indices else np.empty(0, dtype=np.uint32)
    np.save(output / "ar_indices.npy", ar, allow_pickle=False)
    if failure_injector is not None:
        failure_injector("after_first_npy", output)
    joint = np.concatenate(joint_indices) if joint_indices else np.empty(0, dtype=np.uint32)
    if build_joint:
        np.save(output / "joint_indices.npy", joint, allow_pickle=False)
    pq.write_table(pa.Table.from_pylist(episode_records), output / "episodes.parquet", compression="zstd")
    np.save(output / "validity_packed.npy", np.packbits(validity, axis=0), allow_pickle=False)
    np.save(output / "filter_reason.npy", filter_reason, allow_pickle=False)

    stats = None
    if build_joint:
        states = np.concatenate(state_values) if state_values else np.empty((0, 7), np.float32)
        actions = np.concatenate(action_values) if action_values else np.empty((0, 7), np.float32)
        stats = {
            "format_version": 1,
            "stats_key": dataset_id,
            "scope": "joint_action_eligible_frames_canonical_valid_elements",
            "statistics": {"state": _statistics(states), "actions": _statistics(actions)},
        }
        stats["content_hash"] = canonical_json_hash(stats)
        (output / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
        array = np.asarray(list(values), dtype=np.float64)
        return {
            "count": int(len(array)),
            "mean": float(array.mean()) if len(array) else None,
            "p50": float(np.quantile(array, 0.5)) if len(array) else None,
            "p95": float(np.quantile(array, 0.95)) if len(array) else None,
            "p99": float(np.quantile(array, 0.99)) if len(array) else None,
        }

    droid_diagnostics["next_state_translation_error_m"] = distribution(
        droid_diagnostics.pop("next_state_translation_errors_m")
    )
    droid_diagnostics["next_state_rotation_error_rad"] = distribution(
        droid_diagnostics.pop("next_state_rotation_errors_rad")
    )
    for key in (
        "absolute_target_vs_next_state",
        "command_delta_vs_next_state_delta",
        "cartesian_velocity_vs_next_state_velocity",
        "target_gripper_vs_next_state_gripper",
        "gripper_velocity_vs_next_state_velocity",
    ):
        droid_diagnostics[key] = _finalize_pair_diagnostic(droid_diagnostics[key])
    for key in ("action_vs_stride1_state_delta", "action_vs_stride2_state_delta"):
        molmo_diagnostics[key] = _finalize_pair_diagnostic(molmo_diagnostics[key])
    for stride in (1, 2):
        pairs = molmo_diagnostics[f"stride{stride}_pairs"]
        matches = molmo_diagnostics[f"stride{stride}_pose_matches_at_1e-4"]
        molmo_diagnostics[f"stride{stride}_pose_match_rate_at_1e-4"] = (
            matches / pairs if pairs else None
        )
    inventory = _source_inventory(root, embedded_images=embedded_images)
    generation = {
        "horizon": horizon,
        "native_fps": json.loads((root / "meta" / "info.json").read_text()).get("fps"),
        "kind": kind,
        "embedded_images": embedded_images,
        "video_backend": video_backend if not embedded_images else None,
        "wrist_max_abs_offset_ms": 100 if kind == "rh20t" else None,
        "second_external_view_read": False,
        "build_joint": build_joint,
        "runtime_dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            **(
                {"scipy": importlib.metadata.version("scipy")}
                if build_joint
                else {}
            ),
        },
    }
    generation["parameters_sha256"] = canonical_json_hash(generation)
    manifest = {
        "sidecar_format_version": SIDECAR_FORMAT_VERSION,
        "dataset_id": dataset_id,
        "dataset_root": str(root),
        "source_codebase_version": json.loads((root / "meta" / "info.json").read_text()).get("codebase_version"),
        "source_inventory_sha256": inventory["sha256"],
        "source_inventory": inventory,
        "generator_identity": generator_identity(build_joint=build_joint),
        "generation": generation,
        "canonical_schema": CANONICAL_SCHEMA if build_joint else None,
        "counts": {
            "source_episodes": len(episodes),
            "source_frames": int(sum(int(row["length"]) for row in episodes)),
            "ar_eligible_frames": int(len(ar)),
            "joint_action_eligible_frames": int(len(joint)) if build_joint else None,
            "joint_ar_eligible_frames": int(filters["joint_ar_eligible"]) if build_joint else None,
            "wrist_omitted_frames": wrist_omitted,
            "wrist_omitted_rate_given_main": wrist_omitted / max(main_valid_total, 1),
        },
        "filter_counts": dict(sorted(filters.items())),
        "droid_conversion_diagnostics": (
            droid_diagnostics if build_joint and kind == "droid" else None
        ),
        "molmo_temporal_diagnostics": (
            molmo_diagnostics if build_joint and kind == "molmo" else None
        ),
        "droid_camera_timestamp_audit": (
            {
                "status": "unavailable_in_release",
                "per_camera_capture_timestamp_fields": False,
                "shared_episode_local_timestamp_field": "timestamp",
                "decoder_alignment": (
                    "each camera-specific packed-video from_timestamp plus the same "
                    "episode-local row timestamp"
                ),
                "wrist_minus_main_offset_ms": None,
                "note": (
                    "Packed-video from/to offsets are independent storage addresses and "
                    "must not be interpreted as inter-camera capture-time skew."
                ),
            }
            if kind == "droid"
            else None
        ),
        "files": {
            "ar_indices.npy": sha256_file(output / "ar_indices.npy"),
            "episodes.parquet": sha256_file(output / "episodes.parquet"),
            "validity_packed.npy": sha256_file(output / "validity_packed.npy"),
            "filter_reason.npy": sha256_file(output / "filter_reason.npy"),
            **({"joint_indices.npy": sha256_file(output / "joint_indices.npy"), "stats.json": sha256_file(output / "stats.json")} if build_joint else {}),
        },
        "validity_columns": [
            "main_camera_valid", "wrist_camera_valid", "train_data_valid",
            "source_action_step_valid", "joint_fm_count_positive",
        ],
        "filter_reason_codes": {
            "0": "admitted_or_objective_specific_no_reason",
            "1": "missing_main_camera",
            "3": "invalid_or_empty_train_data",
            "4": "episode_gate_failed",
            "5": "joint_fm_count_zero_or_invalid_canonical",
        },
    }
    manifest["content_hash"] = canonical_json_hash(manifest)
    if failure_injector is not None:
        failure_injector("before_manifest", output)
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _validate_generation_identity(manifest: dict[str, Any], path: Path) -> None:
    generation = manifest.get("generation")
    if not isinstance(generation, dict):
        raise ValueError(f"Stage05 sidecar generation metadata is missing: {path}")
    recorded = generation.get("parameters_sha256")
    parameters = {
        key: value for key, value in generation.items() if key != "parameters_sha256"
    }
    if recorded != canonical_json_hash(parameters):
        raise ValueError(f"Stage05 sidecar generation parameters are stale: {path}")


def _validate_sidecar_shapes(path: Path, manifest: dict[str, Any]) -> None:
    source_frames = int(manifest["counts"]["source_frames"])
    ar = np.load(path / "ar_indices.npy", mmap_mode="r", allow_pickle=False)
    validity = np.load(path / "validity_packed.npy", mmap_mode="r", allow_pickle=False)
    reasons = np.load(path / "filter_reason.npy", mmap_mode="r", allow_pickle=False)
    if ar.dtype != np.uint32 or ar.ndim != 1 or len(ar) != int(
        manifest["counts"]["ar_eligible_frames"]
    ):
        raise ValueError(f"Stage05 AR index shape/count is invalid: {path}")
    if len(ar) and (int(ar[-1]) >= source_frames or np.any(ar[1:] <= ar[:-1])):
        raise ValueError(f"Stage05 AR indices are invalid or not strictly sorted: {path}")
    expected_validity_shape = ((source_frames + 7) // 8, len(manifest["validity_columns"]))
    if validity.dtype != np.uint8 or validity.shape != expected_validity_shape:
        raise ValueError(f"Stage05 packed validity shape is invalid: {path}")
    if reasons.dtype != np.uint8 or reasons.shape != (source_frames,):
        raise ValueError(f"Stage05 filter-reason shape is invalid: {path}")

    episodes = pq.read_table(path / "episodes.parquet")
    if len(episodes) != int(manifest["counts"]["source_episodes"]):
        raise ValueError(f"Stage05 episode count is invalid: {path}")
    episode_rows = episodes.to_pylist()
    if sum(int(row["length"]) for row in episode_rows) != source_frames:
        raise ValueError(f"Stage05 episode frame count is invalid: {path}")
    if sum(int(row["ar_eligible_frames"]) for row in episode_rows) != len(ar):
        raise ValueError(f"Stage05 episode AR count is invalid: {path}")

    if bool(manifest["generation"]["build_joint"]):
        joint = np.load(path / "joint_indices.npy", mmap_mode="r", allow_pickle=False)
        expected_joint = int(manifest["counts"]["joint_action_eligible_frames"])
        if joint.dtype != np.uint32 or joint.ndim != 1 or len(joint) != expected_joint:
            raise ValueError(f"Stage05 Joint index shape/count is invalid: {path}")
        if len(joint) and (
            int(joint[-1]) >= source_frames or np.any(joint[1:] <= joint[:-1])
        ):
            raise ValueError(f"Stage05 Joint indices are invalid or not strictly sorted: {path}")
        if sum(int(row["joint_action_eligible_frames"]) for row in episode_rows) != len(joint):
            raise ValueError(f"Stage05 episode Joint count is invalid: {path}")
        stats = load_stage05_stats(
            path / "stats.json", expected_stats_key=str(manifest["dataset_id"])
        )
        for key in ("state", "actions"):
            values = stats["statistics"][key]
            if int(values["count"]) <= 0:
                raise ValueError(f"Stage05 {key} statistics are empty: {path}")
            for field in ("q01", "q99", "min", "max", "mean", "std"):
                array = np.asarray(values[field])
                if array.shape != (7,) or not np.isfinite(array).all():
                    raise ValueError(
                        f"Stage05 {key}.{field} statistics are invalid: {path}"
                    )


def load_stage05_sidecar(
    path: Path | str,
    *,
    verify_source: bool = True,
    expected_generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(path).resolve()
    try:
        manifest = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"failed to read Stage05 sidecar manifest: {path}") from error
    if manifest.get("sidecar_format_version") != SIDECAR_FORMAT_VERSION:
        raise ValueError(
            "Stage05 sidecar format version mismatch: "
            f"expected {SIDECAR_FORMAT_VERSION}, got "
            f"{manifest.get('sidecar_format_version')!r}: {path}"
        )
    expected = manifest.get("content_hash")
    actual = canonical_json_hash(
        {key: value for key, value in manifest.items() if key != "content_hash"}
    )
    if expected != actual:
        raise ValueError(f"Stage05 sidecar manifest hash mismatch: {path}")
    build_joint = bool(manifest.get("generation", {}).get("build_joint"))
    if manifest.get("generator_identity") != generator_identity(build_joint=build_joint):
        raise ValueError(f"Stage05 sidecar generator is stale: {path}")
    _validate_generation_identity(manifest, path)
    if expected_generation is not None:
        for key, expected_value in expected_generation.items():
            if manifest["generation"].get(key) != expected_value:
                raise ValueError(
                    f"Stage05 sidecar generation mismatch for {key}: "
                    f"expected {expected_value!r}, got "
                    f"{manifest['generation'].get(key)!r}: {path}"
                )
    for name, digest in manifest["files"].items():
        file_path = path / name
        if not file_path.is_file() or sha256_file(file_path) != digest:
            raise ValueError(f"Stage05 sidecar file missing or corrupt: {file_path}")
    if verify_source:
        current = _source_inventory(
            Path(manifest["dataset_root"]),
            embedded_images=bool(manifest["generation"]["embedded_images"]),
        )["sha256"]
        if current != manifest["source_inventory_sha256"]:
            raise ValueError(f"Stage05 sidecar source is stale: {path}")
    _validate_sidecar_shapes(path, manifest)
    return manifest


def _cleanup_owned_incomplete_directory(path: Path, output_name: str) -> None:
    expected_prefix = f"{TEMP_DIRECTORY_PREFIX}{output_name}-"
    marker = path / INCOMPLETE_MARKER_NAME
    if (
        path.parent.is_dir()
        and path.name.startswith(expected_prefix)
        and marker.is_file()
    ):
        shutil.rmtree(path)


def build_stage05_sidecar(
    *,
    root: Path | str,
    output: Path | str,
    dataset_id: str,
    kind: str,
    embedded_images: bool,
    horizon: int = 32,
    video_backend: str = "pyav",
    build_joint: bool = True,
    _failure_injector: Callable[[str, Path], None] | None = None,
) -> dict[str, Any]:
    """Build, validate, and atomically publish one Stage05 sidecar."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing sidecar {output}")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f"{TEMP_DIRECTORY_PREFIX}{output.name}-", dir=output.parent
        )
    )
    marker = temporary / INCOMPLETE_MARKER_NAME
    marker.write_text("unpublished Stage05 sidecar\n", encoding="utf-8")
    try:
        manifest = _build_stage05_sidecar_contents(
            root=root,
            output=temporary,
            dataset_id=dataset_id,
            kind=kind,
            embedded_images=embedded_images,
            horizon=horizon,
            video_backend=video_backend,
            build_joint=build_joint,
            failure_injector=_failure_injector,
        )
        if _failure_injector is not None:
            _failure_injector("before_validation", temporary)
        validated = load_stage05_sidecar(temporary, verify_source=True)
        if validated["content_hash"] != manifest["content_hash"]:
            raise ValueError("Stage05 unpublished sidecar validation changed its identity")
        if _failure_injector is not None:
            _failure_injector("before_publish", temporary)
        marker.unlink()
        try:
            os.rename(temporary, output)
        except Exception:
            if temporary.is_dir():
                marker.write_text("unpublished Stage05 sidecar\n", encoding="utf-8")
            raise
        return manifest
    except Exception:
        _cleanup_owned_incomplete_directory(temporary, output.name)
        raise


def load_stage05_stats(path: Path | str, *, expected_stats_key: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    expected = payload.pop("content_hash", None)
    actual = canonical_json_hash(payload)
    payload["content_hash"] = expected
    if expected != actual:
        raise ValueError(f"Stage05 statistics hash mismatch: {path}")
    if payload.get("stats_key") != expected_stats_key:
        raise ValueError(
            f"Stage05 stats_key mismatch: expected {expected_stats_key!r}, got {payload.get('stats_key')!r}"
        )
    return payload
