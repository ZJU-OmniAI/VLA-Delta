"""Optional LeRobot v3 future-difference dataset reader."""

from __future__ import annotations

import io
import json
import math
import pickle
import random
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


from utils.vision_text import (
    SUPPORTED_LOSS_TYPES,
    build_future_difference_message,
    canonicalize_future_difference_target,
    encode_future_difference_target_tokens,
    resolve_future_difference_vision_contract,
    tokenize_future_difference_message,
)

from utils.normalization import min_max_norm
from utils.dataset_spec import (
    resolve_dataset_spec,
    resolve_objective_requirements,
)
from utils.training_tokenization import (
    run_with_same_sample_retries,
)


class _ParquetColumnLRU:
    def __init__(self, max_files: int = 8):
        if max_files < 1:
            raise ValueError("max_files must be positive")
        self.max_files = max_files
        self._tables: OrderedDict[tuple[Path, tuple[str, ...]], Any] = OrderedDict()

    def read(self, path: Path, columns: list[str]):
        key = (path, tuple(columns))
        if key in self._tables:
            self._tables.move_to_end(key)
            return self._tables[key]
        table = pq.read_table(path, columns=columns)
        self._tables[key] = table
        if len(self._tables) > self.max_files:
            self._tables.popitem(last=False)
        return table


class LeRobotV3FutureDifferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        entry: dict[str, Any],
        processor,
        loss_type: str,
        max_length: int,
        action_horizon: int = 32,
        max_pad_state_and_action_length: int = 64,
        dataset_id: int = 0,
    ):
        if loss_type not in SUPPORTED_LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {sorted(SUPPORTED_LOSS_TYPES)}")
        if (
            isinstance(action_horizon, bool)
            or not isinstance(action_horizon, int)
            or action_horizon <= 0
        ):
            raise ValueError("action_horizon must be a positive integer")
        if max_pad_state_and_action_length != 64:
            raise ValueError("future-difference adapter requires padded dimension=64")
        self.entry = dict(entry)
        self.root = Path(entry["dataset_path"]).resolve()
        self.processor = processor
        self.loss_type = loss_type
        self.max_length = max_length
        self.action_horizon = action_horizon
        self.max_pad_length = max_pad_state_and_action_length
        self.dataset_id = dataset_id
        self.max_transient_retries = entry.get("max_transient_retries", 2)
        self.requirements = resolve_objective_requirements(
            loss_type,
            adapter="lerobot_v3_future_difference",
            target_text_field=entry.get("target_text_field", "train_data"),
            dataset_type=str(entry.get("dataset_type", "vla")),
            dataset_entry=str(entry.get("dataset_entry", self.root.name)),
        )
        self.spec = resolve_dataset_spec(
            str(entry.get("dataset_entry", self.root.name)),
            entry,
            action_horizon=action_horizon,
            window_size=1,
            requirements=self.requirements,
        )
        self.camera_keys = list(self.spec.camera_keys)
        self.spec = replace(
            self.spec,
            vision_input_contract=resolve_future_difference_vision_contract(
                self.root, self.camera_keys, self.processor
            ),
        )
        self.target_field = self.spec.target_text_field
        self.state_field = self.spec.state_key
        self.action_field = self.spec.action_key
        self.use_quantile = entry.get("use_quantile", True)
        if not self.use_quantile:
            raise ValueError(
                "future-difference adapter requires quantile normalization"
            )
        # One decoded episode with three image columns is roughly 160 MB in the
        # published dataset. Episode-grouped sampling makes a single entry useful
        # while keeping each DataLoader worker's resident set bounded.
        self._cache = _ParquetColumnLRU(max_files=1)
        self._episode_action_rows: OrderedDict[int, dict[int, np.ndarray]] = (
            OrderedDict()
        )
        self.steps = self._load_steps()
        self.episode_sources = self._load_episode_mapping()
        self.tasks = self._load_tasks()
        self._validate_index_metadata()
        self.stats = self.spec.normalization_stats if self._needs_action else None
        self._init_sampled_indices()

    @property
    def _needs_target(self) -> bool:
        return self.requirements.requires_target

    @property
    def _needs_action(self) -> bool:
        return self.requirements.requires_action

    def _load_steps(self) -> list[tuple[int, int]]:
        path = self.root / "meta/steps_data_index.pkl"
        try:
            with path.open("rb") as source:
                metadata = pickle.load(source)
        except Exception as error:
            raise ValueError(
                f"failed to load v3 steps metadata {path}: {error}"
            ) from error
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("steps"), list
        ):
            raise ValueError(f"v3 steps metadata {path} must contain a steps list")
        steps = []
        for sample_index, step in enumerate(metadata["steps"]):
            if not isinstance(step, (tuple, list)) or len(step) != 2:
                raise ValueError(
                    f"v3 steps[{sample_index}] must be an episode/frame pair"
                )
            episode, frame = step
            if not isinstance(episode, (int, np.integer)) or not isinstance(
                frame, (int, np.integer)
            ):
                raise ValueError(
                    f"v3 steps[{sample_index}] episode/frame must be integers"
                )
            if int(episode) < 0 or int(frame) < 0:
                raise ValueError(
                    f"v3 steps[{sample_index}] episode/frame must be non-negative"
                )
            steps.append((int(episode), int(frame)))
        if metadata.get("total_steps", len(steps)) != len(steps):
            raise ValueError("v3 steps total_steps does not match the steps list")
        if not steps:
            raise ValueError("v3 steps list must not be empty")
        num_trajectories = metadata.get("num_trajectories")
        if not isinstance(num_trajectories, int) or num_trajectories != len(
            {episode for episode, _ in steps}
        ):
            raise ValueError("v3 steps num_trajectories does not match unique episodes")
        return steps

    def _load_episode_mapping(self) -> dict[int, Path]:
        path = self.root / "meta/stage05_episode_mapping.jsonl"
        mapping = {}
        try:
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    row = json.loads(line)
                    episode = row["new_episode_index"]
                    old_episode = row["old_episode_index"]
                    uri = row["source_data_uri"]
                    if not isinstance(episode, int) or not isinstance(old_episode, int):
                        raise ValueError(
                            "new_episode_index and old_episode_index must be integers"
                        )
                    if not isinstance(uri, str) or not uri:
                        raise ValueError("invalid source_data_uri")
                    resolved = (self.root / uri).resolve()
                    if self.root not in resolved.parents or not resolved.is_file():
                        raise ValueError(
                            f"source_data_uri is missing or escapes root: {uri}"
                        )
                    if episode in mapping:
                        raise ValueError(f"duplicate new_episode_index {episode}")
                    mapping[episode] = resolved
        except Exception as error:
            raise ValueError(
                f"failed to load v3 episode mapping {path}: {error}"
            ) from error
        if not mapping:
            raise ValueError("v3 episode mapping must not be empty")
        return mapping

    def _load_tasks(self) -> dict[int, str]:
        path = self.root / "meta/tasks.parquet"
        try:
            rows = pq.read_table(path, columns=["task_index", "task"]).to_pylist()
        except Exception as error:
            raise ValueError(f"failed to load v3 tasks {path}: {error}") from error
        tasks = {}
        for row in rows:
            task_index, task = row["task_index"], row["task"]
            if (
                not isinstance(task_index, int)
                or not isinstance(task, str)
                or not task.strip()
            ):
                raise ValueError(f"v3 tasks contains invalid row {row!r}")
            if task_index in tasks:
                raise ValueError(f"v3 tasks contains duplicate task_index {task_index}")
            tasks[task_index] = task
        if not tasks:
            raise ValueError("v3 tasks must not be empty")
        return tasks

    def _validate_index_metadata(self) -> None:
        previous = None
        seen = set()
        for sample_index, (episode, frame) in enumerate(self.steps):
            if episode not in self.episode_sources:
                raise ValueError(
                    f"v3 steps references episode {episode} missing from mapping"
                )
            if (episode, frame) in seen:
                raise ValueError(
                    f"v3 steps contains duplicate episode/frame {(episode, frame)}"
                )
            seen.add((episode, frame))
            if (
                previous is not None
                and episode == previous[0]
                and frame != previous[1] + 1
            ):
                raise ValueError(
                    f"v3 steps are not consecutive at sample {sample_index}"
                )
            previous = (episode, frame)
        step_episodes = {episode for episode, _ in self.steps}
        extra = sorted(set(self.episode_sources) - step_episodes)
        if extra:
            raise ValueError(
                f"v3 episode mapping has episodes absent from steps: {extra[:5]}"
            )

    def _init_sampled_indices(self) -> None:
        ratio = self.entry.get("sample_ratio", 1.0)
        if (
            not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or not 0 < ratio <= 1
        ):
            raise ValueError("sample_ratio must be finite and in (0, 1]")
        count = max(1, int(len(self.steps) * ratio))
        if count == len(self.steps):
            self.subset_indices = list(range(len(self.steps)))
        else:
            self.subset_indices = random.Random(42).sample(
                range(len(self.steps)), count
            )

    def __len__(self) -> int:
        return len(self.subset_indices)

    def set_epoch(self, epoch: int) -> None:
        del epoch

    def sampling_groups(self) -> list[list[int]]:
        """Return dataset-visible indices grouped by episode and ordered by frame."""
        grouped: OrderedDict[int, list[tuple[int, int]]] = OrderedDict()
        for local_index, sample_index in enumerate(self.subset_indices):
            episode, frame = self.steps[sample_index]
            grouped.setdefault(episode, []).append((frame, local_index))
        return [
            [local_index for _, local_index in sorted(items)]
            for items in grouped.values()
        ]

    def _required_columns(self) -> list[str]:
        columns = [*self.camera_keys, "task_index", "episode_index", "frame_index"]
        if self._needs_target:
            columns.append(self.target_field)
        if self._needs_action:
            columns.extend([self.state_field, self.action_field])
        return list(dict.fromkeys(columns))

    def _row_index(self, table, episode: int, frame: int, sample_id: str) -> int:
        episodes = table["episode_index"].to_numpy(zero_copy_only=False)
        frames = table["frame_index"].to_numpy(zero_copy_only=False)
        matches = np.flatnonzero((episodes == episode) & (frames == frame))
        if len(matches) != 1:
            raise ValueError(
                f"{sample_id}: expected exactly one parquet row, found {len(matches)}"
            )
        return int(matches[0])

    def _decode_image(self, value: Any, sample_id: str, camera_key: str) -> Image.Image:
        if not isinstance(value, dict):
            raise ValueError(f"{sample_id}: {camera_key} must be an image struct")
        raw = value.get("bytes")
        if raw is not None:
            source = io.BytesIO(raw)
        else:
            relative = value.get("path")
            if not isinstance(relative, str) or not relative:
                raise ValueError(
                    f"{sample_id}: {camera_key} has neither bytes nor path"
                )
            path = (self.root / relative).resolve()
            if self.root not in path.parents:
                raise ValueError(f"{sample_id}: {camera_key} path escapes dataset root")
            source = path
        try:
            with Image.open(source) as image:
                return image.convert("RGB")
        except Exception as error:
            raise ValueError(
                f"{sample_id}: failed to decode {camera_key}: {error}"
            ) from error

    def _target_token_ids(self, target: str) -> list[int]:
        return encode_future_difference_target_tokens(self.processor.tokenizer, target)

    def _tokenize(self, messages, sample_id: str, target: str | None):
        # Kept as a method so integrity tests and future adapters can instrument it.
        if target is not None and not self._target_token_ids(target):
            raise ValueError(f"{sample_id}: empty target token sequence")
        return tokenize_future_difference_message(
            messages,
            self.processor,
            self.max_length,
            sample_id,
            target,
            dataset_entry=self.spec.dataset_entry,
        )

    def _action_inputs(
        self, table, row_index: int, episode: int, frame: int, sample_id: str
    ):
        state = np.asarray(table[self.state_field][row_index].as_py(), dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError(f"{sample_id}: state must contain 7 finite values")
        normalized_state = min_max_norm(
            torch.from_numpy(state).unsqueeze(0), self.stats["observation.state"], True
        )
        padded_state = torch.zeros(
            (1, self.max_pad_length), dtype=normalized_state.dtype
        )
        padded_state[:, :7] = normalized_state
        state_mask = torch.zeros((1, self.max_pad_length), dtype=torch.bool)
        state_mask[:, :7] = True

        by_frame = self._episode_action_rows.get(episode)
        if by_frame is None:
            episodes = table["episode_index"].to_numpy(zero_copy_only=False)
            frames = table["frame_index"].to_numpy(zero_copy_only=False)
            action_column = table[self.action_field]
            by_frame = {}
            for candidate_index, (candidate_episode, candidate_frame) in enumerate(
                zip(episodes, frames)
            ):
                if int(candidate_episode) != episode:
                    continue
                candidate_frame = int(candidate_frame)
                if candidate_frame in by_frame:
                    raise ValueError(
                        f"{sample_id}: duplicate action frame {candidate_frame}"
                    )
                by_frame[candidate_frame] = np.asarray(
                    action_column[candidate_index].as_py(), dtype=np.float32
                )
            self._episode_action_rows[episode] = by_frame
            self._episode_action_rows.move_to_end(episode)
            if len(self._episode_action_rows) > 1:
                self._episode_action_rows.popitem(last=False)
        raw_actions = []
        for offset in range(self.action_horizon):
            candidate = by_frame.get(frame + offset)
            if candidate is None:
                break
            action = candidate
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError(
                    f"{sample_id}: action at offset {offset} must contain 7 finite values"
                )
            raw_actions.append(action)
        if not raw_actions:
            raise ValueError(f"{sample_id}: current action is missing")
        normalized_actions = min_max_norm(
            torch.from_numpy(np.stack(raw_actions)), self.stats["action"], True
        )
        padded_actions = torch.zeros(
            (self.action_horizon, self.max_pad_length), dtype=normalized_actions.dtype
        )
        padded_actions[: len(raw_actions), :7] = normalized_actions
        temporal_valid = torch.arange(self.action_horizon) < len(raw_actions)
        dimension_valid = torch.arange(self.max_pad_length) < 7
        action_mask = temporal_valid[:, None] & dimension_valid[None, :]
        return {
            "observation.state": padded_state,
            "state_mask": state_mask,
            "action": padded_actions,
            "action_mask": action_mask,
            "action_supervision_available": torch.tensor(True),
        }

    def _get_item_once(self, sample_index: int) -> dict[str, torch.Tensor]:
        episode, frame = self.steps[sample_index]
        sample_id = f"episode={episode} frame={frame} sample={sample_index}"
        table = self._cache.read(
            self.episode_sources[episode], self._required_columns()
        )
        row_index = self._row_index(table, episode, frame, sample_id)
        row = table.slice(row_index, 1).to_pylist()[0]
        task_index = row["task_index"]
        if task_index not in self.tasks:
            raise ValueError(
                f"{sample_id}: task_index {task_index} is missing from tasks"
            )
        images = [
            (key, self._decode_image(row[key], sample_id, key))
            for key in self.camera_keys
        ]
        target = None
        if self._needs_target:
            target = canonicalize_future_difference_target(
                row[self.target_field], sample_id
            )
        messages = build_future_difference_message(
            self.tasks[task_index], images, target
        )
        result = self._tokenize(messages, sample_id, target)
        if self._needs_action:
            result.update(
                self._action_inputs(table, row_index, episode, frame, sample_id)
            )
        result["sub_task_flag"] = torch.tensor(0)
        return result

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample_index = self.subset_indices[index]
        episode, frame = self.steps[sample_index]
        sample_id = f"episode={episode} frame={frame} sample={sample_index}"
        result, retry_count = run_with_same_sample_retries(
            lambda: self._get_item_once(sample_index),
            dataset_entry=self.spec.dataset_entry,
            sample_id=sample_id,
            max_transient_retries=self.max_transient_retries,
            return_retry_count=True,
        )
        result["data_read_retry_count"] = torch.tensor(retry_count, dtype=torch.long)
        return result
