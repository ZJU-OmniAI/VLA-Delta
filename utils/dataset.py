"""Sidecar-backed, two-view mixed-pretraining adapter."""

from __future__ import annotations

import io
import importlib.metadata
import json
import platform
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from lerobot.common.datasets.video_utils import decode_video_frames

from utils.vision_text import (
    FUTURE_DIFFERENCE_IMAGE_SIZE,
    canonicalize_future_difference_target,
    measure_future_difference_token_boundary,
    process_vision_info,
    tokenize_future_difference_message,
)
from utils.dataset_spec import resolve_dataset_spec, resolve_objective_requirements
from utils.runtime_config import resolve_pretraining_kind
from utils.normalization import min_max_norm
from utils.canonical import (
    CanonicalChunk,
    canonical_droid_arrays,
    canonical_molmo_arrays,
    canonical_rh20t_arrays,
)
from utils.sidecar import load_stage05_sidecar, load_stage05_stats
from utils.training_tokenization import run_with_same_sample_retries


STAGE05_CAMERA_LABELS = ("main camera", "wrist camera")


def build_stage05_message(
    task: str,
    images: list[tuple[str, Image.Image]],
    target: str | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Stage05 task must be non-empty")
    if len(images) not in (1, 2):
        raise ValueError("Stage05 samples require one main image and optional wrist image")
    if images[0][0] != STAGE05_CAMERA_LABELS[0]:
        raise ValueError("Stage05 first image must be the main camera")
    if len(images) == 2 and images[1][0] != STAGE05_CAMERA_LABELS[1]:
        raise ValueError("Stage05 second image must be the wrist camera")
    content = []
    for label, image in images:
        content.extend(
            [
                {"type": "text", "text": label},
                {
                    "type": "image",
                    "image": image,
                    "resized_height": FUTURE_DIFFERENCE_IMAGE_SIZE,
                    "resized_width": FUTURE_DIFFERENCE_IMAGE_SIZE,
                },
            ]
        )
    content.append({"type": "text", "text": "<TASK> " + task.strip() + " </TASK>"})
    result: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if target is not None:
        result.append({"role": "assistant", "content": target})
    return result


def resolve_stage05_vision_contract(
    processor, *, root: Path | None = None, source_camera_keys: list[str] | None = None
) -> dict[str, Any]:
    image_processor = processor.image_processor
    probes = {}
    for count in (1, 2):
        images = [
            (label, Image.new("RGB", (FUTURE_DIFFERENCE_IMAGE_SIZE,) * 2))
            for label in STAGE05_CAMERA_LABELS[:count]
        ]
        encoded, _ = measure_future_difference_token_boundary(
            build_stage05_message("vision contract probe", images), processor
        )
        grid = encoded["image_grid_thw"].detach().cpu().to(torch.long)
        merge = int(image_processor.merge_size)
        per_view = [int(torch.prod(row).item() // (merge**2)) for row in grid]
        probes[str(count)] = {"per_view_visual_tokens": per_view, "total_visual_tokens": sum(per_view)}
    source_images = []
    if root is not None and source_camera_keys is not None:
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        source_images = [
            {
                "camera_key": key,
                "shape_hwc": list(info["features"][key]["shape"]),
            }
            for key in source_camera_keys
        ]
    return {
        "version": 1,
        "camera_order": list(STAGE05_CAMERA_LABELS),
        "valid_image_counts": [1, 2],
        "source_camera_keys": list(source_camera_keys or []),
        "source_images": source_images,
        "color_mode": "RGB",
        "resize": {
            "height": 224,
            "width": 224,
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
        "processor_measured_visual_tokens": probes,
        "spatial_merge_size": int(image_processor.merge_size),
        "patch_size": int(image_processor.patch_size),
        "temporal_patch_size": int(image_processor.temporal_patch_size),
    }


def prepare_stage05_direct_inputs(*, task: str, images, processor):
    messages = build_stage05_message(task, images, target=None)
    encoded, boundary = measure_future_difference_token_boundary(messages, processor)
    result = {
        "input_ids": encoded["input_ids"][0][: boundary["original_total"]],
        "attention_mask": encoded["attention_mask"][0][: boundary["original_total"]],
        "context_token_count": torch.tensor(boundary["original_total"], dtype=torch.long),
        "sub_task_flag": torch.tensor(0),
    }
    for key in ("pixel_values", "image_grid_thw"):
        if key in encoded:
            result[key] = encoded[key]
    return result


class Stage05MixedPretrainingDataset(torch.utils.data.Dataset):
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
        if (
            isinstance(action_horizon, bool)
            or not isinstance(action_horizon, int)
            or action_horizon <= 0
            or max_pad_state_and_action_length != 64
        ):
            raise ValueError(
                "Stage05 mixed adapter requires a positive integer action horizon "
                "and padded dimension=64"
            )
        if loss_type not in {"vlm", "vlm_and_action", "aux"}:
            raise ValueError("Stage05 mixed adapter supports AR-only or Joint training")
        self.entry = dict(entry)
        self.root = Path(entry["dataset_path"]).resolve()
        sidecar_field = "ar_sidecar_path" if loss_type in {"vlm", "aux"} else "joint_sidecar_path"
        self.sidecar_root = Path(entry.get(sidecar_field, entry.get("sidecar_path", ""))).resolve()
        self.processor = processor
        self.loss_type = loss_type
        self.max_length = int(max_length)
        self.action_horizon = action_horizon
        self.max_pad_length = max_pad_state_and_action_length
        self.dataset_id = int(dataset_id)
        self.aux_dataset_identity = entry.get("aux_dataset_identity")
        self.kind = resolve_pretraining_kind(entry)
        self.embedded_images = bool(entry["embedded_images"])
        self.video_backend = str(entry.get("video_backend", "pyav"))
        self.max_transient_retries = int(entry.get("max_transient_retries", 2))
        self.natural_mix_block_size = 128
        audit_cache = None
        sidecar_loader = load_stage05_sidecar
        sidecar_arguments = {"verify_source": bool(entry.get("verify_sidecar_source", True))}
        if entry.get("preparation_audit_cache"):
            from utils.preparation_audit_cache import load_preparation_audit_cache, load_cached_sidecar
            audit_cache = load_preparation_audit_cache(entry["preparation_audit_cache"])
            sidecar_loader = load_cached_sidecar
            sidecar_arguments = {"audit_cache": audit_cache}
        self.source_horizon = entry.get("sidecar_source_horizon", action_horizon)
        if self.source_horizon != action_horizon and not (
                loss_type == "vlm_and_action" and entry.get("frozen_stage_index")
                and self.source_horizon == 10 and action_horizon == 50):
            raise ValueError("horizon adaptation supports only explicit frozen H10-to-H50 Joint inputs")
        self.manifest = sidecar_loader(
            self.sidecar_root,
            **sidecar_arguments,
            expected_generation={
                "horizon": self.source_horizon,
                "kind": self.kind,
                "embedded_images": self.embedded_images,
                "video_backend": self.video_backend if not self.embedded_images else None,
                "wrist_max_abs_offset_ms": 100 if self.kind == "rh20t" else None,
                "second_external_view_read": False,
                "build_joint": loss_type == "vlm_and_action",
                "runtime_dependencies": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pyarrow": importlib.metadata.version("pyarrow"),
                    **(
                        {"scipy": importlib.metadata.version("scipy")}
                        if loss_type == "vlm_and_action"
                        else {}
                    ),
                },
            },
        )
        if self.manifest["dataset_root"] != str(self.root):
            raise ValueError("Stage05 sidecar dataset root mismatch")
        expected_dataset_id = str(entry.get("stats_key") or "")
        if expected_dataset_id and self.manifest["dataset_id"] != expected_dataset_id:
            raise ValueError("Stage05 sidecar dataset/stats identity mismatch")
        index_name = "ar_indices.npy" if loss_type in {"vlm", "aux"} else "joint_indices.npy"
        if index_name not in self.manifest["files"]:
            raise ValueError(f"Stage05 sidecar does not support {loss_type}: {self.sidecar_root}")
        self.indices = np.load(self.sidecar_root / index_name, mmap_mode="r", allow_pickle=False)
        self.episodes = pq.read_table(self.sidecar_root / "episodes.parquet").to_pylist()
        self.episodes.sort(key=lambda row: int(row["dataset_from_index"]))
        if loss_type == "aux" and not (audit_cache is not None and entry.get("frozen_stage_index")):
            packed = np.load(self.sidecar_root / "validity_packed.npy", mmap_mode="r", allow_pickle=False)
            main_valid = np.unpackbits(packed[:, 0])[:self.manifest["counts"]["source_frames"]]
            self.indices = np.flatnonzero(main_valid)
        self.episode_by_id = {
            int(row["episode_index"]): row for row in self.episodes
        }
        self.episode_starts = np.asarray(
            [int(row["dataset_from_index"]) for row in self.episodes], dtype=np.int64
        )
        self.episode_stops = np.asarray(
            [int(row["dataset_to_index"]) for row in self.episodes], dtype=np.int64
        )
        self.source_episode_metadata = self._load_source_episode_metadata()
        self.flow_reader = None
        if entry.get("flow_enabled", False):
            from utils.optical_flow_reader import OpticalFlowReader
            from utils.aux_data_contract import flow_contract
            flow_root = entry.get("optical_flow_data_root")
            if not flow_root:
                raise ValueError(f"{entry['dataset_entry']}: flow manifest has no data root")
            self.flow_reader = OpticalFlowReader(
                flow_root, entry["optical_flow_manifest"],
                delta_frames=int(entry.get("flow_delta_frames", 20)),
                label_source=int(entry.get("flow_label_source", 1)),
                contract=flow_contract(entry),
            )
            if self.flow_reader.camera_key not in tuple(entry["camera_keys"]):
                raise ValueError(
                    f"{entry['dataset_entry']}: flow camera {self.flow_reader.camera_key!r} "
                    "is absent from the supervised input views"
                )
        self._parquet_cache: OrderedDict[Path, Any] = OrderedDict()
        self._episode_rows_cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self._canonical_cache: OrderedDict[int, tuple] = OrderedDict()
        self.stats = None
        if loss_type == "vlm_and_action":
            payload = load_stage05_stats(
                self.sidecar_root / "stats.json", expected_stats_key=str(entry["stats_key"])
            )
            self.stats = {
                "observation.state": {
                    key: np.asarray(payload["statistics"]["state"][key], dtype=np.float32)
                    for key in ("q01", "q99")
                },
                "action": {
                    key: np.asarray(payload["statistics"]["actions"][key], dtype=np.float32)
                    for key in ("q01", "q99")
                },
            }
        self.requirements = resolve_objective_requirements(
            loss_type,
            adapter="stage05_mixed_pretraining",
            target_text_field="train_data",
            dataset_type="vla",
            dataset_entry=str(entry["dataset_entry"]),
        )
        self.spec = resolve_dataset_spec(
            str(entry["dataset_entry"]), entry, action_horizon=action_horizon, window_size=1,
            requirements=self.requirements,
        )
        if self.flow_reader is not None:
            from utils.aux_data_contract import public_flow_contract
            self.spec = replace(self.spec, auxiliary_contract={"flow": public_flow_contract(self.flow_reader.contract)})
            if loss_type == "aux" and not entry.get("slot_enabled", False):
                from utils.aux_sampling import load_flow_candidates
                self.unfiltered_length = len(self.indices)
                self.indices = np.intersect1d(self.indices, load_flow_candidates(self), assume_unique=True)
                from utils.aux_data_contract import digest_file
                auxiliary = dict(self.spec.auxiliary_contract)
                auxiliary["sampling"] = {"flow_candidates_sha256": digest_file(entry["flow_candidate_index"]),
                    "unfiltered_length": self.unfiltered_length, "filtered_length": len(self.indices)}
                self.spec = replace(self.spec, auxiliary_contract=auxiliary)
        self.spec = replace(
            self.spec,
            vision_input_contract=resolve_stage05_vision_contract(
                processor,
                root=self.root,
                source_camera_keys=list(entry["camera_keys"]),
            ),
        )
        self.frozen_index = bool(entry.get("frozen_stage_index"))
        if self.frozen_index:
            from utils.frozen_stage_index import load_frozen_stage_index
            self.indices, identity = load_frozen_stage_index(entry["frozen_stage_index"],
                dataset_root=self.root, sidecar_root=self.sidecar_root,
                phase="joint" if loss_type == "vlm_and_action" else "ar", audit_cache=audit_cache)
            auxiliary = dict(self.spec.auxiliary_contract or {})
            auxiliary["frozen_stage_index"] = identity
            self.spec = replace(self.spec, auxiliary_contract=auxiliary,
                training_eligibility_source=str(Path(entry["frozen_stage_index"]).resolve()))
            self._frozen_validity = np.load(self.sidecar_root / "validity_packed.npy", mmap_mode="r", allow_pickle=False)

        if self.source_horizon != action_horizon:
            auxiliary = dict(self.spec.auxiliary_contract or {})
            auxiliary["action_horizon_adaptation"] = dict(version=1, source_horizon=10, runtime_horizon=50,
                fm_eligibility="canonical_state_and_any_valid_action_in_runtime_horizon",
                statistics="unchanged_independent_frozen_training_statistics")
            self.spec = replace(self.spec, auxiliary_contract=auxiliary)

    def _load_source_episode_metadata(self) -> dict[int, dict[str, Any]]:
        result = {}
        for path in sorted((self.root / "meta" / "episodes").glob("**/*.parquet")):
            columns = ["episode_index", "data/chunk_index", "data/file_index"]
            if not self.embedded_images:
                for key in (
                    "observation.images.exterior_1_left", "observation.images.wrist_left"
                ):
                    columns.extend(
                        [f"videos/{key}/chunk_index", f"videos/{key}/file_index",
                         f"videos/{key}/from_timestamp", f"videos/{key}/to_timestamp"]
                    )
            for row in pq.read_table(path, columns=columns).to_pylist():
                episode = int(row["episode_index"])
                if episode in result:
                    raise ValueError(f"duplicate source episode metadata {episode}")
                result[episode] = row
        if len(result) != len(self.episodes):
            raise ValueError("source and sidecar episode counts differ")
        return result

    def __len__(self) -> int:
        return int(len(self.indices))

    def set_epoch(self, epoch: int) -> None:
        del epoch

    def sampling_group_ranges(self) -> list[tuple[int, int]]:
        if self.loss_type == "aux" or getattr(self, "frozen_index", False):
            boundaries = np.searchsorted(self.indices, self.episode_stops)
            starts = np.concatenate(([0], boundaries[:-1]))
            return [(int(a), int(b)) for a, b in zip(starts, boundaries) if b > a]
        result = []
        cursor = 0
        count_key = (
            "ar_eligible_frames"
            if self.loss_type == "vlm"
            else "joint_action_eligible_frames"
        )
        for episode in self.episodes:
            count = int(episode[count_key])
            if count:
                result.append((cursor, cursor + count))
                cursor += count
        if cursor != len(self.indices):
            raise ValueError("sidecar episode eligible counts do not cover the index")
        return result

    def _episode_for_global(self, global_index: int) -> tuple[dict[str, Any], int]:
        position = int(np.searchsorted(self.episode_starts, global_index, side="right") - 1)
        if position < 0 or global_index >= self.episode_stops[position]:
            raise IndexError(f"global index {global_index} has no episode")
        return self.episodes[position], global_index - int(self.episode_starts[position])

    def _source_data_path(self, episode: int) -> Path:
        row = self.source_episode_metadata[episode]
        return self.root / "data" / f"chunk-{int(row['data/chunk_index']):03d}" / f"file-{int(row['data/file_index']):03d}.parquet"

    def _columns(self) -> list[str]:
        columns = ["episode_index", "frame_index", "index", "task_index", "train_data", "slot_data"]
        if getattr(self, "frozen_index", False) and self.loss_type == "vlm":
            columns.remove("slot_data")
        if self.embedded_images:
            columns += ["first_view", "wrist_image"]
        else:
            columns += ["timestamp"]
            if self.kind == "rh20t":
                columns += ["observation.camera_sync.wrist_left.timestamp_offset_ms"]
        if self.loss_type == "vlm_and_action":
            if self.kind == "molmo":
                columns += ["state", "actions"]
            elif self.kind == "droid":
                columns += [
                    "observation.state.cartesian_position",
                    "observation.state.gripper_position",
                    "action.cartesian_position",
                    "action.gripper_position",
                ]
            else:
                columns += ["observation.state", "action", "action.valid"]
        return list(dict.fromkeys(columns))

    def _episode_rows(self, episode: int) -> list[dict[str, Any]]:
        cached_rows = self._episode_rows_cache.pop(episode, None)
        if cached_rows is not None:
            self._episode_rows_cache[episode] = cached_rows
            return cached_rows
        path = self._source_data_path(episode)
        table = self._parquet_cache.pop(path, None)
        if table is None:
            table = pq.read_table(path, columns=self._columns())
        self._parquet_cache[path] = table
        while len(self._parquet_cache) > 1:
            self._parquet_cache.popitem(last=False)
        episode_meta = self.episode_by_id[episode]
        global_start = int(episode_meta["dataset_from_index"])
        episode_length = int(episode_meta["dataset_to_index"]) - global_start
        file_first_index = int(table.column("index")[0].as_py())
        offset = global_start - file_first_index
        episode_table = table.slice(offset, episode_length)
        if (
            offset < 0
            or len(episode_table) != episode_length
            or int(episode_table.column("episode_index")[0].as_py()) != episode
            or int(episode_table.column("episode_index")[-1].as_py()) != episode
        ):
            raise ValueError(
                f"episode {episode} is not a contiguous slice of {path}; "
                "source data no longer matches its sidecar"
            )
        rows = episode_table.to_pylist()
        self._episode_rows_cache[episode] = rows
        while len(self._episode_rows_cache) > 2:
            self._episode_rows_cache.popitem(last=False)
        return rows

    def _decode_embedded(self, value: Any, identity: str) -> Image.Image:
        if not isinstance(value, dict):
            raise ValueError(f"{identity}: image value is not an Arrow image struct")
        source = io.BytesIO(value["bytes"]) if value.get("bytes") else self.root / value["path"]
        with Image.open(source) as image:
            return image.convert("RGB")

    def _video_path(self, episode: int, key: str) -> tuple[Path, float]:
        row = self.source_episode_metadata[episode]
        prefix = f"videos/{key}"
        path = self.root / "videos" / key / f"chunk-{int(row[prefix + '/chunk_index']):03d}" / f"file-{int(row[prefix + '/file_index']):03d}.mp4"
        return path, float(row[prefix + "/from_timestamp"])

    def _decode_video(self, episode: int, key: str, timestamp: float) -> Image.Image:
        path, offset = self._video_path(episode, key)
        fps = float(self.manifest["generation"]["native_fps"])
        frame = decode_video_frames(
            path, [offset + timestamp], tolerance_s=max(0.100001, 0.51 / fps), backend=self.video_backend
        )[0]
        array = (frame.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy())
        return Image.fromarray(array, mode="RGB")

    def _images(self, episode: int, row: dict[str, Any], identity: str):
        if self.embedded_images:
            main = self._decode_embedded(row["first_view"], identity)
            raw_wrist = row.get("wrist_image")
            wrist = (
                self._decode_embedded(raw_wrist, identity)
                if isinstance(raw_wrist, dict)
                and bool(raw_wrist.get("bytes") or raw_wrist.get("path"))
                else None
            )
        else:
            timestamp = float(row["timestamp"])
            main = self._decode_video(
                episode, "observation.images.exterior_1_left", timestamp
            )
            wrist = None
            synchronized = self.kind != "rh20t" or abs(
                int(row["observation.camera_sync.wrist_left.timestamp_offset_ms"])
            ) <= 100
            wrist_path, _ = self._video_path(
                episode, "observation.images.wrist_left"
            )
            if synchronized and wrist_path.is_file():
                wrist = self._decode_video(
                    episode, "observation.images.wrist_left", timestamp
                )
        images = [(STAGE05_CAMERA_LABELS[0], main)]
        if wrist is not None:
            images.append((STAGE05_CAMERA_LABELS[1], wrist))
        return images

    def _canonical_chunk(self, episode: int, rows: list[dict[str, Any]], base: int):
        cached = self._canonical_cache.pop(episode, None)
        if cached is None:
            if self.kind == "molmo":
                cached = canonical_molmo_arrays(
                    np.asarray([row["state"] for row in rows]),
                    np.asarray([row["actions"] for row in rows]),
                )
            elif self.kind == "droid":
                cached = canonical_droid_arrays(
                    np.asarray([row["observation.state.cartesian_position"] for row in rows]),
                    np.asarray([row["observation.state.gripper_position"] for row in rows]),
                    np.asarray([row["action.cartesian_position"] for row in rows]),
                    np.asarray([row["action.gripper_position"] for row in rows]),
                )
            else:
                cached = canonical_rh20t_arrays(
                    np.asarray([row["observation.state"] for row in rows]),
                    np.asarray([row["action"] for row in rows]),
                    np.asarray([row["action.valid"] for row in rows]),
                )
        self._canonical_cache[episode] = cached
        while len(self._canonical_cache) > 1:
            self._canonical_cache.popitem(last=False)
        states, actions, state_valid, action_valid = cached
        if not state_valid[base]:
            raise ValueError(f"{self.kind} canonical state is invalid")
        raw_indices = base + np.arange(self.action_horizon, dtype=np.int64)
        temporal_mask = raw_indices < len(rows)
        indices = np.minimum(raw_indices, len(rows) - 1)
        return CanonicalChunk(
            state=states[base],
            action=actions[indices],
            temporal_mask=temporal_mask & action_valid[indices],
            dimension_mask=np.ones(7, dtype=bool),
        )

    def _action_inputs(self, episode: int, rows: list[dict[str, Any]], base: int):
        chunk = self._canonical_chunk(episode, rows, base)
        normalized_state = min_max_norm(
            torch.from_numpy(chunk.state)[None], self.stats["observation.state"], True
        )
        normalized_action = min_max_norm(
            torch.from_numpy(chunk.action), self.stats["action"], True
        )
        state = torch.zeros((1, self.max_pad_length), dtype=torch.float32)
        state[:, :7] = normalized_state
        state_mask = torch.zeros_like(state, dtype=torch.bool)
        state_mask[:, :7] = True
        action = torch.zeros((self.action_horizon, self.max_pad_length), dtype=torch.float32)
        action[:, :7] = normalized_action
        action_mask = torch.from_numpy(chunk.temporal_mask[:, None] & chunk.dimension_mask[None, :])
        padded_mask = torch.zeros_like(action, dtype=torch.bool)
        padded_mask[:, :7] = action_mask
        action.masked_fill_(~padded_mask, 0)
        if not padded_mask.any():
            raise ValueError("Joint sidecar admitted a zero-FM sample")
        return {
            "observation.state": state,
            "state_mask": state_mask,
            "action": action,
            "action_mask": padded_mask,
            "action_supervision_available": torch.tensor(True),
        }

    def _get_item_once(self, global_index: int):
        episode_meta, base = self._episode_for_global(global_index)
        episode = int(episode_meta["episode_index"])
        rows = self._episode_rows(episode)
        row = rows[base]
        identity = f"dataset={self.spec.dataset_entry} episode={episode} frame={base} index={global_index}"
        target = None
        if self.loss_type != "aux" and _is_nonempty(row.get("train_data")):
            try:
                target = canonicalize_future_difference_target(row["train_data"], identity)
            except ValueError:
                if self.loss_type == "vlm":
                    raise
        if self.loss_type == "vlm" and target is None:
            raise ValueError(f"{identity}: AR sidecar admitted an invalid target")
        messages = build_stage05_message(
            str(episode_meta["task"]), self._images(episode, row, identity), target
        )
        result = tokenize_future_difference_message(
            messages, self.processor, self.max_length, identity, target,
            dataset_entry=self.spec.dataset_entry,
        )
        if target is None:
            result["labels"] = torch.full_like(result["input_ids"], -100)
        if self.loss_type == "vlm_and_action":
            fm_available = not self.frozen_index or bool(
                (self._frozen_validity[global_index // 8, 4] >> (7 - global_index % 8)) & 1)
            if fm_available:
                result.update(self._action_inputs(episode, rows, base))
            else:
                result.update({"observation.state": torch.zeros(1, self.max_pad_length),
                    "state_mask": torch.zeros(1, self.max_pad_length, dtype=torch.bool),
                    "action": torch.zeros(self.action_horizon, self.max_pad_length),
                    "action_mask": torch.zeros(self.action_horizon, self.max_pad_length, dtype=torch.bool),
                    "action_supervision_available": torch.tensor(False)})
        result["sub_task_flag"] = torch.tensor(0)
        result["dataset_id"] = torch.tensor(self.dataset_id, dtype=torch.long)
        result["sample_global_index"] = torch.tensor(global_index, dtype=torch.long)
        result["episode_id"] = torch.tensor(episode, dtype=torch.long)
        result["frame_id"] = torch.tensor(int(row["frame_index"]), dtype=torch.long)
        result["ar_eligible"] = torch.tensor(target is not None)
        result["fm_eligible"] = (result.get("action_supervision_available", torch.tensor(False)) if self.frozen_index
                                 else torch.tensor(self.loss_type == "vlm_and_action"))
        result["strict_joint_fm"] = torch.tensor(self.loss_type == "vlm_and_action" and not self.frozen_index)
        if self.source_horizon != self.action_horizon:
            from utils.stage3_horizon_runtime import extend_fm_supervision
            result = extend_fm_supervision(self, global_index, result)
        # Preserve source annotations and routing identity for audits. These fields
        # are collated as metadata lists and are never consumed by either loss.
        result["task"] = str(episode_meta["task"])
        result["train_data"] = target
        if self.loss_type != "aux" and not (self.frozen_index and self.loss_type == "vlm"):
            result["slot_data"] = row.get("slot_data")
        result["stats_key"] = str(self.entry.get("stats_key") or "")
        if self.flow_reader is not None:
            result.update(self.flow_reader.read(episode, int(row["frame_index"])))
        return result

    def __getitem__(self, index: int):
        global_index = int(self.indices[index])
        result, retries = run_with_same_sample_retries(
            lambda: self._get_item_once(global_index),
            dataset_entry=self.spec.dataset_entry,
            sample_id=f"global_index={global_index}",
            max_transient_retries=self.max_transient_retries,
            return_retry_count=True,
        )
        result["data_read_retry_count"] = torch.tensor(retries, dtype=torch.long)
        return result


def _is_nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
