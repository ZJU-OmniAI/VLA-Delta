"""Read-only source alignment, fixed train statistics, and dataset integration."""

from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from utils.annotation_io import _load_episode_mapping, _resolve_annotation_root, _active_training_samples, _data_paths
from utils.optical_flow_checkpoint import json_hash
from utils.slot_labels import VOCABULARIES, normalize_slot_labels


def slot_index_implementation_identity():
    root = Path(__file__).resolve().parents[1]
    paths = ("utils/slot_labels.py", "utils/slot_supervision.py", "utils/slot_config.py", "utils/pretraining_preparation.py")
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def class_weights(counts, max_ratio):
    counts = np.asarray(counts, dtype=np.float64)
    if not counts.sum():
        raise ValueError("training classification statistics have no supervision")
    weights = 1 / np.sqrt(np.maximum(counts, 1))
    weights = np.minimum(weights, weights.min() * max_ratio)
    return (weights / (weights * counts).sum() * counts.sum()).tolist()


def audit_slots(dataset_root, config, *, annotation_root=None, progress=None, dataset_identity=None, camera="first_view",
                allow_unannotated_episodes=False):
    root = Path(dataset_root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("splits") != {"train": f"0:{info['total_episodes']}"}:
        raise ValueError("Slot v1 audit requires the declared full train split; no validation/test statistics allowed")
    mapping = _load_episode_mapping(root)
    source_root, _ = _resolve_annotation_root(root, annotation_root)
    if source_root is None:
        raise ValueError("source annotation root unavailable")
    unannotated = set()
    if allow_unannotated_episodes:
        merge = json.loads((root / "meta/stage05_merge.json").read_text())
        if merge.get("missing_stage05_policy") != "null":
            raise ValueError("unannotated Slot episodes require declared null merge policy")
    anchors, source_hashes, intervals = {}, {}, []
    schemas, cameras, coordinates = Counter(), Counter(), Counter()
    for episode, entry in mapping.items():
        path = _active_training_samples(source_root, int(entry["old_episode_index"]))
        if path is None:
            if allow_unannotated_episodes and not (source_root / f"episode_{int(entry['old_episode_index']):06d}").exists():
                unannotated.add(episode)
                continue
            raise ValueError(f"episode={episode}: missing active source annotations")
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            record = json.loads(line)
            base = record["base_data"]
            frame, end = base["semantic_anchor_frame"], base["next_keyframe_frame"]
            if type(frame) is not int or type(end) is not int or end <= frame:
                raise ValueError(f"episode={episode}: invalid semantic interval")
            if (base["schema_version"] != config.slot_schema_version or base["camera_key"] != camera or base["fps"] != info["fps"]
                    or (dataset_identity is not None and base.get("dataset_id") != dataset_identity)):
                raise ValueError(f"episode={episode} frame={frame}: schema/camera/FPS conflict")
            if base["axis_convention"]["unit"] != "meter" or base["coordinate_frame"] != "per_keyframe_pinhole_camera":
                raise ValueError("Q5 coordinate contract conflict")
            key = (episode, frame)
            if key in anchors:
                raise ValueError(f"duplicate source anchor {key}")
            anchors[key] = {"sha256": json_hash(record["slot_data"]), "end_frame": end}
            intervals.append(end - frame)
            schemas[base["schema_version"]] += 1
            cameras[base["camera_key"]] += 1
            coordinates[base["coordinate_frame"]] += 1
    classes = {q: np.zeros(len(vocab), dtype=np.int64) for q, vocab in VOCABULARIES.items()}
    full_classes = {q: Counter() for q in VOCABULARIES}
    vectors = [[], [], []]
    coverage, full_coverage, types, mask_patterns, q9_positive = Counter(), Counter(), Counter(), Counter(), np.zeros(3, dtype=np.int64)
    q9_negative, invalid_masks, monotonic_violations, max_residual = 0, 0, 0, 0.
    risk_values, bbox_values, point_values = [], [], []
    seen, rows_count, fields = set(), 0, Counter()
    data_hashes = {}
    for path in _data_paths(root, info):
        # Hash only supervised columns, avoiding multi-GB image payload reads.
        table = pq.read_table(path, columns=["episode_index", "frame_index", "index", "slot_data"])
        digest = hashlib.sha256()
        for row in table.to_pylist():
            rows_count += 1
            raw = row["slot_data"]
            types[type(raw).__name__] += 1
            raw = json.loads(raw) if isinstance(raw, str) else raw
            digest.update(json_hash(row).encode())
            key = (int(row["episode_index"]), int(row["frame_index"]))
            if key[0] in unannotated:
                if raw is not None:
                    raise ValueError(f"episode={key[0]} frame={key[1]}: unverifiable Slot labels in unannotated episode")
                continue
            identity = f"dataset={root.name} episode={key[0]} frame={key[1]}"
            fields.update(k for k in (raw or {}) if k.startswith("query_"))
            if dataset_identity is not None and key not in anchors:
                continue
            labels = normalize_slot_labels(raw, is_anchor=True, identity=identity)
            for q in VOCABULARIES:
                full_classes[q].update(labels[f"slot_{q}"][labels[f"slot_{q}_mask"]].tolist())
            for k, v in labels.items():
                if k.endswith("_mask"):
                    full_coverage[k] += int(v.any())
            for q in range(2, 10):
                item = (raw or {}).get(f"query_{q}") or {}
                m = item.get("valid_mask_t" if q == 6 else "valid_mask")
                for value in m if isinstance(m, list) else [m]:
                    if value is not None and (type(value) not in (int, bool, float) or value not in (0, 1)):
                        invalid_masks += 1
            if key not in anchors:
                continue
            if key in seen or json_hash(raw) != anchors[key]["sha256"]:
                raise ValueError(f"{identity}: duplicate or source/merged Slot mismatch")
            seen.add(key)
            anchors[key]["global_index"] = int(row["index"])
            anchors[key]["valid"] = any(bool(v.any()) for k, v in labels.items() if k.endswith("_mask"))
            for k, value in labels.items():
                if k.endswith("_mask"):
                    coverage[k] += int(value.any())
                    mask_patterns[k + ":" + str(value.tolist())] += 1
            for q in VOCABULARIES:
                classes[q] += np.bincount(labels[f"slot_{q}"][labels[f"slot_{q}_mask"]].numpy(), minlength=len(VOCABULARIES[q]))
            for i in range(3):
                if labels["slot_Q5_mask"][i]:
                    # Preserve source double precision for fixed normalization statistics.
                    name = ("target", "gripper", "relative")[i] + "_displacement_m"
                    vectors[i].append(raw["query_5"][name])
            if labels["slot_Q5_mask"].all():
                a, b, c = (np.array(raw["query_5"][n + "_displacement_m"]) for n in ("target", "gripper", "relative"))
                max_residual = max(max_residual, float(np.max(np.abs(c - (b - a)))))
            q9_positive += (labels["slot_Q9_presence"] * labels["slot_Q9_presence_mask"]).numpy().astype(np.int64)
            q9_negative += int(((labels["slot_Q9_presence"] == 0) & labels["slot_Q9_presence_mask"]).sum())
            risks = labels["slot_Q9_risk"][labels["slot_Q9_risk_mask"]].tolist()
            if risks != sorted(risks, reverse=True):
                raise ValueError(f"{identity}: Q9 risk ordering conflict")
            risk_values.extend(risks)
            point_values.extend(labels["slot_Q6"][labels["slot_Q6_mask"]].flatten().tolist())
            for q in ("Q3", "Q4", "Q9_bbox"):
                bbox_values.extend(labels[f"slot_{q}"][labels[f"slot_{q}_mask"]].flatten().tolist())
            if labels["slot_Q1_mask"].all():
                monotonic_violations += int(labels["slot_Q1"][0] > labels["slot_Q1"][1])
        data_hashes[str(path.relative_to(root))] = digest.hexdigest()
        if progress:
            progress(rows_count)
    if seen != set(anchors) or rows_count != info["total_frames"]:
        raise ValueError("source anchors or published rows are missing")
    arrays = [np.asarray(v, dtype=np.float64) for v in vectors]
    if any(len(a) == 0 for a in arrays):
        raise ValueError("Q5 train statistics missing")
    stats = {"version": 1, "schema_version": config.slot_schema_version, "split": "train", "dataset_root": str(root),
             "classes": {q: {"vocabulary": VOCABULARIES[q], "counts": classes[q].tolist(),
                              "weights": class_weights(classes[q], config.slot_class_max_weight_ratio)} for q in VOCABULARIES},
             "q5_mean": [a.mean(0).tolist() for a in arrays],
             "q5_std": [np.maximum(a.std(0), config.slot_q5_std_floor).tolist() for a in arrays],
             "q5_count": [len(a) for a in arrays],
             "rules": {"class_weight": "sqrt_inverse_frequency_clip_ratio_then_train_label_mean_one", "max_weight_ratio": config.slot_class_max_weight_ratio,
                       "std_floor_m": config.slot_q5_std_floor, "std_ddof": 0, "time_alignment": "source_semantic_anchor_only",
                       "q5_order": ["target", "gripper", "relative"], "risk_range": [0, 1], "camera": camera},
             "source_identity": json_hash(source_hashes), "data_identity": json_hash(data_hashes)}
    index = {"version": 1, "dataset_root": str(root), "stats_sha256": json_hash(stats),
             "anchors": [[ep, fr, value] for (ep, fr), value in sorted(anchors.items())],
             "source_hashes": source_hashes, "data_hashes": data_hashes}
    if dataset_identity is not None:
        stats["dataset_identity"] = dataset_identity
        index["stats_sha256"] = json_hash(stats)
    if allow_unannotated_episodes:
        stats["unannotated_episodes"] = sorted(unannotated)
        stats["unannotated_policy"] = "absent_source_episode_and_all_merged_slot_labels_null"
        index["stats_sha256"] = json_hash(stats)
    report = {"episodes": len(mapping), "frames": rows_count, "anchors": len(anchors), "carried_labels_masked": rows_count - len(anchors),
              "splits": info["splits"], "slot_storage_types": dict(types), "fields": dict(fields), "schemas": dict(schemas), "cameras": dict(cameras),
              "coordinates": dict(coordinates), "fps": info["fps"], "anchor_valid_samples": dict(coverage), "full_frame_valid_samples": dict(full_coverage),
              "anchor_mask_patterns": dict(mask_patterns), "full_frame_classes": {q: {v: full_classes[q][i] for i, v in enumerate(VOCABULARIES[q])} for q in VOCABULARIES},
              "q9_positive_by_position": q9_positive.tolist(), "q9_explicit_negative": q9_negative,
              "q9_risk_range": [min(risk_values), max(risk_values)], "bbox_range": [min(bbox_values), max(bbox_values)],
              "contact_point_range": [min(point_values), max(point_values)], "invalid_masks": invalid_masks,
              "q5_range_m": [{"min": a.min(0).tolist(), "max": a.max(0).tolist()} for a in arrays],
              "q5_max_consistency_residual_m": max_residual, "progress_monotonic_violations": monotonic_violations,
              "interval_frames": {"min": min(intervals), "max": max(intervals), "p50": float(np.percentile(intervals, 50)), "p90": float(np.percentile(intervals, 90))},
              "stats": stats}
    return report, stats, index


def audit_slot_geometry(dataset_root):
    """Inspect raw positive-mask values, including rejected geometry and padding."""
    from utils.slot_labels import _numeric, _binary
    root = Path(dataset_root)
    info = json.loads((root / "meta/info.json").read_text())
    counts = Counter({"frames": 0, "invalid_positive_bbox": 0, "invalid_positive_contact_point": 0,
                      "invalid_positive_risk": 0, "nonzero_q9_padding": 0, "null_fields": 0,
                      "invalid_masks": 0, "nonfinite_values": 0})
    def visit(value):
        if value is None:
            counts["null_fields"] += 1
        elif isinstance(value, float) and not np.isfinite(value):
            counts["nonfinite_values"] += 1
        elif isinstance(value, dict):
            for v in value.values():
                visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)
    for path in _data_paths(root, info):
        for row in pq.read_table(path, columns=["slot_data"]).to_pylist():
            raw = json.loads(row["slot_data"])
            counts["frames"] += 1
            visit(raw)
            for i in range(2, 10):
                data = raw[f"query_{i}"]
                mask = data.get("valid_mask_t" if i == 6 else "valid_mask")
                values = mask if isinstance(mask, list) else [mask]
                counts["invalid_masks"] += sum(not _binary(v) for v in values)
            for q, prefix in ((3, "target"), (4, "gripper")):
                data = raw[f"query_{q}"]
                for j, suffix in enumerate(("t", "tK")):
                    if data["valid_mask"][j] == 1:
                        counts["invalid_positive_bbox"] += not _numeric(data[f"{prefix}_bbox_{suffix}"], width=4, bounded=True, bbox=True)
            data = raw["query_6"]
            for j, valid in enumerate(data["valid_mask_t"]):
                if valid == 1:
                    counts["invalid_positive_contact_point"] += not _numeric(data["affordance_contact_points"][j], width=2, bounded=True)
            data = raw["query_9"]
            for j, valid in enumerate(data["valid_mask"]):
                obstacle = data["obstacles"][j]
                if valid == 1:
                    counts["invalid_positive_bbox"] += not _numeric(obstacle["bbox"], width=4, bounded=True, bbox=True)
                    counts["invalid_positive_risk"] += not _numeric(obstacle["risk_score"], bounded=True)
                elif valid == 0:
                    counts["nonzero_q9_padding"] += obstacle["bbox"] != [0, 0, 0, 0] or obstacle["risk_score"] != 0
    return dict(counts)


class SlotSupervisionReader:
    def __init__(self, directory, *, audit_cache=None):
        root = Path(directory)
        self.audit_cache = audit_cache
        if audit_cache is not None:
            for name in ("slot_supervision_stats.json", "slot_anchor_index.json"):
                audit_cache.check(root / name)
        self.stats = json.loads((root / "slot_supervision_stats.json").read_text())
        index = json.loads((root / "slot_anchor_index.json").read_text())
        if index["stats_sha256"] != json_hash(self.stats):
            raise ValueError("Slot index/statistics mismatch")
        self.root = Path(index["dataset_root"])
        if str(self.root) != self.stats["dataset_root"] or index.get("version") not in (1, 2):
            raise ValueError("Slot index dataset/version mismatch")
        if json_hash(index["source_hashes"]) != self.stats["source_identity"] or json_hash(index["data_hashes"]) != self.stats["data_identity"]:
            raise ValueError("Slot audit source/data identity mismatch")
        self.anchors = {(ep, fr): value for ep, fr, value in index["anchors"]}
        if len(self.anchors) != len(index["anchors"]):
            raise ValueError("duplicate Slot anchor index")
        self.mapping = _load_episode_mapping(self.root)
        source_root, _ = _resolve_annotation_root(
            self.root, self.stats.get("anchor_contract", {}).get("annotation_root"))
        if source_root is None:
            raise ValueError("Slot source annotations unavailable for anchor verification")
        self.source_root, self.source_hashes = source_root, index["source_hashes"]
        self.unannotated = set(self.stats.get("unannotated_episodes", []))
        if self.unannotated and (self.stats.get("unannotated_policy") != "absent_source_episode_and_all_merged_slot_labels_null"
                or not self.unannotated <= self.mapping.keys()
                or any(ep in self.unannotated for ep, _ in self.anchors)):
            raise ValueError("invalid unannotated Slot episode contract")
        self.anchors_by_episode = {}
        for key in self.anchors:
            self.anchors_by_episode.setdefault(key[0], set()).add(key)
        if not self.anchors_by_episode.keys() <= self.mapping.keys():
            raise ValueError("Slot index contains non-anchor supervision")
        self.verified_episodes = OrderedDict()
        self.lazy_sources = index["version"] == 2
        if self.lazy_sources and audit_cache is None:
            from utils.aux_data_contract import digest_file
            expected = {"anchors_sha256": json_hash(index["anchors"]),
                        "mapping_sha256": digest_file(self.root / "meta/stage05_episode_mapping.jsonl"),
                        "annotation_root": str(source_root)}
            if self.stats.get("anchor_contract") != expected:
                raise ValueError("Slot anchor/mapping/source-root contract mismatch")
            if self.stats.get("index_implementation_identity") != slot_index_implementation_identity():
                raise ValueError("Slot index implementation identity changed")
        elif not self.lazy_sources:
            for episode in self.mapping:
                self._verify_source_episode(episode)
        self.cache = OrderedDict()

    def _verify_source_episode(self, episode):
        if episode in self.verified_episodes:
            self.verified_episodes.move_to_end(episode)
            return
        if episode in self.unannotated:
            source = self.source_root / f"episode_{int(self.mapping[episode]['old_episode_index']):06d}"
            if source.exists():
                raise ValueError(f"previously unannotated Slot source changed: {source}")
            return
        path = _active_training_samples(self.source_root, int(self.mapping[episode]["old_episode_index"]))
        if path is None:
            raise ValueError(f"missing Slot source episode={episode}")
        if self.audit_cache is not None:
            if str(path) not in self.source_hashes:
                raise ValueError(f"Slot source is absent from the saved alignment audit: {path}")
            self.audit_cache.check(path, self.source_hashes[str(path)])
            return
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != self.source_hashes.get(str(path)):
            raise ValueError(f"Slot source changed since audit: {path}")
        verified = set()
        for line in content.splitlines():
            record = json.loads(line)
            base = record["base_data"]
            key = (episode, base["semantic_anchor_frame"])
            saved = self.anchors.get(key)
            if key in verified or saved is None or saved["end_frame"] != base["next_keyframe_frame"] or saved["sha256"] != json_hash(record["slot_data"]):
                raise ValueError(f"Slot index/source alignment conflict: {key}")
            verified.add(key)
        if verified != self.anchors_by_episode.get(episode, set()):
            raise ValueError("Slot index contains non-anchor supervision")
        self.verified_episodes[episode] = True
        while len(self.verified_episodes) > 4:
            self.verified_episodes.popitem(last=False)

    def read(self, episode, frame):
        if self.lazy_sources:
            self._verify_source_episode(episode)
        key = (episode, frame)
        if key not in self.anchors and episode not in self.unannotated:
            return normalize_slot_labels(None, is_anchor=False)
        path = self.root / self.mapping[episode]["source_data_uri"]
        if self.audit_cache is not None:
            self.audit_cache.check(path)
        if path not in self.cache:
            rows = pq.read_table(path, columns=["episode_index", "frame_index", "slot_data"]).to_pylist()
            self.cache[path] = {(r["episode_index"], r["frame_index"]): r["slot_data"] for r in rows}
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        raw = self.cache[path][key]
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if episode in self.unannotated:
            if parsed is not None:
                raise ValueError(f"unannotated Slot labels changed: episode={episode} frame={frame}")
            return normalize_slot_labels(None, is_anchor=False)
        identity = f"dataset={self.root.name} episode={episode} frame={frame}"
        if json_hash(parsed) != self.anchors[key]["sha256"]:
            raise ValueError(f"{identity}: Slot labels changed since alignment audit")
        return normalize_slot_labels(parsed, is_anchor=True, identity=identity)


class SlotSupervisedDataset(torch.utils.data.Dataset):
    """Attach labels after tokenization and filter Stage 2 by audited eligibility."""
    def __init__(self, dataset, reader, config, flow_config=None):
        if hasattr(reader, "for_dataset"):
            reader, self.slot_dataset_index = reader.for_dataset(dataset.spec)
        else:
            self.slot_dataset_index = None
        self.dataset, self.reader = dataset, reader
        self.spec, self.requirements = dataset.spec, dataset.requirements
        self.unfiltered_length = len(dataset)
        if hasattr(dataset, "natural_mix_block_size"):
            self.natural_mix_block_size = dataset.natural_mix_block_size
        if hasattr(dataset, "manifest"):
            self.manifest = dataset.manifest
        self.matched = Path(self.spec.dataset_path).resolve() == reader.root
        if not self.matched:
            raise ValueError(f"Slot route mismatch: {self.spec.dataset_path} != {reader.root}")
        if getattr(dataset, "aux_dataset_identity", None) is not None and reader.stats.get("dataset_identity") != dataset.aux_dataset_identity:
            raise ValueError("Slot route registered dataset identity mismatch")
        if hasattr(reader, "stats"):
            from dataclasses import replace
            auxiliary = dict(getattr(self.spec, "auxiliary_contract", None) or {})
            auxiliary["slot"] = {"stats_sha256": json_hash(reader.stats), "sampling": config.stage2_aux_sampling}
            self.spec = replace(self.spec, auxiliary_contract=auxiliary)
        self.anchor_by_global = ({value["global_index"]: key for key, value in reader.anchors.items()}
                                 if dataset.loss_type == "aux" and self.matched and hasattr(dataset, "_episode_for_global") else {})
        cached_selection = bool(getattr(dataset, "frozen_index", False) and getattr(reader, "audit_cache", None) is not None)
        self.indices = range(len(dataset)) if dataset.loss_type != "aux" or cached_selection else []
        flow_reader = getattr(dataset, "flow_reader", None)
        flow_eligible = set()
        if dataset.loss_type == "aux" and config.stage2_aux_sampling == "any_aux_valid" and flow_reader is not None:
            if flow_config is None or not flow_config.enabled:
                raise ValueError("Flow sampling requires its enabled configuration")
            from utils.aux_sampling import load_flow_candidates
            flow_eligible = set(load_flow_candidates(dataset).tolist())
        for i in range(len(dataset)) if dataset.loss_type == "aux" and not cached_selection else ():
            ep, frame = self.identity(i)
            valid = self.matched and reader.anchors.get((ep, frame), {}).get("valid", False)
            if dataset.loss_type == "aux":
                flow_valid = int(dataset.indices[i]) in flow_eligible if hasattr(dataset, "indices") else False
                if not valid and not (config.stage2_aux_sampling == "any_aux_valid" and flow_valid):
                    continue
            self.indices.append(i)
        if getattr(dataset, "frozen_index", False) and len(self.indices) != len(dataset):
            raise ValueError("Slot eligibility changed after stage index was frozen")
        if hasattr(reader, "stats"):
            auxiliary = dict(self.spec.auxiliary_contract)
            auxiliary["sampling"] = {"unfiltered_length": self.unfiltered_length, "filtered_length": len(self.indices)}
            if flow_eligible:
                from utils.aux_data_contract import digest_file
                auxiliary["sampling"]["flow_candidates_sha256"] = digest_file(dataset.entry["flow_candidate_index"])
            self.spec = replace(self.spec, auxiliary_contract=auxiliary)

    def identity(self, index):
        if hasattr(self.dataset, "steps"):
            return self.dataset.steps[int(self.dataset.subset_indices[index])]
        if not hasattr(self.dataset, "_episode_for_global"):
            return -1, index
        global_index = int(self.dataset.indices[index])
        episode, frame = self.dataset._episode_for_global(global_index)
        if self.matched:
            # A row offset is not a frame number in sparse/reordered episodes.
            return self.anchor_by_global.get(global_index, (int(episode["episode_index"]), -1))
        return int(episode["episode_index"]), int(frame)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original = self.indices[index]
        sample = self.dataset[original]
        if "episode_id" in sample and "frame_id" in sample:
            ep, frame = int(sample["episode_id"]), int(sample["frame_id"])
        else:
            ep, frame = self.identity(original)
        sample.update(self.reader.read(ep, frame) if self.matched else normalize_slot_labels(None, is_anchor=False))
        sample["slot_anchor"] = torch.tensor((ep, frame) in self.reader.anchors)
        if self.slot_dataset_index is not None:
            sample["slot_dataset_index"] = torch.tensor(self.slot_dataset_index, dtype=torch.long)
        sample.pop("slot_data", None)
        return sample

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def sampling_group_ranges(self):
        original = getattr(self.dataset, "sampling_group_ranges", None)
        if self.dataset.loss_type != "aux" and callable(original):
            return original()
        result, start, previous = [], 0, None
        for i, original in enumerate(self.indices):
            episode, _ = self.identity(original)
            if previous is not None and previous != episode:
                result.append((start, i))
                start = i
            previous = episode
        if self.indices:
            result.append((start, len(self.indices)))
        return result
