"""Reusable H10 sidecars and frozen sample admission for three-stage pretraining.

Adapted from the repository's original prepare_three_stage_validation phases.
No training or GPU allocation occurs in this module.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from utils.runtime_config import load_mapping, resolve_pretraining_kind
from utils.sidecar import build_stage05_sidecar, load_stage05_sidecar, load_stage05_stats, sha256_file, canonical_json_hash


def write_json(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite preparation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def entries(config):
    registry = load_mapping(config["dataset_registry"])
    output = Path(config["output_root"])
    result = {}
    for key in config["dataset_entries"]:
        entry = dict(registry[key])
        name = entry["aux_dataset_identity"]
        if name in result or not name.replace("_", "").isalnum():
            raise ValueError("auxiliary dataset identities must be unique safe names")
        entry.update(dataset_entry=key, ar_sidecar_path=str(output / name / "ar"),
            joint_sidecar_path=str(output / name / "joint"),
            frozen_stage_index=str(output / name / "frozen_index.json"),
            sidecar_source_horizon=10, max_transient_retries=0)
        result[name] = entry
    return result


def sidecars(config):
    for name, entry in entries(config).items():
        for phase in ("ar", "joint"):
            print(f"Building {name}/{phase}", flush=True)
            build_stage05_sidecar(root=entry["dataset_path"], output=entry[phase + "_sidecar_path"],
                dataset_id=entry["stats_key"], kind=resolve_pretraining_kind(entry), embedded_images=entry["embedded_images"],
                video_backend=entry.get("video_backend", "pyav"), horizon=10, build_joint=phase == "joint")


def sources(config):
    from utils.annotation_io import _load_episode_mapping
    from utils.file_identity import file_identity
    output = Path(config["output_root"])
    for name, entry in entries(config).items():
        root = Path(entry["dataset_path"])
        mapping = _load_episode_mapping(root)
        metadata = [row for path in sorted((root / "meta/episodes").rglob("*.parquet"))
                    for row in pq.read_table(path).to_pylist()]
        paths = set((root / "meta").rglob("*.parquet")) | set((root / "meta").glob("*.json*"))
        episodes = []
        for row in metadata:
            episode = int(row["episode_index"])
            data = root / "data" / f"chunk-{int(row['data/chunk_index']):03d}" / f"file-{int(row['data/file_index']):03d}.parquet"
            cameras = {}
            for camera in entry["camera_keys"]:
                prefix = "videos/" + camera
                cameras[camera] = str(data if entry["embedded_images"] else root / prefix /
                    f"chunk-{int(row[prefix + '/chunk_index']):03d}" / f"file-{int(row[prefix + '/file_index']):03d}.mp4")
            paths.update([data, *map(Path, cameras.values())])
            episodes.append({"episode_index": episode, "original_identity": mapping[episode],
                "frames": int(row["length"]), "split": "train", "data": str(data), "cameras": cameras})
        records = {}
        def identify(path):
            before = path.stat()
            record = file_identity(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"source changed while hashing: {path}")
            return str(path), dict(record, mtime_ns=after.st_mtime_ns)
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i, (path, record) in enumerate(executor.map(identify, sorted(paths))):
                records[path] = record
                if i % 100 == 0:
                    print(f"Source SHA256 {name}: {i + 1}/{len(paths)}", flush=True)
        write_json(output / name / "source_audit.json", {"status": "passed", "dataset_root": str(root),
            "files": records, "episodes": sorted(episodes, key=lambda row: row["episode_index"]),
            "camera_order": entry["camera_keys"], "source_hash_algorithm": "sha256_entire_file"})


def slots(config):
    from utils.slot_config import SlotConfig
    from utils.slot_supervision import audit_slots, slot_index_implementation_identity
    from utils.annotation_io import _resolve_annotation_root
    output = Path(config["output_root"])
    routes = {}
    for name, entry in entries(config).items():
        root = Path(entry["dataset_path"])
        directory = output / name / "slot"
        directory.mkdir(exist_ok=False)
        _, stats, index = audit_slots(root, SlotConfig(), annotation_root=entry.get("annotation_root"), dataset_identity=entry["aux_dataset_identity"],
            camera=entry["camera_keys"][0], allow_unannotated_episodes=entry.get("allow_unannotated_episodes", False),
            progress=lambda n: print(f"Slot {name}: {n}", flush=True))
        source_root, _ = _resolve_annotation_root(root, entry.get("annotation_root"))
        stats.update(anchor_contract={"anchors_sha256": canonical_json_hash(index["anchors"]),
            "mapping_sha256": sha256_file(root / "meta/stage05_episode_mapping.jsonl"), "annotation_root": str(source_root)},
            index_implementation_identity=slot_index_implementation_identity())
        index.update(version=2, stats_sha256=canonical_json_hash(stats))
        write_json(directory / "slot_supervision_stats.json", stats)
        write_json(directory / "slot_anchor_index.json", index)
        routes[entry["aux_dataset_identity"]] = f"{name}/slot"
    write_json(output / "slot_routes.json", {"version": 1, "datasets": routes})


def freeze(config):
    output = Path(config["output_root"])
    for name, entry in entries(config).items():
        directory = output / name
        manifests = {phase: load_stage05_sidecar(entry[phase + "_sidecar_path"], verify_source=True)
                     for phase in ("ar", "joint")}
        source_audit = json.loads((directory / "source_audit.json").read_text())
        if (source_audit["dataset_root"] != entry["dataset_path"]
                or any(manifest["dataset_root"] != entry["dataset_path"] for manifest in manifests.values())):
            raise ValueError("source/sidecar dataset root changed during preparation")
        manifest = manifests["joint"]
        n = manifest["counts"]["source_frames"]
        packed = np.load(directory / "joint/validity_packed.npy", mmap_mode="r")
        flags = np.unpackbits(packed, axis=0)[:n].astype(bool)
        quality = np.zeros(n, dtype=bool)
        reasons = {}
        episode_rows = pq.read_table(directory / "joint/episodes.parquet").to_pylist()
        for row in episode_rows:
            quality[row["dataset_from_index"]:row["dataset_to_index"]] = row["episode_allowed"]
            if not row["episode_allowed"]:
                key = row["exclusion_reason"]
                reasons[key] = reasons.get(key, 0) + row["length"]
        slot_index = json.loads((directory / "slot/slot_anchor_index.json").read_text())
        anchors = np.zeros(n, dtype=bool)
        for _, _, value in slot_index["anchors"]:
            if value["valid"]:
                anchors[value["global_index"]] = True
        dual = flags[:, 0] & flags[:, 1]
        selections = {"stage12": quality & dual & flags[:, 2] & anchors, "stage3": quality & dual}
        admissions = []
        for row in episode_rows:
            start, stop = row["dataset_from_index"], row["dataset_to_index"]
            admissions.append({"episode_index": row["episode_index"], "source_frames": row["length"],
                "dataset_from_index": start, "dataset_to_index": stop, "quality_allowed": row["episode_allowed"],
                "exclusion_reason": row["exclusion_reason"], "split": "train",
                **{phase + "_valid_samples": int(valid[start:stop].sum()) for phase, valid in selections.items()}})
        admission_path = directory / "episode_admission.parquet"
        if admission_path.exists():
            raise ValueError(f"refusing overwritten episode admission audit: {admission_path}")
        pq.write_table(pa.Table.from_pylist(admissions), admission_path)
        stat_path = directory / "joint/stats.json"
        stats = load_stage05_stats(stat_path, expected_stats_key=entry["stats_key"])
        report = {"version": 1, "dataset_root": entry["dataset_path"], "source_frames": n,
            "source_audit": {"path": str(directory / "source_audit.json"), "sha256": sha256_file(directory / "source_audit.json")},
            "episode_admission": {"path": str(admission_path), "sha256": sha256_file(admission_path)},
            "quality": int(quality.sum()), "dual_camera": int(dual.sum()), "legal_ar": int(flags[:, 2].sum()),
            "slot_anchor_valid": int(anchors.sum()), "fm_eligible": int(flags[:, 4].sum()),
            "independent_exclusions": {"quality": int((~quality).sum()), "dual_camera": int((~dual).sum()),
                "legal_ar": int((~flags[:, 2]).sum()), "valid_slot_anchor": int((~anchors).sum())},
            "stage12_sequential_exclusions": {"quality": int((~quality).sum()),
                "dual_camera_after_quality": int((quality & ~dual).sum()),
                "ar_after_quality_dual_camera": int((quality & dual & ~flags[:, 2]).sum()),
                "slot_after_quality_dual_camera_ar": int((quality & dual & flags[:, 2] & ~anchors).sum())},
            "excluded_quality": reasons, "missing_wrist_after_quality": int((quality & flags[:, 0] & ~flags[:, 1]).sum()),
            "sidecar_manifest_sha256": {phase: sha256_file(directory / phase / "manifest.json") for phase in manifests},
            "slot_index_sha256": sha256_file(directory / "slot/slot_anchor_index.json"),
            "statistics": {"path": str(stat_path), "sha256": sha256_file(stat_path),
                "stats_key": entry["stats_key"], "content_hash": stats["content_hash"], "dataset_root": entry["dataset_path"]}}
        for phase, valid in selections.items():
            values = np.flatnonzero(valid).astype(np.int64)
            path = directory / (phase + "_indices.npy")
            if path.exists() or not len(values):
                raise ValueError(f"refusing empty or overwritten frozen index: {path}")
            np.save(path, values, allow_pickle=False)
            report[phase] = {"file": path.name, "sha256": sha256_file(path), "count": len(values)}
        report["content_hash"] = canonical_json_hash(report)
        write_json(directory / "frozen_index.json", report)
    write_json(output / "aux_dataset_routes.json", {"version": 1,
        "datasets": {entry["dataset_entry"]: entry for entry in entries(config).values()}})
