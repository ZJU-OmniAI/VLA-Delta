"""Ordered real-frame probes using the production LeRobot preprocessing."""
import json
from pathlib import Path


def read_probe(path, entries):
    probe = json.loads(Path(path).read_text())
    if probe.get("version", 1) == 1:
        if list(entries) != [probe["dataset_entry"]]:
            raise ValueError("legacy probe requires its single dataset")
        rows = [[probe["dataset_entry"], *row] for row in probe["frames"]]
    elif probe["version"] == 2:
        if probe.get("dataset_entries") != list(entries):
            raise ValueError("probe dataset entry order differs")
        rows = probe["frames"]
    else:
        raise ValueError("unsupported probe version")
    if not rows or any(len(row) != 3 or row[0] not in entries or
            any(type(x) is not int or x < 0 for x in row[1:]) for row in rows):
        raise ValueError("invalid probe sample identity")
    if set(row[0] for row in rows) != set(entries):
        raise ValueError("probe must cover every dataset entry")
    return rows


def configure_downstream_dataset(dataset, probe_manifest=None, *, entries=None, contract_version=1):
    from utils.load_training_dataset import StreamingLeRobotSampleDataset
    if (type(dataset) is not StreamingLeRobotSampleDataset or dataset.requirements.requires_target
            or dataset.window_size != 1 or dataset.sample_ratio != 1.0 or dataset.ecot_supported):
        raise ValueError("downstream state requires deterministic full-frame action-only LeRobot inputs")
    dataset.track_downstream = True
    dataset.downstream_contract_version = contract_version
    if contract_version == 2:
        episodes = dataset.ds.episodes if dataset.ds.episodes is not None else sorted(dataset.ds.meta.episodes)
        dataset.downstream_episode_offsets = dict(zip(episodes, dataset.ds.episode_data_index["from"].tolist()))
    dataset.manifest = {"counts": {"source_frames": dataset.ds.meta.total_frames}}
    if probe_manifest:
        rows = read_probe(probe_manifest, entries or [dataset.dataset_entry])
        offsets = dict(zip(dataset.ds.episodes, dataset.ds.episode_data_index["from"].tolist()))
        indices = []
        for name, episode, frame in rows:
            if name != dataset.dataset_entry:
                continue
            if episode not in offsets or not 0 <= frame < dataset.ds.meta.episodes[episode]["length"]:
                raise ValueError(f"probe identity outside selected episode: {(name, episode, frame)}")
            indices.append(offsets[episode] + frame)
        dataset.probe_indices = indices
        dataset.fixed_probe_order = True
        dataset._init_sampled_indices()


def bind_global_probe_order(concat, path):
    entries = [ds.dataset_entry for ds in concat.datasets]
    rows = read_probe(path, entries)
    offsets = dict(zip(entries, [0] + concat.cumulative_sizes[:-1]))
    consumed = dict.fromkeys(entries, 0)
    order = []
    for name, _, _ in rows:
        order.append(offsets[name] + consumed[name])
        consumed[name] += 1
    if sorted(order) != list(range(len(concat))):
        raise ValueError("probe order does not cover concatenated samples")
    concat.downstream_probe_order = order
