#!/usr/bin/env python3
"""Validate a converted RoboTwin LeRobot v2 dataset before action-only training."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "lerobot")]


def validate_dataset(root, *, check_all_files=False):
    from utils.robotwin_data import CAMERA_KEYS, validate_robotwin_metadata

    root = Path(root).expanduser().resolve()
    meta = root / "meta"
    for name in ("info.json", "stats.json", "tasks.jsonl", "episodes.jsonl"):
        if not (meta / name).is_file():
            raise FileNotFoundError(f"missing LeRobot metadata: {meta / name}")
    info = json.loads((meta / "info.json").read_text())
    stats = json.loads((meta / "stats.json").read_text())
    contract = validate_robotwin_metadata(info, stats)
    episodes = [
        json.loads(line)
        for line in (meta / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    tasks = [
        json.loads(line)
        for line in (meta / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    task_ids = [row["task_index"] for row in tasks]
    if (
        not tasks
        or len(set(task_ids)) != len(task_ids)
        or any(type(index) is not int or index < 0 for index in task_ids)
        or any(
            not isinstance(row.get("task"), str) or not row["task"].strip()
            for row in tasks
        )
    ):
        raise ValueError(
            "tasks.jsonl must define unique task IDs and non-empty language instructions"
        )
    if (
        not episodes
        or len(episodes) != info["total_episodes"]
        or [row["episode_index"] for row in episodes] != list(range(len(episodes)))
        or any(type(row["length"]) is not int or row["length"] <= 0 for row in episodes)
        or sum(row["length"] for row in episodes) != info["total_frames"]
    ):
        raise ValueError("episode IDs, lengths or metadata totals disagree")
    if info.get("splits", {}).get("train") != f"0:{len(episodes)}":
        raise ValueError(
            "this recipe requires a training-only dataset; export the train split separately"
        )
    selected = episodes if check_all_files else [episodes[0], *episodes[-1:]]
    checked = set()
    for row in selected:
        episode = row["episode_index"]
        if episode in checked:
            continue
        checked.add(episode)
        fields = dict(
            episode_index=episode, episode_chunk=episode // info["chunks_size"]
        )
        paths = [root / info["data_path"].format(**fields)]
        paths.extend(
            root / info["video_path"].format(**fields, video_key=key)
            for key in CAMERA_KEYS
            if info["features"][key]["dtype"] == "video"
        )
        for path in paths:
            if not path.resolve().is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(
                    f"missing or out-of-root training payload: {path}"
                )
    # Inspect the first/last episodes, without loading the complete training corpus.
    import numpy as np
    import pyarrow.parquet as pq

    for row in {r["episode_index"]: r for r in (episodes[0], episodes[-1])}.values():
        episode = row["episode_index"]
        path = root / info["data_path"].format(
            episode_index=episode, episode_chunk=episode // info["chunks_size"]
        )
        table = pq.read_table(
            path, columns=["observation.state", "action", "task_index", "episode_index"]
        )
        if len(table) != row["length"] or set(table["episode_index"].to_pylist()) != {
            episode
        }:
            raise ValueError(
                f"episode {episode}: Parquet length or episode identity mismatch"
            )
        for key in ("observation.state", "action"):
            values = np.asarray(table[key].to_pylist())
            if values.shape != (row["length"], 14) or not np.isfinite(values).all():
                raise ValueError(f"episode {episode}: invalid finite 14D {key}")
        if not set(table["task_index"].to_pylist()).issubset(set(task_ids)):
            raise ValueError(f"episode {episode}: unknown language task IDs")
    return dict(
        dataset_path=str(root),
        episodes=len(episodes),
        frames=info["total_frames"],
        payload_episodes_checked=len(checked),
        parquet_episodes_checked=min(2, len(episodes)),
        **contract,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", "--dataset_path", type=Path, required=True)
    parser.add_argument(
        "--check-all-files",
        action="store_true",
        help="Check every episode's Parquet/video paths; default checks first and last",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            validate_dataset(args.dataset_path, check_all_files=args.check_all_files),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
