#!/usr/bin/env python3
"""Reuse or compute LeRobot v2 state/action statistics for LIBERO and RoboTwin."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "lerobot")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", "--dataset_path", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.dataset_path.resolve()
    from lerobot.common.datasets.utils import calculate_global_stats, load_stats
    if (root / "meta/stats.json").is_file():
        load_stats(root)
        print(f"Using existing statistics: {root / 'meta/stats.json'}")
        return
    if not (root / "meta/info.json").is_file() or not (root / "data").is_dir():
        parser.error("requires a complete local LeRobot v2 dataset (meta/ and data/)")
    if not any((root / "data").rglob("*.parquet")):
        parser.error("no training Parquet files found under data/")
    calculate_global_stats(str(root))


if __name__ == "__main__":
    main()
