#!/usr/bin/env python3
"""Build portable pretraining artifacts from existing Stage05/Slot annotations."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "lerobot")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("all", "sidecars", "sources", "slots", "freeze"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from utils.runtime_config import load_mapping
    config = load_mapping(args.config)
    if set(config) != {"dataset_registry", "dataset_entries", "output_root"}:
        parser.error("preparation config requires dataset_registry, dataset_entries, output_root only")
    from utils.pretraining_preparation import entries, sidecars, sources, slots, freeze
    resolved = entries(config)
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    phases = dict(sidecars=sidecars, sources=sources, slots=slots, freeze=freeze)
    try:
        for name in phases if args.phase == "all" else [args.phase]:
            phases[name](config)
            print(f"Completed phase: {name}", flush=True)
    except BaseException as error:
        print(f"Failed phase: {name}; error: {type(error).__name__}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
