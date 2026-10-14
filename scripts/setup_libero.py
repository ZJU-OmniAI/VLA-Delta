#!/usr/bin/env python3
"""Write an explicit LIBERO path configuration without importing its interactive initializer."""
import argparse
import os
from pathlib import Path
import shutil
import yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-root", type=Path, required=True, help="External LIBERO checkout")
    parser.add_argument("--config-dir", type=Path, default=Path(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")).expanduser())
    parser.add_argument("--overwrite", action="store_true",
                        help="Update an existing configuration after moving LIBERO; keep its first backup")
    args = parser.parse_args(argv)
    args.libero_root = args.libero_root.expanduser().resolve()
    args.config_dir = args.config_dir.expanduser().resolve()
    package = args.libero_root / "libero/libero"
    paths = dict(benchmark_root=str(package), bddl_files=str(package / "bddl_files"),
                 init_states=str(package / "init_files"), assets=str(package / "assets"),
                 datasets=str(args.libero_root / "datasets"))
    for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
        if not Path(paths[key]).is_dir():
            parser.error(f"missing LIBERO resource: {key}: {paths[key]}")
    args.config_dir.mkdir(parents=True, exist_ok=True)
    path = args.config_dir / "config.yaml"
    if path.exists() and yaml.safe_load(path.read_text(encoding="utf-8")) != paths:
        if not args.overwrite:
            parser.error(f"different config already exists: {path}; choose a separate --config-dir or use --overwrite")
        backup = path.with_name(path.name + ".vla-delta-original")
        if not backup.exists():
            shutil.copy2(path, backup)
    path.write_text(yaml.safe_dump(paths), encoding="utf-8")
    print(f"export LIBERO_CONFIG_PATH={str(args.config_dir.resolve())!r}")


if __name__ == "__main__":
    main()
