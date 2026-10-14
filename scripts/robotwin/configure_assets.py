#!/usr/bin/env python3
"""Render RoboTwin planner paths from existing asset templates (MIT adaptation)."""
import argparse
from pathlib import Path
import shutil


def configure_paths(root, *, dry_run=False):
    root = Path(root).resolve()
    templates = sorted((root / "assets/embodiments").rglob("*_tmp.yml"))
    if not templates:
        raise FileNotFoundError("No embodiment templates; download/unzip assets first")
    changed = []
    for template in templates:
        target = template.with_name(template.name.replace("_tmp.yml", ".yml"))
        content = template.read_text(encoding="utf-8")
        rendered = content.replace("${ASSETS_PATH}", str(root)).replace("$ASSETS_PATH", str(root))
        if target.exists() and target.read_text(encoding="utf-8") == rendered:
            continue
        if not dry_run:
            if target.exists():
                backup = target.with_name(target.name + ".vla-delta-original")
                if not backup.exists():
                    shutil.copy2(target, backup)
            target.write_text(rendered, encoding="utf-8")
        changed.append(str(target.relative_to(root)))
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "third_party/robotwin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    changed = configure_paths(args.robotwin_root, dry_run=args.dry_run)
    print(("Would update" if args.dry_run else "Updated") + f" {len(changed)} planner configurations")
    for path in changed:
        print(path)
