#!/usr/bin/env python3
"""Apply the two existing RoboTwin simulator compatibility edits explicitly."""
import argparse
from importlib import metadata, util
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Modify the active environment; otherwise inspect only')
    args = parser.parse_args()
    patches = [
        ('sapien', '3.0.0b1', 'wrapper/urdf_loader.py', [
            ('open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")'),
            ('open(srdf_file, "r")', 'open(srdf_file, "r", encoding="utf-8")')]),
        ('mplib', '0.2.1', 'planner.py', [
            ('if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:',
             'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:')]),
    ]
    for package, version, relative, changes in patches:
        if metadata.version(package) != version:
            raise ValueError(f'{package}: patch is restricted to {version}')
        path = Path(util.find_spec(package).origin).parent / relative
        original = path.read_text()
        updated = original
        for before, after in changes:
            if before not in updated and after not in updated:
                raise ValueError(f'{package}: expected source was not found; inspect manually')
            updated = updated.replace(before, after)
        print(f'{package}: ' + ('already patched' if updated == original else 'patch required'))
        if args.apply and updated != original:
            backup = path.with_suffix(path.suffix + '.vla-delta-original')
            if backup.exists():
                raise FileExistsError(f'original backup already exists: {backup}')
            shutil.copy2(path, backup)
            path.write_text(updated)


if __name__ == '__main__':
    main()
