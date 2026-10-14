#!/usr/bin/env bash
# Adapted from RoboTwin's MIT-licensed installer; use an isolated Python 3.10 environment.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
python -m pip install -r "$REPO_ROOT/requirements/robotwin.txt"
python -m pip install 'setuptools==69.5.1' 'setuptools-scm<9' ninja wheel
# Keep transitive extension dependencies within the simulator's pinned versions.
python -m pip install --no-build-isolation -c "$REPO_ROOT/requirements/robotwin.txt" 'git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47'
python -m pip install --no-build-isolation -c "$REPO_ROOT/requirements/robotwin.txt" 'git+https://github.com/NVlabs/curobo.git@d64c4b005459db10c5dd867d8b30a87d5bda9bdb'
python "$REPO_ROOT/scripts/patch_robotwin_environment.py" --apply
python "$REPO_ROOT/scripts/robotwin/patch_tasks.py" --apply
python -m pip check
