"""Implementation identity used by reusable preparation audit caches."""

from pathlib import Path

from utils.sidecar import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def implementation_identity():
    paths = [ROOT / "train_vla.py", ROOT / "scripts/train.py",
             ROOT / "scripts/prepare_pretraining.py", ROOT / "dataset2feature.yaml",
             ROOT / "lerobot/lerobot/common/datasets/video_utils.py"]
    for directory in ("model", "utils"):
        paths.extend(sorted((ROOT / directory).glob("*.py")))
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}
