"""Source-file identities for reproducible pretraining preparation."""

from pathlib import Path
from utils.sidecar import sha256_file


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
