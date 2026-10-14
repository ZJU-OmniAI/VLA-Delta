"""Reuse an immutable audit snapshot without scanning original payloads again."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path


def stat_identity(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "device": stat.st_dev, "inode": stat.st_ino}


class PreparationAuditCache:
    def __init__(self, path, *, require_passed=True):
        self.path = Path(path).absolute()
        self.identity = stat_identity(self.path)
        payload = self.path.read_bytes()
        self.sha256 = hashlib.sha256(payload).hexdigest()
        self.snapshot = json.loads(payload)
        if self.snapshot.get("version") != 1:
            raise ValueError("unsupported preparation audit cache")
        if require_passed and self.snapshot.get("status") != "passed":
            raise ValueError(f"saved preparation audit did not pass: {self.snapshot.get('blockers')}")
        self.files = self.snapshot["files"]

    def check(self, path, expected_sha256=None):
        name = str(Path(path).absolute())
        record = self.files.get(name)
        if record is None:
            raise ValueError(f"file is absent from the saved audit: {name}")
        if stat_identity(path) != record["stat"]:
            raise ValueError(f"file changed since the saved audit; no automatic re-audit: {name}")
        if expected_sha256 is not None and record["sha256"] != expected_sha256:
            raise ValueError(f"saved audit SHA256 contract mismatch: {name}")
        return record["sha256"]

    def check_all(self):
        if stat_identity(self.path) != self.identity:
            raise ValueError("preparation audit snapshot changed after loading")
        for path in self.files:
            self.check(path)


@lru_cache(maxsize=2)
def load_preparation_audit_cache(path):
    cache = PreparationAuditCache(path)
    from utils.three_stage_preflight import implementation_identity
    expected_implementation = cache.snapshot["runtime_implementation"]
    binding_path = cache.path.parent / "audit_runtime_binding.json"
    if binding_path.is_file():
        binding = json.loads(binding_path.read_text())
        if binding.get("version") != 1 or binding["audit_snapshot_sha256"] != cache.sha256:
            raise ValueError("runtime binding does not reference the unchanged saved audit")
        expected_implementation = binding["runtime_implementation"]
    if expected_implementation != implementation_identity():
        raise ValueError("implementation changed since the saved preparation audit")
    import importlib.metadata
    import sys
    environment_path = Path(cache.snapshot.get("runtime_environment_path", cache.path.parent / "runtime_environment.json"))
    cache.check(environment_path)
    environment = json.loads(environment_path.read_text())
    if environment["python"] != sys.executable or environment["python_version"] != sys.version:
        raise ValueError("Python runtime changed since the saved preparation audit")
    for package, version in environment["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"package changed since the saved preparation audit: {package}")
    return cache


def load_cached_sidecar(path, *, audit_cache, expected_generation=None):
    """Manifest semantics and array contents were checked before snapshot creation."""
    from utils.sidecar import MANIFEST_NAME, SIDECAR_FORMAT_VERSION
    path = Path(path).resolve()
    audit_cache.check(path / MANIFEST_NAME)
    manifest = json.loads((path / MANIFEST_NAME).read_text())
    if manifest.get("sidecar_format_version") != SIDECAR_FORMAT_VERSION:
        raise ValueError("cached Stage05 sidecar format mismatch")
    for key, expected in (expected_generation or {}).items():
        if manifest["generation"].get(key) != expected:
            raise ValueError(f"cached Stage05 sidecar generation mismatch: {key}")
    for name, digest in manifest["files"].items():
        audit_cache.check(path / name, digest)
    return manifest
