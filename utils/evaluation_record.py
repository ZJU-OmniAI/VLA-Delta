"""Incremental evaluation records, independent of training and simulator packages."""
from datetime import datetime, timezone
import json
from pathlib import Path


class EvaluationRecord:
    def __init__(self, directory, *, expected, metadata):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "episodes.jsonl"
        self.stream = self.path.open("x", buffering=1)
        self.expected = expected
        self.completed = self.successes = self.errors = 0
        self.metadata = metadata
        (self.root / "evaluation.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
        self.write_summary()

    def episode(self, *, success, error=None, **values):
        row = dict(**values, success=bool(success) and error is None, error=error,
                   recorded_at=datetime.now(timezone.utc).isoformat())
        self.stream.write(json.dumps(row) + "\n")
        self.stream.flush()
        self.errors += error is not None
        self.completed += error is None
        self.successes += row["success"]
        self.write_summary()

    def write_summary(self):
        complete = self.completed == self.expected and self.errors == 0
        summary = dict(expected=self.expected, completed=self.completed, successes=self.successes,
            errors=self.errors, complete=complete,
            success_rate=self.successes / self.completed if self.completed else None,
            status="complete" if complete else "incomplete")
        temporary = self.root / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(self.root / "summary.json")
        return summary

    def close(self):
        self.stream.close()
        return self.write_summary()


def validate_server(metadata, *, action_dim, execution_horizon):
    if not metadata:
        raise ValueError("evaluation requires server metadata; use this release's server.py")
    if metadata.get("action_dimension") != action_dim:
        raise ValueError("checkpoint action dimension does not match the benchmark")
    if execution_horizon < 1 or execution_horizon > metadata["action_horizon"]:
        raise ValueError("execution horizon exceeds checkpoint prediction horizon")
