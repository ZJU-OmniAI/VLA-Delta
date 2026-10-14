"""Versioned, conservative Flow candidate indexes; exact masks stay sample-local."""

import json
from pathlib import Path

import numpy as np

from utils.aux_data_contract import digest_file, public_flow_contract


def load_flow_candidates(dataset):
    path = Path(dataset.entry["flow_candidate_index"])
    meta = json.loads(path.with_suffix(".json").read_text())
    expected = public_flow_contract(dataset.flow_reader.contract)
    if meta.get("version") != 1 or meta["flow_contract"] != expected or digest_file(path) != meta["sha256"]:
        raise ValueError("Flow candidate index identity mismatch")
    indices = np.load(path, mmap_mode="r", allow_pickle=False)
    if indices.ndim != 1 or indices.dtype != np.int64 or (len(indices) > 1 and (indices[1:] <= indices[:-1]).any()):
        raise ValueError("invalid Flow candidate index")
    return indices
