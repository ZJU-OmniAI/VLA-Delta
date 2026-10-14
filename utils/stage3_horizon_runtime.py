"""Explicit H50 runtime adaptation over immutable, audited H10 sidecars."""

import numpy as np


def extend_fm_supervision(dataset, global_index, result):
    if bool(result["fm_eligible"]):
        return result
    episode_meta, base = dataset._episode_for_global(global_index)
    stop = min(global_index + dataset.action_horizon, int(episode_meta["dataset_to_index"]))
    indices = np.arange(global_index, stop, dtype=np.int64)
    valid = (dataset._frozen_validity[indices // 8, 3] >> (7 - indices % 8)) & 1
    if not valid.any():
        return result
    episode = int(episode_meta["episode_index"])
    rows = dataset._episode_rows(episode)
    try:
        chunk = dataset._canonical_chunk(episode, rows, base)
    except ValueError:
        # The original routine caches canonical validity before rejecting a state.
        cached = dataset._canonical_cache.get(episode)
        if cached is None or bool(cached[2][base]):
            raise
        return result
    if chunk.temporal_mask.any():
        result.update(dataset._action_inputs(episode, rows, base))
        result["fm_eligible"] = result["action_supervision_available"]
    return result
