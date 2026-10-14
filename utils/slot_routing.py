"""Independent audited Slot sources, shared classes, per-source displacement scales."""

import json
from pathlib import Path

import numpy as np

from utils.slot_supervision import SlotSupervisionReader, class_weights
from utils.slot_labels import VOCABULARIES


class SlotSupervisionRouter:
    def __init__(self, directory, *, audit_cache=None):
        root = Path(directory)
        if audit_cache is not None:
            audit_cache.check(root / "slot_routes.json")
        routes = json.loads((root / "slot_routes.json").read_text())
        if routes.get("version") != 1:
            raise ValueError("unsupported Slot route version")
        self.readers = {}
        for identity, location in sorted(routes["datasets"].items()):
            reader = SlotSupervisionReader(root / location, audit_cache=audit_cache)
            if reader.stats.get("dataset_identity") != identity:
                raise ValueError("Slot route dataset identity mismatch")
            self.readers[identity] = reader
        roots = [str(reader.root.resolve()) for reader in self.readers.values()]
        if len(set(roots)) != len(roots):
            raise ValueError("duplicate Slot dataset route")
        self.stats = combined_slot_stats({key: reader.stats for key, reader in self.readers.items()})

    def for_dataset(self, spec):
        for index, (identity, reader) in enumerate(self.readers.items()):
            if Path(spec.dataset_path).resolve() == reader.root.resolve():
                if reader.stats["rules"]["camera"] not in spec.camera_keys:
                    raise ValueError(f"Slot route camera absent from {spec.dataset_entry}")
                return reader, index
        raise ValueError(f"Slot route missing for {spec.dataset_entry}: {spec.dataset_path}")


def combined_slot_stats(datasets):
    if not datasets:
        raise ValueError("empty Slot routes")
    ordered = dict(sorted(datasets.items()))
    ratios = {stats["rules"]["max_weight_ratio"] for stats in ordered.values()}
    if len(ratios) != 1:
        raise ValueError("Slot statistical rules differ between datasets")
    classes = {}
    for q, vocabulary in VOCABULARIES.items():
        if any(stats["classes"][q]["vocabulary"] != vocabulary for stats in ordered.values()):
            raise ValueError(f"Slot class semantics differ: {q}")
        counts = np.sum([stats["classes"][q]["counts"] for stats in ordered.values()], axis=0).tolist()
        classes[q] = {"vocabulary": vocabulary, "counts": counts, "weights": class_weights(counts, next(iter(ratios)))}
    return {"version": 2, "schema_version": "future_difference_training_v5", "split": "train",
            "dataset_order": list(ordered), "datasets": ordered, "classes": classes}


def load_slot_supervision(directory, *, audit_cache=None):
    if (Path(directory) / "slot_routes.json").is_file():
        return SlotSupervisionRouter(directory, audit_cache=audit_cache)
    return SlotSupervisionReader(directory, audit_cache=audit_cache)
