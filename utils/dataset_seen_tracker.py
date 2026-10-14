"""Exact per-source seen/unique/duplicate accounting for Stage05 runs."""

from __future__ import annotations

import hashlib
from utils.checkpoint_metadata import checkpoint_metadata
import json
from pathlib import Path

import numpy as np
import torch


class DatasetSeenTracker:
    def __init__(self, concat_dataset, accelerator, *, resume_directory=None, flow_config=None):
        self.accelerator = accelerator
        self.flow_config = flow_config
        self.auxiliary_enabled = any(getattr(dataset.spec, "auxiliary_contract", None) for dataset in concat_dataset.datasets)
        self.entries = []
        for dataset_id, dataset in enumerate(concat_dataset.datasets):
            manifest = getattr(dataset, "manifest", None)
            if manifest is None:
                self.entries = []
                break
            self.entries.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_entry": dataset.spec.dataset_entry,
                    "source_frames": int(manifest["counts"]["source_frames"]),
                    **({"unfiltered_length": getattr(dataset, "unfiltered_length", len(dataset)), "filtered_length": len(dataset)} if self.auxiliary_enabled else {}),
                }
            )
        self.enabled = bool(self.entries)
        self.seen = np.zeros(len(self.entries), dtype=np.int64)
        self.unique = np.zeros(len(self.entries), dtype=np.int64)
        self.duplicates = np.zeros(len(self.entries), dtype=np.int64)
        self.ar_eligible = np.zeros(len(self.entries), dtype=np.int64)
        self.fm_eligible = np.zeros(len(self.entries), dtype=np.int64)
        self.aux_names = ["anchor", *[f"Q{i}_valid" for i in range(1, 10)], "flow_available", "flow_eligible",
                          "flow_valid_pixels", "flow_missing", "flow_tail", "flow_empty_mask"]
        self.aux_counts = np.zeros((len(self.entries), len(self.aux_names)), dtype=np.int64)
        self.bitsets = (
            [np.zeros(entry["source_frames"], dtype=bool) for entry in self.entries]
            if self.enabled and accelerator.is_main_process
            else []
        )
        if self.enabled and resume_directory is not None:
            self._load(Path(resume_directory))

    def _load(self, directory: Path) -> None:
        if not self.accelerator.is_main_process:
            return
        path = directory / "data_seen_state.npz"
        if not path.is_file():
            raise ValueError(f"resume checkpoint lacks Stage05 data seen state: {path}")
        payload = np.load(path, allow_pickle=False)
        stored_entries = json.loads(str(payload["entries_json"].item()))
        if stored_entries != self.entries:
            raise ValueError("resume data seen state dataset contract mismatch")
        for name in ("seen", "unique", "duplicates", "ar_eligible", "fm_eligible"):
            setattr(self, name, payload[name].astype(np.int64))
        if self.auxiliary_enabled:
            self.aux_counts = payload["aux_counts"].astype(np.int64)
        for index, bitset in enumerate(self.bitsets):
            unpacked = np.unpackbits(payload[f"bitset_{index}"])[: len(bitset)]
            bitset[:] = unpacked.astype(bool)

    def update(self, batches: list[dict]) -> None:
        if not self.enabled:
            return
        for batch in batches:
            required = ("dataset_id", "sample_global_index", "ar_eligible", "fm_eligible")
            if any(not isinstance(batch.get(key), torch.Tensor) for key in required):
                raise ValueError("Stage05 batch lacks exact data-accounting fields")
            local = torch.stack(
                [
                    batch["dataset_id"].reshape(-1).to(torch.long),
                    batch["sample_global_index"].reshape(-1).to(torch.long),
                    batch["ar_eligible"].reshape(-1).to(torch.long),
                    batch["fm_eligible"].reshape(-1).to(torch.long),
                ],
                dim=1,
            )
            gathered = self.accelerator.gather(local).detach().cpu().numpy()
            if self.auxiliary_enabled:
                from utils.slot_labels import task_validity
                device, size = local.device, len(local)
                zero = torch.zeros(size, dtype=torch.long, device=device)
                valid = task_validity(batch, device=device)
                values = [batch.get("slot_anchor", zero).to(device), *[valid[f"Q{i}"] for i in range(1, 10)]]
                available = batch.get("flow_supervision_available", zero).to(device)
                eligible, pixels = zero.clone(), zero.clone()
                if self.flow_config is not None and self.flow_config.enabled:
                    from utils.optical_flow_loss import flow_supervision_indices, prepare_flow_targets
                    indices = flow_supervision_indices(batch, self.flow_config, device)
                    eligible[indices] = 1
                    _, _, masks = prepare_flow_targets(batch, self.flow_config, device)
                    pixels[indices] = masks.flatten(1).sum(1)
                reasons = batch.get("flow_exclusion_reason", torch.full_like(zero, -1)).to(device)
                values += [available, eligible, pixels, reasons == 1, reasons == 2, (available.bool() & ~eligible.bool())]
                counters = torch.stack([value.long() for value in values], dim=1)
                all_counts = self.accelerator.gather(counters).detach().cpu().numpy()
                if self.accelerator.is_main_process:
                    for row, counts in zip(gathered, all_counts):
                        self.aux_counts[int(row[0])] += counts
            if not self.accelerator.is_main_process:
                continue
            for dataset_id, global_index, ar_valid, fm_valid in gathered:
                dataset_id, global_index = int(dataset_id), int(global_index)
                if not 0 <= dataset_id < len(self.entries):
                    raise ValueError("batch contains invalid dataset_id")
                bitset = self.bitsets[dataset_id]
                if not 0 <= global_index < len(bitset):
                    raise ValueError("batch contains invalid source frame index")
                self.seen[dataset_id] += 1
                self.ar_eligible[dataset_id] += int(ar_valid)
                self.fm_eligible[dataset_id] += int(fm_valid)
                if bitset[global_index]:
                    self.duplicates[dataset_id] += 1
                else:
                    bitset[global_index] = True
                    self.unique[dataset_id] += 1

    def manifest(self, *, epoch: int, global_step: int) -> dict:
        total_seen = int(self.seen.sum())
        rows = []
        for index, entry in enumerate(self.entries):
            seen = int(self.seen[index])
            rows.append(
                {
                    **entry,
                    "seen": seen,
                    "unique": int(self.unique[index]),
                    "duplicate": int(self.duplicates[index]),
                    "ar_eligible_seen": int(self.ar_eligible[index]),
                    "fm_eligible_seen": int(self.fm_eligible[index]),
                    **({"auxiliary": dict(zip(self.aux_names, self.aux_counts[index].tolist()))} if self.auxiliary_enabled else {}),
                    "seen_ratio": seen / total_seen if total_seen else 0.0,
                    "ar_eligible_ratio_within_seen": int(self.ar_eligible[index]) / seen if seen else 0.0,
                    "fm_eligible_ratio_within_seen": int(self.fm_eligible[index]) / seen if seen else 0.0,
                }
            )
        result = {
            "format_version": 1,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "total_seen": total_seen,
            "datasets": rows,
        }
        result["content_hash"] = hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return result

    def save(self, directory, *, epoch: int, global_step: int) -> None:
        if not self.enabled or not self.accelerator.is_main_process:
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = self.manifest(epoch=epoch, global_step=global_step)
        (directory / "data_seen_manifest.json").write_text(
            json.dumps(checkpoint_metadata(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        arrays = {
            "entries_json": np.asarray(json.dumps(self.entries, sort_keys=True)),
            "seen": self.seen,
            "unique": self.unique,
            "duplicates": self.duplicates,
            "ar_eligible": self.ar_eligible,
            "fm_eligible": self.fm_eligible,
        }
        arrays.update(
            {f"bitset_{index}": np.packbits(bitset) for index, bitset in enumerate(self.bitsets)}
        )
        if self.auxiliary_enabled:
            arrays["aux_counts"] = self.aux_counts
        np.savez_compressed(directory / "data_seen_state.npz", **arrays)
