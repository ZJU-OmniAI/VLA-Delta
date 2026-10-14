"""Strict Stage06 manifest join with process-local bounded HDF5 handles."""

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from utils.training_tokenization import DatasetIntegrityError
from utils.preparation_audit_cache import stat_identity
from utils.aux_data_contract import digest_file


VALID_FRACTION_ATOL = 1e-6
CURRENT_MANIFEST_SCHEMA = "stage06_flow_manifest_v2"
CURRENT_FILE_SCHEMA = "stage06_flow_v2"
LEGACY_SCHEMA_CONTRACT = "stage06_flow_legacy_v1"


def _valid_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


class FlowFileIntegrityVerifier:
    """Cache a verified content digest while the complete file stat stays fixed."""

    def __init__(self, *, audit_cache=None):
        self.audit_cache = audit_cache
        self._records = {}
        self._pid = os.getpid()

    def reset_for_process(self):
        if self._pid != os.getpid():
            self._records.clear()
            self._pid = os.getpid()

    def is_current(self, path, expected_sha256):
        self.reset_for_process()
        path = Path(path).resolve()
        record = self._records.get(str(path))
        if not record or record["expected_sha256"] != expected_sha256:
            return False
        try:
            current = stat_identity(path)
        except OSError:
            return False
        return record["stat"] == current

    def verify(self, path, expected_sha256, *, source):
        self.reset_for_process()
        path = Path(path).resolve()
        if not _valid_sha256(expected_sha256):
            raise DatasetIntegrityError(
                f"Flow HDF5 manifest has no valid content SHA256: file={path}, source={source}"
            )
        try:
            before = stat_identity(path)
            record = self._records.get(str(path))
            if (record and record["expected_sha256"] == expected_sha256
                    and record["stat"] == before):
                return expected_sha256
            if self.audit_cache is not None:
                # The saved audit binds the expected source/stat identity. The
                # content digest still comes from the current HDF5 bytes.
                self.audit_cache.check(path, expected_sha256)
            actual = digest_file(path)
            after = stat_identity(path)
        except Exception as error:
            raise DatasetIntegrityError(
                f"Flow HDF5 integrity verification failed: file={path}, source={source}: {error}"
            ) from error
        if before != after:
            raise DatasetIntegrityError(
                f"Flow HDF5 changed during integrity verification: file={path}, source={source}"
            )
        if actual != expected_sha256:
            raise DatasetIntegrityError(
                f"Flow HDF5 content SHA256 differs from manifest: file={path}, "
                f"source={source}, expected={expected_sha256}, actual={actual}"
            )
        self._records[str(path)] = {
            "expected_sha256": expected_sha256, "stat": after,
        }
        return actual

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_records"] = {}
        state["_pid"] = None
        return state


class OpticalFlowReader:
    def __init__(self, root, manifest, *, delta_frames=10, label_source=1,
                 camera_key=None, max_handles=4, contract=None, schema_contract=None):
        import h5py

        self.root = Path(root).resolve()
        self.manifest = Path(manifest)
        if not self.manifest.is_absolute():
            self.manifest = self.root / self.manifest
        self.delta_frames = delta_frames
        if isinstance(label_source, bool) or not isinstance(label_source, int) or label_source < 1:
            raise ValueError("label_source must be a positive integer")
        self.label_source = label_source
        self.contract = contract
        if schema_contract not in (None, LEGACY_SCHEMA_CONTRACT):
            raise ValueError(f"unsupported explicit Flow schema contract: {schema_contract}")
        if contract is not None and schema_contract == LEGACY_SCHEMA_CONTRACT:
            raise ValueError("verified Stage05 Flow contract cannot be combined with legacy schema")
        self.schema_contract = schema_contract
        self.expected_fps = contract["fps"] if contract else 10.0
        self.camera_key = camera_key
        if max_handles < 1:
            raise ValueError("max_handles must be positive")
        self.max_handles = max_handles
        self._handles = OrderedDict()
        self._pid = os.getpid()
        self.rows = {}
        self.episodes = {}
        self.episode_schemas = {}
        self._audit_cache = contract.get("audit_cache") if contract else None
        self._integrity_verifier = FlowFileIntegrityVerifier(audit_cache=self._audit_cache)
        payload = self.manifest.read_bytes()
        self.manifest_sha256 = hashlib.sha256(payload).hexdigest()
        try:
            entries = [json.loads(line) for line in payload.splitlines() if line.strip()]
            if not entries:
                raise ValueError("empty flow manifest")
            declared_ids = {entry.get("dataset_id") for entry in entries if entry.get("dataset_id")}
            if len(declared_ids) > 1:
                raise ValueError("flow manifest mixes dataset_id values")
            self.dataset_id = (contract["dataset_id"] if contract else
                               next(iter(declared_ids), f"unversioned-{self.manifest_sha256[:16]}"))
            if camera_key is None:
                camera_key = entries[0].get("camera_key")
                if not isinstance(camera_key, str) or not camera_key:
                    raise ValueError("manifest has no camera key")
            self.camera_key = camera_key
            for entry in entries:
                source_episode = entry["merged_episode_index"]
                remapping = contract.get("flow_to_dataset_episode") if contract else None
                if remapping is not None and str(source_episode) not in remapping:
                    raise ValueError("Flow manifest record is absent from verified remapping")
                episode = remapping[str(source_episode)] if remapping is not None else source_episode
                if not isinstance(episode, int) or episode < 0 or episode in self.episodes:
                    raise ValueError(f"duplicate/invalid episode {episode}")
                if entry["camera_key"] != camera_key:
                    raise ValueError("flow camera mapping mismatch")
                if contract is not None:
                    if (entry.get("schema_version") != CURRENT_MANIFEST_SCHEMA
                            or entry.get("dataset_id") != contract["dataset_id"]
                            or str(episode) not in contract["mapping"]
                            or entry["source_episode_index"] != contract["mapping"][str(episode)]["old_episode_index"]):
                        raise ValueError("flow dataset_id/schema/episode mapping mismatch")
                manifest_schema = entry.get("schema_version")
                legacy_declared = (schema_contract == LEGACY_SCHEMA_CONTRACT
                                   or entry.get("schema_contract") == LEGACY_SCHEMA_CONTRACT)
                if manifest_schema == CURRENT_MANIFEST_SCHEMA and not legacy_declared:
                    resolved_schema = CURRENT_FILE_SCHEMA
                elif manifest_schema is None and legacy_declared:
                    resolved_schema = LEGACY_SCHEMA_CONTRACT
                else:
                    raise ValueError(
                        "Flow schema is missing, conflicting, or unsupported; current data require "
                        f"{CURRENT_MANIFEST_SCHEMA}, and legacy data require explicit "
                        f"{LEGACY_SCHEMA_CONTRACT}"
                    )
                if not _valid_sha256(entry.get("sha256")):
                    raise ValueError("Flow manifest record requires a lowercase HDF5 content sha256")
                path = (self.root / entry["hdf5_path"]).resolve()
                if not path.is_relative_to(self.root):
                    raise ValueError("HDF5 path escapes root")
                self.episodes[episode] = (path, entry)
                self.episode_schemas[episode] = resolved_schema
        except Exception as error:
            raise DatasetIntegrityError(f"flow manifest {self.manifest}: {error}") from error

    @staticmethod
    def _validate_open_schema(handle, resolved_schema):
        file_schema = handle.attrs.get("schema_version")
        if resolved_schema == CURRENT_FILE_SCHEMA:
            if file_schema != CURRENT_FILE_SCHEMA:
                raise ValueError("current Stage06 Flow manifest/file schema versions must agree")
            if "valid_fraction" not in handle:
                raise ValueError("current Stage06 Flow schema requires valid_fraction")
        elif file_schema is not None:
            raise ValueError("explicit legacy Flow contract requires an unversioned HDF5 file")

    def _validate_structure(self, handle, entry, resolved_schema, verified_sha256):
        n = entry["frame_count"]
        self._validate_open_schema(handle, resolved_schema)
        current_schema = resolved_schema == CURRENT_FILE_SCHEMA
        for key, shape, kinds in (
            ("flow", (n, 2, 224, 224), "f"), ("valid_mask", (n, 1, 224, 224), "bu"),
            ("frame_index", (n,), "iu"), ("target_frame_index", (n,), "iu"),
            ("actual_delta_frames", (n,), "iu"), ("label_source", (n,), "iu"),
        ):
            value = handle[key]
            if value.shape != shape or value.dtype.kind not in kinds:
                raise ValueError(f"invalid flow field {key}: {value.shape}, {value.dtype}")
        if current_schema or "valid_fraction" in handle:
            if "valid_fraction" not in handle:
                raise ValueError("current Stage06 Flow schema requires valid_fraction")
            fractions = handle["valid_fraction"]
            if fractions.shape != (n,) or fractions.dtype.kind != "f":
                raise ValueError(f"invalid flow field valid_fraction: {fractions.shape}, {fractions.dtype}")
            values = fractions[:]
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("invalid flow valid_fraction values")
        expected = {"camera_key": self.camera_key, "merged_episode_index": entry["merged_episode_index"],
                    "source_episode_index": entry["source_episode_index"],
                    "flow_units": "normalized_source_image_extent", "flow_direction": "forward_only",
                    "tail_policy": "clamp",
                    "validity_semantics": "finite_and_forward_destination_in_bounds_not_occlusion"}
        for key, value in expected.items():
            if handle.attrs.get(key) != value:
                raise ValueError(f"flow metadata mismatch: {key}")
        if "dataset_id" in entry and handle.attrs.get("dataset_id") != entry["dataset_id"]:
            raise ValueError("flow metadata mismatch: dataset_id")
        if "timestamp" in handle:
            raise ValueError("unsupported Flow timestamp alias; require source_timestamp_s/target_timestamp_s/actual_delta_s")
        nominal = int(handle.attrs["nominal_delta_frames"])
        if nominal != self.delta_frames:
            raise ValueError(f"flow metadata mismatch: nominal_delta_frames={nominal}")
        fps = float(handle.attrs["fps"])
        if not np.isfinite(fps) or fps != self.expected_fps:
            raise ValueError("invalid flow fps")
        frames = handle["frame_index"][:]
        targets = handle["target_frame_index"][:]
        delta = handle["actual_delta_frames"][:]
        source = handle["label_source"][:]
        if n < 1 or len(np.unique(frames)) != n or (frames < 0).any():
            raise ValueError("duplicate/invalid frame_index")
        if not np.isin(targets, frames).all() or not np.array_equal(targets - frames, delta):
            raise ValueError("target frame crosses episode or delta mismatch")
        if (delta < 0).any() or (delta > self.delta_frames).any() or not np.isin(source, [1, 2]).all():
            raise ValueError("invalid delta or label_source")
        if ((source == 2) != (delta == 0)).any():
            raise ValueError("identity label_source must have delta zero")
        if self.contract is not None:
            from utils.aux_data_contract import validate_flow_file
            validate_flow_file(handle, entry, self.contract, frames, targets,
                               verified_sha256=verified_sha256)
        return frames

    @staticmethod
    def _stored_valid_fraction(handle, entry, row, resolved_schema):
        current_schema = resolved_schema == CURRENT_FILE_SCHEMA
        if "valid_fraction" not in handle:
            if current_schema:
                raise ValueError("current Stage06 Flow schema requires valid_fraction")
            return None
        values = handle["valid_fraction"]
        if values.shape != (entry["frame_count"],) or values.dtype.kind != "f":
            raise ValueError(f"invalid flow field valid_fraction: {values.shape}, {values.dtype}")
        value = float(values[row])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid flow valid_fraction value")
        return value

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self.rows.clear()

    def _evict_episode(self, episode):
        failed = self._handles.pop(episode, None)
        if failed is not None:
            failed.close()
        self.rows = {key: row for key, row in self.rows.items() if key[0] != episode}

    def _verify_episode_file(self, episode):
        path, entry = self.episodes[episode]
        expected = entry["sha256"]
        if not self._integrity_verifier.is_current(path, expected):
            self._evict_episode(episode)
        source = (f"dataset={self.dataset_id}, camera={self.camera_key}, episode={episode}, "
                  f"label={entry.get('label_identity', '')}")
        return self._integrity_verifier.verify(path, expected, source=source)

    def eligible_frames(self, min_valid_fraction):
        """Optional Slot+Flow sampling uses the same fixed-horizon pooled mask rule."""
        eligible = set()
        for episode in self.episodes:
            handle = self._handle(episode)
            _, entry = self.episodes[episode]
            resolved_schema = self.episode_schemas[episode]
            current_schema = resolved_schema == CURRENT_FILE_SCHEMA
            if current_schema and "valid_fraction" not in handle:
                raise ValueError(f"flow episode={episode}: current schema requires valid_fraction")
            for start in range(0, len(handle["frame_index"]), 32):
                stop = min(start + 32, len(handle["frame_index"]))
                masks = handle["valid_mask"][start:stop]
                if not np.isin(masks, [0, 1]).all():
                    raise ValueError(f"flow episode={episode}: invalid mask during eligibility scan")
                if "valid_fraction" in handle:
                    stored = np.asarray([
                        self._stored_valid_fraction(handle, entry, row, resolved_schema)
                        for row in range(start, stop)
                    ])
                    measured = masks.reshape(stop - start, -1).mean(axis=1)
                    if not np.allclose(stored, measured, rtol=0, atol=VALID_FRACTION_ATOL):
                        raise ValueError(f"flow episode={episode}: valid_fraction does not match valid_mask")
                pooled = masks.reshape(-1, 1, 56, 4, 56, 4).mean(axis=(3, 5))
                valid = (pooled >= min_valid_fraction).reshape(stop - start, -1).any(-1)
                valid &= handle["actual_delta_frames"][start:stop] == self.delta_frames
                valid &= handle["label_source"][start:stop] == self.label_source
                if self.contract and self.contract.get("excluded_frames"):
                    valid &= ~np.isin(handle["frame_index"][start:stop], self.contract["excluded_frames"].get(str(episode), []))
                eligible.update((episode, int(frame)) for frame in handle["frame_index"][start:stop][valid])
            self._verify_episode_file(episode)
        return eligible

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["rows"] = {}
        state["_pid"] = None
        return state

    def _handle(self, episode):
        import h5py

        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
            self._integrity_verifier.reset_for_process()
        verified_sha256 = self._verify_episode_file(episode)
        handle = self._handles.pop(episode, None)
        if handle is None:
            path, entry = self.episodes[episode]
            handle = h5py.File(path, "r")
            try:
                resolved_schema = self.episode_schemas[episode]
                self._validate_open_schema(handle, resolved_schema)
                frames = (handle["frame_index"][:] if self._audit_cache is not None else
                          self._validate_structure(handle, entry, resolved_schema, verified_sha256))
                mapping = {(episode, int(frame)): row for row, frame in enumerate(frames)}
            except Exception:
                handle.close()
                raise
            self.rows.update(mapping)
        self._handles[episode] = handle
        while len(self._handles) > self.max_handles:
            old_episode, old_handle = self._handles.popitem(last=False)
            old_handle.close()
            self.rows = {key: row for key, row in self.rows.items() if key[0] != old_episode}
        return handle

    def read(self, episode, frame):
        result = {"flow_supervision_available": torch.tensor(False),
                  "flow_episode_id": torch.tensor(episode), "flow_frame_index": torch.tensor(frame),
                  "flow_label_episode_id": torch.tensor(-1), "flow_target_frame_index": torch.tensor(-1),
                  "flow_actual_delta_frames": torch.tensor(-1), "flow_label_source": torch.tensor(-1),
                  "flow_actual_delta_s": torch.tensor(float("nan"), dtype=torch.float64),
                  "flow_source_timestamp_s": torch.tensor(float("nan"), dtype=torch.float64),
                  "flow_target_timestamp_s": torch.tensor(float("nan"), dtype=torch.float64),
                  "flow_dataset_id": self.dataset_id, "flow_camera": self.camera_key,
                  "flow_manifest_sha256": self.manifest_sha256,
                  "flow_generation_identity": "", "flow_label_identity": "",
                  "flow_label_file_sha256": "",
                  "flow_target": None, "flow_valid_mask": None}
        result["flow_nominal_delta_frames"] = torch.tensor(self.delta_frames)
        result["flow_fps"] = torch.tensor(self.expected_fps)
        result["flow_exclusion_reason"] = torch.tensor(1)
        if episode not in self.episodes:
            return result
        try:
            handle = self._handle(episode)
            row = self.rows.get((episode, frame))
            if row is None:
                self._verify_episode_file(episode)
                return result
            if int(handle["frame_index"][row]) != frame:
                raise ValueError("flow frame_index changed after manifest validation")
            _, entry = self.episodes[episode]
            stored_fraction = self._stored_valid_fraction(
                handle, entry, row, self.episode_schemas[episode])
            delta, source = int(handle["actual_delta_frames"][row]), int(handle["label_source"][row])
            target_frame = int(handle["target_frame_index"][row])
            source_time = (float(handle["source_timestamp_s"][row]) if "source_timestamp_s" in handle
                           else frame / self.expected_fps)
            target_time = (float(handle["target_timestamp_s"][row]) if "target_timestamp_s" in handle
                           else target_frame / self.expected_fps)
            actual_delta_s = (float(handle["actual_delta_s"][row]) if "actual_delta_s" in handle
                              else target_time - source_time)
            result.update(flow_actual_delta_frames=torch.tensor(delta), flow_label_source=torch.tensor(source),
                          flow_label_episode_id=torch.tensor(int(entry["merged_episode_index"])),
                          flow_target_frame_index=torch.tensor(target_frame),
                          flow_source_timestamp_s=torch.tensor(source_time, dtype=torch.float64),
                          flow_target_timestamp_s=torch.tensor(target_time, dtype=torch.float64),
                          flow_actual_delta_s=torch.tensor(actual_delta_s, dtype=torch.float64),
                          flow_generation_identity=str(entry.get("generation_identity") or ""),
                          flow_label_identity=str(entry.get("label_identity") or ""),
                          flow_label_file_sha256=str(entry.get("sha256") or ""))
            result["flow_exclusion_reason"] = torch.tensor(2)
            mask = handle["valid_mask"][row]
            if not np.isin(mask, [0, 1]).all():
                raise ValueError("nonbinary flow mask")
            if stored_fraction is not None:
                measured_fraction = float(mask.mean())
                if not np.isclose(stored_fraction, measured_fraction, rtol=0,
                                  atol=VALID_FRACTION_ATOL):
                    raise ValueError("valid_fraction does not match valid_mask")
            excluded = self.contract.get("excluded_frames", {}).get(str(episode), []) if self.contract else []
            if int(frame) in excluded:
                result["flow_exclusion_reason"] = torch.tensor(5)
                self._verify_episode_file(episode)
                return result
            if delta != self.delta_frames or source != self.label_source:
                self._verify_episode_file(episode)
                return result
            flow = handle["flow"][row].astype(np.float32)
            if not np.isfinite(flow).all():
                raise ValueError("nonfinite flow")
            result["flow_exclusion_reason"] = torch.tensor(0 if mask.any() else 3)
            result.update(flow_supervision_available=torch.tensor(True),
                          flow_target=torch.from_numpy(flow), flow_valid_mask=torch.from_numpy(mask.astype(bool)))
            self._verify_episode_file(episode)
            return result
        except Exception as error:
            self._evict_episode(episode)
            raise DatasetIntegrityError(f"flow episode={episode} frame={frame}: {error}") from error
