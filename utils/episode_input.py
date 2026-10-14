"""Opt-in, immutable episode selection and input-language mapping for LeRobot v2."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class EpisodeInput:
    episode_ids: tuple[int, ...]
    instructions: dict[int, str]
    contract: dict


def validate_input_instruction(text, contract):
    """Require the same language prefix at deployment as in training."""
    if contract is None:
        return text
    prefix = contract["instruction_prefix"]
    if not isinstance(text, str) or not text.startswith(prefix) or not text[len(prefix):].strip():
        raise ValueError(f"episode input requires a non-empty instruction prefixed with {prefix!r}")
    if text[len(prefix):].startswith(prefix):
        raise ValueError("episode input instruction has a duplicated prefix")
    return text


def resolve_episode_input(dataset_entry, entry):
    """Read only when explicitly enabled; bind selection, language and statistics."""
    filename = entry.get("episode_input_manifest")
    if filename is None:
        if entry.get("episode_input_manifest_sha256") is not None:
            raise ValueError("episode input hash requires episode_input_manifest")
        return None
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry.get("episode_input_manifest_sha256"):
        raise ValueError("episode input manifest SHA-256 mismatch")
    payload = json.loads(raw)
    if payload.get("version") != 1 or payload.get("instruction_source") != "episode_remarks":
        raise ValueError("unsupported episode input manifest")
    if payload.get("selection") != "all_episodes_train" or payload.get("validation_episodes") != []:
        raise ValueError("this episode input version only supports audited all-episode training")
    prefix = payload.get("instruction_prefix")
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("episode input requires an explicit non-empty prefix")
    task = payload.get("datasets", {}).get(dataset_entry)
    root = Path(entry["dataset_path"]).expanduser().resolve()
    if not isinstance(task, dict) or root != Path(task["dataset_path"]).resolve():
        raise ValueError(f"{dataset_entry}: episode input dataset path mismatch")
    metadata_hashes = task.get("metadata_sha256", {})
    required = {"info.json", "episodes.jsonl", "tasks.jsonl", "stats.json", "modality.json"}
    if set(metadata_hashes) != required:
        raise ValueError("episode input must bind info, episodes, tasks, stats and modality")
    content = {}
    for name in required:
        content[name] = (root / "meta" / name).read_bytes()
        if hashlib.sha256(content[name]).hexdigest() != metadata_hashes[name]:
            raise ValueError(f"{dataset_entry}: episode input metadata changed: {name}")
    ids = task.get("train_episode_ids")
    if (not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids)
            or ids != sorted(set(ids))):
        raise ValueError("episode input requires sorted unique integer episode IDs")
    episodes = [json.loads(line) for line in content["episodes.jsonl"].splitlines() if line.strip()]
    if ids != [e["episode_index"] for e in episodes]:
        raise ValueError("episode input all-episode selection differs from source metadata")
    info = json.loads(content["info.json"])
    stats = json.loads(content["stats.json"])
    count = sum(e["length"] for e in episodes)
    if (count != task.get("train_frames") or count != info["total_frames"]
            or len(ids) != info["total_episodes"]
            or stats.get("total_trajectory_length") != count
            or stats.get("num_trajectories") != len(ids)):
        raise ValueError("episode input selection and normalization population differ")
    if entry.get("use_quantile") is not False or entry.get("sample_ratio") != 1.0:
        raise ValueError("this episode input contract requires all frames and per-task min/max")
    instructions = {}
    for episode in episodes:
        text = episode.get("remarks")
        if not isinstance(text, str) or not text.strip() or text.startswith(prefix):
            raise ValueError(f"episode {episode['episode_index']}: invalid remarks instruction")
        instructions[episode["episode_index"]] = prefix + text.strip()
    contract = dict(version=1, manifest_sha256=digest, dataset_entry=dataset_entry,
                    selection=payload["selection"], train_episodes=len(ids), train_frames=count,
                    instruction_source="episode_remarks", instruction_prefix=prefix,
                    metadata_sha256=metadata_hashes, normalization="per_task_min_max")
    return EpisodeInput(tuple(ids), instructions, contract)
