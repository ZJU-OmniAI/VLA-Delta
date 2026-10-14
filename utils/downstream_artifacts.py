"""Self-contained multi-dataset snapshots and transactional downstream exports."""
import json
import os
from pathlib import Path
import shutil
import uuid

from utils.downstream_contracts import checkpoint_version, validate_entry
from utils.downstream_training_runtime import atomic_json
from utils.checkpoint_metadata import checkpoint_metadata, validate_normalization_snapshot

META_FILES = ("info.json", "stats.json", "episodes.jsonl", "tasks.jsonl", "modality.json")


def safe_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or (root / relative).is_symlink():
        raise ValueError(f"invalid artifact path: {relative}")
    return path


def inventory(root):
    return {str(p.relative_to(root)): dict(size=p.stat().st_size)
            for p in sorted(root.rglob('*')) if p.is_file()}


def check_inventory(root, files):
    for name, record in files.items():
        path = safe_file(root, name)
        if path.stat().st_size != record['size']:
            raise ValueError(f"artifact size mismatch: {name}")


def save_dataset_artifacts(checkpoint, manifest):
    checkpoint = Path(checkpoint)
    index = dict(version=2, entries=[])
    for number, entry in enumerate(manifest['entries']):
        relative = f'dataset_artifacts/entries/{number:03d}'
        target = checkpoint / relative
        (target / 'meta').mkdir(parents=True)
        source = Path(entry['dataset_path']) / 'meta'
        for name in META_FILES:
            shutil.copy2(source / name, target / 'meta' / name)
        if (source / 'metadata.json').is_file():
            shutil.copy2(source / 'metadata.json', target / 'meta/metadata.json')
        atomic_json(target / 'vision_contract.json', entry['vision_input_contract'])
        atomic_json(target / 'episode_input_contract.json', checkpoint_metadata(entry['episode_input_contract']))
        episodes = [json.loads(line) for line in (target / 'meta/episodes.jsonl').read_text().splitlines()]
        atomic_json(target / 'instruction_selection.json', [dict(episode_id=e['episode_index'],
            instruction=entry['episode_input_contract']['instruction_prefix'] + e['remarks'].strip()) for e in episodes])
        index['entries'].append(dict(dataset_entry=entry['dataset_entry'], directory=relative,
                                    files=inventory(target)))
    atomic_json(checkpoint / 'dataset_artifacts/index.json', index)
    atomic_json(checkpoint / 'downstream_contract.json', dict(downstream_contract_version=2))
    config = json.loads((checkpoint / 'config.json').read_text())
    config['downstream_contract_version'] = 2
    atomic_json(checkpoint / 'config.json', config)
    validate_dataset_artifacts(checkpoint, live=True)


def validate_dataset_artifacts(checkpoint, *, live=False):
    from utils.dataset_manifest import load_resolved_dataset_manifest
    checkpoint = Path(checkpoint)
    if checkpoint_version(checkpoint) != 2:
        raise ValueError('multi-dataset artifact validation requires v2')
    manifest = load_resolved_dataset_manifest(checkpoint)
    index = json.loads((checkpoint / 'dataset_artifacts/index.json').read_text())
    entries = manifest['entries']
    if index.get('version') != 2 or [e['dataset_entry'] for e in index['entries']] != [e['dataset_entry'] for e in entries]:
        raise ValueError('artifact task set/order differs from resolved manifest')
    if len(set(e['dataset_entry'] for e in entries)) != len(entries):
        raise ValueError('duplicate dataset entry')
    result = {}
    for entry, saved in zip(entries, index['entries']):
        root = (checkpoint / saved['directory']).resolve()
        if not root.is_relative_to(checkpoint.resolve()) or root.is_symlink():
            raise ValueError('artifact directory escapes checkpoint')
        required = {f'meta/{name}' for name in META_FILES} | {'vision_contract.json', 'episode_input_contract.json', 'instruction_selection.json'}
        if not required.issubset(saved['files']):
            raise ValueError('incomplete dataset artifact index')
        check_inventory(root, saved['files'])
        info = json.loads((root / 'meta/info.json').read_text())
        validate_entry(entry, info)
        if json.loads((root / 'vision_contract.json').read_text()) != entry['vision_input_contract']:
            raise ValueError('snapshot vision contract differs')
        contract = checkpoint_metadata(entry['episode_input_contract'])
        if checkpoint_metadata(json.loads((root / 'episode_input_contract.json').read_text())) != contract:
            raise ValueError('snapshot episode contract differs')
        validate_normalization_snapshot(json.loads((root / 'meta/stats.json').read_text()), entry)
        for name in META_FILES:
            if live and (root / 'meta' / name).read_bytes() != (Path(entry['dataset_path']) / 'meta' / name).read_bytes():
                raise ValueError(f"{entry['dataset_entry']}: live metadata changed: {name}")
        episodes = [json.loads(line) for line in (root / 'meta/episodes.jsonl').read_text().splitlines()]
        instructions = [dict(episode_id=e['episode_index'], instruction=contract['instruction_prefix'] + e['remarks'].strip()) for e in episodes]
        if json.loads((root / 'instruction_selection.json').read_text()) != instructions:
            raise ValueError('saved language selection differs from episode remarks')
        if len(episodes) != contract['train_episodes'] or sum(e['length'] for e in episodes) != contract['train_frames']:
            raise ValueError('saved episode selection differs')
        result[entry['dataset_entry']] = dict(directory=str(root), stats_file=str(root / 'meta/stats.json'))
    return result


def sync_tree(root):
    for path in root.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_export(directory):
    from utils.downstream_training_runtime import validate_model_artifacts
    directory = Path(directory)
    receipt = json.loads((directory / 'model_export_complete.json').read_text())
    if receipt.get('version') != 2:
        raise ValueError('missing model export completion')
    check_inventory(directory, receipt['files'])
    required = {'config.json','preprocessor_config.json','tokenizer.json','tokenizer_config.json',
                'action_expert.safetensors','action_expert_config.json','difference_query.safetensors',
                'difference_query_config.json','resolved_dataset_manifest.json','dataset_artifacts/index.json'}
    if not required.issubset(receipt['files']):
        raise ValueError('incomplete inference model export')
    if 'model.safetensors.index.json' in receipt['files']:
        shards = json.loads((directory/'model.safetensors.index.json').read_text())['weight_map'].values()
        if not set(shards).issubset(receipt['files']):
            raise ValueError('missing exported VLM shards')
    elif 'model.safetensors' not in receipt['files']:
        raise ValueError('missing exported VLM weights')
    validate_model_artifacts(directory, live=False)
    return receipt


def publish_export(root, checkpoint, step, identity):
    root, checkpoint = Path(root), Path(checkpoint)
    # Stable for publication retries, without exposing update counts in the name.
    export_id = uuid.uuid5(uuid.NAMESPACE_URL, str(checkpoint.resolve())).hex
    destination = root / f'checkpoint-{export_id}'
    if destination.exists():
        receipt = validate_export(destination)
        if receipt['experiment_id'] != identity['experiment_id'] or receipt['source_checkpoint'] != str(checkpoint):
            raise ValueError('refuse to replace a different published model export')
        return destination
    pending = root / f'.model-pending-{uuid.uuid4().hex}'
    pending.mkdir()
    # All inference files and contracts, but no native optimizer/RNG payloads.
    for path in checkpoint.iterdir():
        if path.is_dir() and path.name == 'dataset_artifacts':
            shutil.copytree(path, pending/path.name)
        elif path.is_file() and (path.suffix in {'.json', '.safetensors', '.txt', '.model', '.jinja'}):
            shutil.copy2(path, pending/path.name)
    files = inventory(pending)
    source_files = json.loads((checkpoint.parent/'checkpoint_complete.json').read_text())['files']
    for name, record in files.items():
        if checkpoint_metadata(source_files.get(name)) != record:
            raise ValueError(f'inference export differs from completed recovery: {name}')
    atomic_json(pending/'model_export_complete.json', dict(version=2, experiment_id=identity['experiment_id'],
        step=step, source_checkpoint=str(checkpoint), files=files))
    validate_export(pending)
    sync_tree(pending)
    pending.rename(destination)
    return destination


def finish_publication(root, checkpoint, step, consumed_windows, identity, *, export):
    """Idempotently finish a published recovery, commit index, then collect old versions."""
    root, checkpoint = Path(root), Path(checkpoint)
    from utils.downstream_training_runtime import validate_complete_checkpoint
    validate_complete_checkpoint(checkpoint, identity=identity)
    model = publish_export(root, checkpoint, step, identity) if export else None
    index_path = root/'checkpoint_index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else dict(version=2, experiment_id=identity['experiment_id'], records=[])
    if index['experiment_id'] != identity['experiment_id']:
        raise ValueError('foreign checkpoint publication index')
    record = dict(checkpoint=str(checkpoint), model_export=str(model) if model else None,
                  step=step, consumed_windows=consumed_windows, complete=True)
    records = [r for r in index['records'] if r['checkpoint'] != str(checkpoint)] + [record]
    index['records'] = records
    atomic_json(index_path, index)
    atomic_json(root/'latest_complete.json', record)
    # Only index-managed versions of this experiment are eligible; never glob-delete.
    retained = {r['checkpoint'] for r in records[-2:]}
    for item in records[:-2]:
        path = Path(item['checkpoint'])
        if str(path) in retained or not path.exists():
            continue
        if path.parent.parent.resolve() != (root/'recovery_checkpoints').resolve() or path.parent.is_symlink():
            raise ValueError('refusing recovery deletion outside this experiment')
        validate_complete_checkpoint(path, identity=identity)
        if item['model_export']:
            validate_export(item['model_export'])
        try:
            shutil.rmtree(path.parent)
        except OSError as exc:
            atomic_json(root/'retention_warning.json', dict(path=str(path), error=str(exc)))
    return record
