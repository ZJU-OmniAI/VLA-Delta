# Normalization Statistics

This directory provides the statistics calculation entry point and example statistics used by the released LIBERO model. **When fine-tuning your own model, specify your training dataset and statistics paths. Evaluation must use the same statistics as training.**

```text
stats/
├── calculate_global_stats.py
├── README.md
└── demo_libero/
    └── meta/
        ├── stats.json
        ├── info.json
        ├── tasks.jsonl
        └── episodes.jsonl
```

`demo_libero/` contains only statistics and metadata, with no training videos or Parquet files, so it cannot serve as a fine-tuning dataset. Its four files were copied unchanged from the [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero) training dataset referenced by the released model's original training manifest; the statistics were not recomputed. The normalization quantiles in `stats.json` match that checkpoint's training record (loading compares values rather than hashes), and `info.json` uses the training camera fields `image` / `image2`. Follow the upstream dataset's usage and redistribution terms.

Compute RoboTwin statistics on your training data as described below, and specify the actual `meta/stats.json` path for evaluation. The accompanying metadata is also read from that dataset's `meta/` directory. This repository does not provide a default RoboTwin statistics package.

## 1. Compute Training Statistics

First install the `vla-delta` environment using the [main README](../README.md#environment-setup). The required LeRobot data modules and project adaptations are included in the root `lerobot/` directory; no separate download is needed. Run the following commands from the repository root. **Replace every `/absolute/path/to/...` placeholder with your actual absolute path.**

Data must use the LeRobot v2.0/v2.1 format supported by this project: `meta/info.json` must exist, and `data/**/*.parquet` must contain numeric `observation.state` and `action` vectors.

```bash
conda activate vla-delta
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python stats/calculate_global_stats.py \
  --dataset_path /absolute/path/to/your/training_dataset
```

The output path is fixed:

```text
/absolute/path/to/your/training_dataset/meta/stats.json
```

The script reads all `data/**/*.parquet` files and computes per-dimension `mean`, `std`, `min`, `max`, `q01`, and `q99` over every frame's state/action values. It needs enough memory for these numeric arrays but does not read video pixels. The current LIBERO configuration normalizes with `q01`/`q99`; evaluation uses the same bounds to denormalize model actions.

If `meta/stats.json` already exists, the script reads and reuses it without overwriting it. To recompute statistics, first back up or move the original file and ensure no training or resume job depends on it. There is no separate output-path argument, and results are not automatically written to `stats/demo_libero/`.

LIBERO and RoboTwin share this statistics entry point, which also accepts `--dataset-path`. The old `scripts/prepare_libero.py` entry point has been removed.

## 2. Use Statistics During Training

**Training reads statistics from `<dataset_path>/meta/stats.json`. Set your dataset root in the registry and ensure that the file computed above is at that location.** The training entry point has no separate `--stats_file` argument, and you do not need to copy statistics into `demo_libero/`.

Edit the root `dataset2feature.yaml` directly, or create a private copy for the first time:

```bash
cp -n dataset2feature.yaml dataset2feature.local.yaml
```

In that copy, update the `libero_wo_ecot_pt` entry and retain the other fields:

```yaml
libero_wo_ecot_pt:
  dataset_path: /absolute/path/to/your/training_dataset  # Set this to your actual path
  dataset_type: vla
  sample_ratio: 1.0
  use_quantile: true
```

The corresponding normalization file must be `/absolute/path/to/your/training_dataset/meta/stats.json`. Camera, task, and episode metadata also come from that training dataset; do not overwrite them with the entire `demo_libero/meta/` directory. Relative YAML paths resolve against the registry file's directory. The private copy is ignored by Git.

After preparing the base weights and training data, run these commands from the repository root. W&B is disabled by default, so login is not required:

```bash
export VLA_PRETRAIN_CHECKPOINT="$PWD/checkpoints/VLA-Delta-base"
export VLA_OUTPUT_ROOT="$PWD/outputs"
python scripts/train.py --config configs/finetune/libero.yaml \
  --dataset_registry "$PWD/dataset2feature.local.yaml" --dry-run
python scripts/train.py --config configs/finetune/libero.yaml \
  --dataset_registry "$PWD/dataset2feature.local.yaml" --execute
```

`--dry-run` checks only the configuration and launch command; it does not verify that all data or weights can load. `--execute` starts training. Omit `--dataset_registry` if you edited the original registry directly. See [Fine-Tuning VLA-Δ](../README.md#fine-tuning) for commands and arguments, and [Saving and Resuming](../README_v1.md#5-saving-and-resuming) for checkpoint recovery.

## 3. Use Statistics During Evaluation

**When evaluating your own fine-tuned model, set `--stats_file` to the actual `meta/stats.json` used during training**, and select your fine-tuned checkpoint:

```bash
export VLA_EVAL_CHECKPOINT=/absolute/path/to/your/libero_checkpoint
python server.py --config configs/eval/libero_server.yaml \
  --stats_file /absolute/path/to/your/training_dataset/meta/stats.json
```

Alternatively, edit `configs/eval/libero_server.yaml`:

```yaml
stats_file: /absolute/path/to/your/training_dataset/meta/stats.json  # Set this to your actual path
```

CLI paths resolve against the current working directory; YAML paths resolve against the configuration file's directory. Absolute paths are the most direct option. A missing statistics file causes an error, with no automatic recomputation or fallback. The server checks statistics against the checkpoint's training record. Do not compute new statistics from evaluation episodes or replace camera metadata to bypass validation.

When the statistics file is in a complete `meta/` directory, the loader first looks there for `info.json`, `tasks.jsonl`, and `episodes.jsonl`. Released models therefore need no bundled `inference_metadata/` snapshot, training videos, or Parquet files. To prepare evaluation resources, copy all four files together from your training dataset's `meta/` directory, or point directly to its `meta/stats.json`. If you provide only a standalone statistics file, the checkpoint or registered dataset path must still provide the accompanying metadata.

Keep the default `stats_file: ../../stats/demo_libero/meta/stats.json` only when evaluating the maintainer's matching released LIBERO model. For your own fine-tuned model, explicitly specify its statistics path. The RoboTwin server likewise uses `stats_file` in `configs/eval/robotwin_server.yaml` or `--stats_file` to select training statistics; LIBERO statistics cannot replace RoboTwin statistics.

The released `checkpoints/VLA-Delta-robotwin/` no longer contains bundled `dataset_artifacts/`, and its default dataset entry is `robotwin_aloha_agilex`. It requires the original training metadata for three cameras and 14-dimensional joint states/actions, along with matching q01/q99 statistics. As with your own fine-tuned LIBERO model, compute training statistics as described above and set the actual path through `--stats_file` or the server YAML. The path in `robotwin_server.yaml` is a placeholder to replace, not a statistics file supplied by this repository. The same directory should contain `info.json`, `tasks.jsonl`, and `episodes.jsonl`:

```bash
export VLA_EVAL_CHECKPOINT="$PWD/checkpoints/VLA-Delta-robotwin"
python server.py --config configs/eval/robotwin_server.yaml \
  --stats_file /absolute/path/to/your/robotwin_dataset/meta/stats.json
```

Released model manifests retain normalization bounds only for compatibility comparisons. These do not replace external statistics files, and checkpoint hashes are not validated. Use the method above to compute statistics for new fine-tuning; when evaluating an existing model, use the same statistics as its training run.

The commands above start only the model server. In another terminal, follow the [LIBERO evaluation guide](../evaluation/libero/README.md) or [RoboTwin evaluation guide](../evaluation/robotwin/README.md) to start the corresponding simulation client. Evaluation records include the actual statistics file path and SHA-256. Output locations are specified by `--output-dir` in each evaluation guide.

New training runs do not copy these four files into checkpoints or generate `inference_metadata.json` by default. To evaluate your model, set `--stats_file` to the training `<dataset_path>/meta/stats.json` and keep the other three metadata files alongside it. To bundle them with a model, explicitly add `--export_inference_metadata` to the training command. Bundled snapshots in older checkpoints remain supported.
