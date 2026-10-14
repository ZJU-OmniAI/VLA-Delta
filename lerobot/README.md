# LeRobot dataset subset

This is the bundled LeRobot v2.0/v2.1 reader used by VLA-∆. It is included in the repository at `lerobot/` and retains dataset metadata, frame loading, video decoding, statistics, serialization helpers and their import dependencies. It is not a full LeRobot installation.

The project entry points prepend this directory to `sys.path`. Install the [training requirements](../requirements/train.txt); no separate LeRobot download or patch application is needed. Do not install an unrelated LeRobot version over this copy. For direct Python imports use `PYTHONPATH=.:lerobot` from the repository root.

The video encoding helper in `lerobot/common/datasets/video_utils.py` resolves `ffmpeg` from the active `PATH` and reports a missing executable before creating output. Activate an environment with FFmpeg installed to use this helper; decoding existing videos keeps its current behavior.

Original Apache-2.0 [license](LICENSE) and file headers are retained. The source is compatible with the [ZR-0 snapshot at b1440d4cf27624da2b1aa31268637cf46601c15d](https://github.com/RUCKBReasoning/ZR-0/tree/b1440d4cf27624da2b1aa31268637cf46601c15d/lerobot), with the project modifications below included directly in this source tree. Upstream version is locally labeled 0.1.0; the original Hugging Face import commit is unknown. See [third-party notices](../THIRD_PARTY_NOTICES.md).

The `Robot` typing protocol is kept in `lerobot/configs/types.py`; concrete robot drivers and policy implementations are omitted. The dataset reader and existing creation/statistics helpers remain together to avoid rewriting their behavior.

## Local modifications

- `lerobot/common/datasets/lerobot_dataset.py`: optional statistics loading, configurable Parquet loading workers, materialization of dataset columns before `torch.stack`, and corrected boundaries when loading an episode subset.
- `lerobot/common/datasets/video_utils.py`: PyAV decoding with configurable thread counts and resource cleanup, retaining timestamp matching and tolerance checks.
- `lerobot/configs/types.py`, `lerobot/common/datasets/lerobot_dataset.py` and `lerobot/common/datasets/utils.py`: the dataset-facing `Robot` protocol is available without importing hardware drivers.

Maintain these source files directly and update this description when their behavior changes. No separate patch file is maintained.
