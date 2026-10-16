# RoboTwin Evaluation

This workflow targets the dual-arm ALOHA AgileX robot in RoboTwin 2.0, using a separate model server and simulation client. First install the `vla-delta` model environment following the [main README](../../README.md#environment-setup). Run commands from the repository root except where explicitly instructed to enter the assets directory.

## 1. Download the Simulator Source

Clone the latest source from RoboTwin's default branch into `third_party/robotwin`:

```bash
mkdir -p third_party
git clone --filter=blob:none https://github.com/RoboTwin-Platform/RoboTwin.git third_party/robotwin
```

Skip cloning if you have already downloaded the source. This command retrieves the latest upstream default branch at the time of cloning. `third_party/` is excluded from this repository's Git history; evaluation requires the tasks, language descriptions, configuration, and assets in that directory.

## 2. Install the Simulation Environment and Assets

The client uses a separate Python 3.10 environment.

```bash
conda create -n vla-robotwin -c conda-forge python=3.10 pip ffmpeg libvulkan-loader vulkan-tools -y
conda activate vla-robotwin
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME=/usr/local/cuda  # Replace with your local CUDA 12.1 toolkit directory
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
bash scripts/robotwin/install.sh
```

Download and extract the assets, then configure the planner paths:

```bash
cd third_party/robotwin/assets
python _download.py
unzip -q embodiments.zip
unzip -q objects.zip
unzip -q background_texture.zip
cd ../../..
python scripts/robotwin/configure_assets.py
```

## 3. Start the Model Server

Run the following in the first terminal, replacing the statistics path:

```bash
conda activate vla-delta
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLA_EVAL_CHECKPOINT="$PWD/checkpoints/VLA-Delta-robotwin"
python server.py --config configs/eval/robotwin_server.yaml \
  --stats_file /absolute/path/to/your/robotwin_dataset/meta/stats.json
```

## 4. Start Evaluation

Open a second terminal, enter the repository root, and run:

```bash
conda activate vla-robotwin
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python scripts/evaluate.py --config configs/eval/robotwin.yaml --execute
```

This command starts task evaluation directly, without additional rendering checks or small-scale evaluations. Defaults come from the [client configuration](../../configs/eval/robotwin.yaml):

| Argument | Default | Description |
| --- | --- | --- |
| `--tasks` | All 50 tasks | Specify task names to evaluate a subset |
| `--profiles` | `demo_clean demo_randomized` | Clean and randomized environments |
| `--trials` | `100` | Episodes per task and profile; 10,000 episodes in total |
| `--execution-horizon` | `16` | Action steps executed after each prediction; must not exceed the model's prediction horizon |
| `--seed` | `0` | Client seed parameter; the initial candidate seed is `100000 × (1 + seed)` |
| `--host` / `--port` | `127.0.0.1` / `8000` | Must match the model server |
| `--output-dir` | `outputs/eval/robotwin` | Directory for evaluation results |
| `--continue-on-error` | Disabled | Continue other task/profile groups after an error; any failed group keeps the run incomplete and the final exit code nonzero |

The installer applies reproducible expert-planning guards for unreachable grasps, empty grasp candidate lists, and cup variants without the contact index requested by `place_container_plate`. These scenes are rejected through the existing expert feasibility filter. Missing files or malformed asset metadata still raise errors. To update an existing simulator checkout, run `python scripts/robotwin/patch_tasks.py --apply`; original source backups are retained.
