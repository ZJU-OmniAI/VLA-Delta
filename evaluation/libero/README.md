# LIBERO Evaluation

This workflow uses a separate model server and LIBERO simulation client. Run all commands from the repository root. First install the `vla-delta` model environment following the [main README](../../README.md#environment-setup).

## 1. Install the Simulation Environment

Install the LIBERO client in a Python 3.8 environment, separate from the model environment:

```bash
conda create -n vla-libero -c conda-forge python=3.8.13 pip libgl libegl -y
conda activate vla-libero
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install torch==1.11.0+cpu torchvision==0.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/libero.txt
mkdir -p third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
python -m pip install -e ./third_party/libero --no-deps
python scripts/setup_libero.py --libero-root ./third_party/libero --config-dir ./data/libero-config
```

## 2. Start the Model Server

Run the following in the first terminal:

```bash
conda activate vla-delta
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLA_EVAL_CHECKPOINT="$PWD/checkpoints/VLA-Delta-libero"
python server.py --config configs/eval/libero_server.yaml
```

To evaluate your own model, set the checkpoint path above to its actual directory and replace the final command with:

```bash
python server.py --config configs/eval/libero_server.yaml \
  --stats_file /absolute/path/to/your/training_dataset/meta/stats.json
```

## 4. Start Evaluation

Open a second terminal, enter the repository root, and run:

```bash
conda activate vla-libero
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBERO_CONFIG_PATH="$PWD/data/libero-config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python scripts/evaluate.py --config configs/eval/libero.yaml --execute
```

This command starts the full evaluation directly, without an additional small-scale evaluation.

| Argument | Default | Description |
| --- | --- | --- |
| `--suites` | `libero_spatial libero_object libero_goal libero_10` | Four suites, each containing 10 tasks |
| `--trials` | `50` | Episodes per task; 2,000 episodes in total |
| `--execution-horizon` | `10` | Execute 10 consecutive steps after each prediction |
| `--seed` | `7` | Client random seed |
| `--host` / `--port` | `127.0.0.1` / `8000` | Must match the model server |
| `--output-dir` | `outputs/eval/libero` | Directory for evaluation results |
