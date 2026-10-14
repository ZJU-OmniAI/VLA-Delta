"""Raw registry defaults. Runtime entrypoints explicitly resolve configured paths."""
from pathlib import Path
import yaml

DATASET2FEATURE = yaml.safe_load((Path(__file__).resolve().parents[1] / "dataset2feature.yaml").read_text())
