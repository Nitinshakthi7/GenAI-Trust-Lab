# utils.py - small helpers shared by every file in model/.
# It holds: the ONE random seed for the whole project, the folder paths,
# a function to pick the GPU (or CPU), and a function to save a JSON file.
# Nothing here trains anything.

import json
import random
from pathlib import Path

import numpy as np
import torch

# The one and only place the seed lives. Change it here, not anywhere else.
SEED = 42

# Folder paths are built from the project root, so they work on any computer.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"   # trained model files (ignored by git)
FAKES_DIR = ROOT / "fakes"       # generated images (ignored by git)
CONFIGS_DIR = ROOT / "configs"   # one settings file per run
RESULTS_DIR = ROOT / "results"   # measured numbers per run
LOGS_DIR = ROOT / "logs"         # training logs (ignored by git)


def set_seed(seed=SEED):
    """Make random numbers repeatable (Python, numpy and torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Use the GPU if there is one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model):
    """How many numbers (weights) the model has to learn."""
    return sum(p.numel() for p in model.parameters())


def save_json(path, data):
    """Save a dict as a readable JSON file. Makes the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
