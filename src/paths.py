"""Project paths, resolved relative to the repository root (independent of the working directory)."""

import os
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parents[1]
DATA_DIR     = REPO_ROOT / "data"
FIGURES_ROOT = REPO_ROOT / "figures"


def fig_dir(subdir: str = "") -> str:
    """Return (and create) the figure output directory ``figures/<subdir>``."""
    path = FIGURES_ROOT / subdir
    os.makedirs(path, exist_ok=True)
    return str(path)
