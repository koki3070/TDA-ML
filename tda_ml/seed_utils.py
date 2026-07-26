"""Utilities for reproducible random seed initialization."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_algorithms: bool = False) -> None:
    """Set random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
        # Required by PyTorch for deterministic cublas behavior on CUDA.
        # Keep this local to the process for reproducible runs.
        import os

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
