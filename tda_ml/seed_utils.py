"""Utilities for reproducible random seed initialization."""

from __future__ import annotations

import random
from typing import Any

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


def resolve_model_seed(config: dict[str, Any]) -> int:
    """Return model/init seed; defaults to ``data.seed`` when unset.

    ``data.seed`` owns dataset sampling (``noise_seed``). ``training.model_seed``
    owns parameter init and training RNG. Separating them lets rings Methods
    retry cliff failures without changing the evaluation clouds.
    """
    data_cfg = config.get("data") or {}
    if "seed" not in data_cfg:
        raise ValueError("data.seed must be set explicitly; refusing silent seed=42")
    data_seed = int(data_cfg["seed"])
    training = config.get("training") or {}
    if "model_seed" not in training or training["model_seed"] is None:
        return data_seed
    return int(training["model_seed"])


def cliff_init_model_seeds(data_seed: int, *, n_restarts: int) -> list[int]:
    """Deterministic init-seed schedule for val_topo cliff multi-restart.

    Attempt 0 uses ``data_seed`` (legacy coupling; keeps easy seeds fast).
    Later attempts stride by a large prime. Offsets are ordered to explore
    distant basins early (seed-123 probe: offset 7 crossed the cliff; 1–6 did
    not), instead of walking neighbors of a failed coupled init first.
    """
    if n_restarts < 1:
        raise ValueError(f"n_restarts must be >= 1; got {n_restarts}")
    stride = 100_003
    preferred = [
        0,
        7,
        1,
        14,
        3,
        11,
        5,
        9,
        2,
        4,
        6,
        8,
        10,
        12,
        13,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
    ]
    idxs: list[int] = []
    for i in preferred:
        if i < n_restarts and i not in idxs:
            idxs.append(i)
    for i in range(int(n_restarts)):
        if i not in idxs:
            idxs.append(i)
    return [int(data_seed) + i * stride for i in idxs]
