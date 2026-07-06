"""Tests for train-topo checkpoint selection (saved epochs only)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.checkpoint_selection import (
    best_topo_checkpoint,
    best_topo_epoch,
    list_checkpoint_epochs,
)


def test_best_topo_epoch_restricted_to_saved(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True)
    metrics = log_dir / "metrics.csv"
    with metrics.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_topo_loss"])
        w.writerow([28, 0.09])
        w.writerow([29, 0.05])
        w.writerow([30, 0.06])

    (run_dir / "checkpoint_epoch_30.pth").write_bytes(b"x")

    saved = list_checkpoint_epochs(run_dir)
    assert saved == {30}

    ep, loss = best_topo_epoch(metrics, saved_epochs=saved)
    assert ep == 30
    assert loss == pytest.approx(0.06)

    global_ep, global_loss = best_topo_epoch(metrics)
    assert global_ep == 29
    assert global_loss == pytest.approx(0.05)

    ckpt, ep, loss, gep, gt = best_topo_checkpoint(run_dir)
    assert ckpt == "checkpoint_epoch_30.pth"
    assert ep == 30
    assert loss == pytest.approx(0.06)
    assert gep == 29
    assert gt == pytest.approx(0.05)


def test_best_topo_epoch_prefers_val_topo_column(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    with metrics.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_topo_loss", "val_topo_loss"])
        w.writerow([1, 0.20, 0.11])
        w.writerow([2, 0.10, 0.12])

    ep, loss = best_topo_epoch(metrics)
    assert ep == 1
    assert loss == pytest.approx(0.11)
