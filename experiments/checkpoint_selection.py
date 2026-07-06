"""Select training checkpoints by topo loss column in metrics.csv (saved epochs only)."""

from __future__ import annotations

import csv
from pathlib import Path

DEFAULT_TOPO_COLUMN = "train_topo_loss"


def topo_loss_column(metrics_path: Path) -> str:
    """Prefer ``val_topo_loss`` when logged; else ``train_topo_loss``."""
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or ()
    if "val_topo_loss" in fields:
        return "val_topo_loss"
    return DEFAULT_TOPO_COLUMN


def list_checkpoint_epochs(run_dir: Path) -> set[int]:
    """Epoch numbers with ``checkpoint_epoch_{k}.pth`` on disk."""
    prefix = "checkpoint_epoch_"
    epochs: set[int] = set()
    for path in run_dir.glob(f"{prefix}*.pth"):
        suffix = path.stem.removeprefix(prefix)
        if suffix.isdigit():
            epochs.add(int(suffix))
    return epochs


def best_topo_epoch(
    metrics_path: Path,
    *,
    column: str | None = None,
    saved_epochs: set[int] | None = None,
) -> tuple[int, float]:
    """Return epoch with minimum topo loss, optionally restricted to saved ckpts."""
    col = column or topo_loss_column(metrics_path)
    with metrics_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty metrics: {metrics_path}")
    if col not in (rows[0].keys() if rows else ()):
        raise RuntimeError(f"column {col!r} missing from {metrics_path}")
    pairs = [(int(r["epoch"]), float(r[col])) for r in rows if r.get(col, "")]
    if saved_epochs is not None:
        pairs = [p for p in pairs if p[0] in saved_epochs]
        if not pairs:
            raise RuntimeError(
                f"no metrics rows for saved checkpoints under {metrics_path.parent.parent} "
                f"(saved epochs: {sorted(saved_epochs)})"
            )
    return min(pairs, key=lambda t: t[1])


def best_topo_checkpoint(
    run_dir: Path,
    *,
    column: str | None = None,
) -> tuple[str, int, float, int | None, float | None]:
    """Pick ``checkpoint_epoch_{k}.pth`` with minimum topo loss among saved ckpts.

    Returns:
        (checkpoint_name, epoch, topo_loss, global_best_epoch, global_train_topo_loss)
        Global fields are set when the unrestricted minimum lies on an unsaved epoch.
    """
    run_dir = run_dir.resolve()
    metrics_path = run_dir / "logs" / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing {metrics_path}")

    col = column or topo_loss_column(metrics_path)
    saved_epochs = list_checkpoint_epochs(run_dir)
    if not saved_epochs:
        raise FileNotFoundError(f"no checkpoint_epoch_*.pth under {run_dir}")

    epoch, topo_loss = best_topo_epoch(metrics_path, column=col, saved_epochs=saved_epochs)
    global_epoch, global_topo = best_topo_epoch(metrics_path, column=col)
    global_epoch_out = global_epoch if global_epoch != epoch else None
    global_topo_out = global_topo if global_epoch != epoch else None
    return (
        f"checkpoint_epoch_{epoch}.pth",
        epoch,
        topo_loss,
        global_epoch_out,
        global_topo_out,
    )
