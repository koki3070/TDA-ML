"""val_topo phase-transition (\"cliff\") gate for thin-rings Methods runs.

On rings, successful learning shows a sharp drop of ``val_topo_loss`` from
~0.45 to ~0.1 (see docs/20260729_rings_h01_playbook.md follow-ups). Seeds that
never cross this threshold stay in a bad basin (W-Dist ~14, MCC~0) and must not
be treated as completed paper results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# Declared Methods threshold: success seeds finish ~0.09; plateau seeds stay ~0.45.
RINGS_VAL_TOPO_CLIFF_MAX = 0.3


class ValTopoCliffError(RuntimeError):
    """Raised when training finished without crossing the val_topo cliff."""


def cliff_max_from_config(config: dict[str, Any] | None) -> float | None:
    """Return cliff max if ``training.require_val_topo_cliff`` is enabled, else None."""
    training = (config or {}).get("training") or {}
    if "require_val_topo_cliff" not in training:
        return None
    if not bool(training["require_val_topo_cliff"]):
        return None
    if "val_topo_cliff_max" not in training:
        raise ValueError(
            "training.require_val_topo_cliff=true requires explicit "
            "training.val_topo_cliff_max (refusing silent default)"
        )
    cliff_max = float(training["val_topo_cliff_max"])
    if not (cliff_max > 0.0):
        raise ValueError(
            f"training.val_topo_cliff_max must be > 0; got {cliff_max}"
        )
    return cliff_max


def best_val_topo_from_history(metrics_history: list[dict[str, Any]]) -> float:
    if not metrics_history:
        raise ValueError("metrics_history is empty; cannot evaluate val_topo cliff")
    values = []
    for row in metrics_history:
        if "val_topo_loss" not in row or row["val_topo_loss"] is None:
            raise ValueError(
                "metrics_history row missing val_topo_loss; refusing silent skip"
            )
        values.append(float(row["val_topo_loss"]))
    return float(min(values))


def best_val_topo_from_metrics_csv(metrics_csv: Path) -> tuple[float, int]:
    """Return ``(best_val_topo_loss, epoch_at_best)`` from a run metrics.csv."""
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_csv}")
    with metrics_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"metrics.csv is empty: {metrics_csv}")
    best_ep: int | None = None
    best_val: float | None = None
    for row in rows:
        if "val_topo_loss" not in row or row["val_topo_loss"] == "":
            raise ValueError(
                f"metrics.csv row missing val_topo_loss: {metrics_csv}"
            )
        if "epoch" not in row or row["epoch"] == "":
            raise ValueError(f"metrics.csv row missing epoch: {metrics_csv}")
        val = float(row["val_topo_loss"])
        ep = int(float(row["epoch"]))
        if best_val is None or val < best_val:
            best_val = val
            best_ep = ep
    assert best_val is not None and best_ep is not None
    return best_val, best_ep


def cliff_passed(best_val_topo: float, *, cliff_max: float) -> bool:
    return float(best_val_topo) <= float(cliff_max)


def assert_val_topo_cliff(
    best_val_topo: float,
    *,
    cliff_max: float,
    context: str,
) -> None:
    if cliff_passed(best_val_topo, cliff_max=cliff_max):
        return
    raise ValTopoCliffError(
        f"val_topo cliff not reached ({context}): "
        f"best_val_topo_loss={best_val_topo:.6f} > cliff_max={cliff_max:.6f}. "
        "Treat this run as empty-result (bad basin); do not use for paper aggregate."
    )


def cliff_deadline_epoch_from_config(config: dict[str, Any] | None) -> int | None:
    """Optional early-abort epoch when cliff is required but not yet crossed."""
    training = (config or {}).get("training") or {}
    if cliff_max_from_config(config) is None:
        return None
    if "val_topo_cliff_deadline_epoch" not in training:
        return None
    deadline = int(training["val_topo_cliff_deadline_epoch"])
    if deadline < 1:
        raise ValueError(
            f"training.val_topo_cliff_deadline_epoch must be >= 1; got {deadline}"
        )
    return deadline


def maybe_raise_cliff_deadline(
    *,
    epoch: int,
    best_val_topo: float,
    cliff_max: float,
    deadline_epoch: int | None,
    context: str,
) -> None:
    """Raise if past the deadline and the cliff has not been crossed."""
    if deadline_epoch is None:
        return
    if int(epoch) < int(deadline_epoch):
        return
    if cliff_passed(best_val_topo, cliff_max=cliff_max):
        return
    raise ValTopoCliffError(
        f"val_topo cliff deadline missed ({context}): "
        f"epoch={epoch} >= deadline={deadline_epoch}, "
        f"best_val_topo_loss={best_val_topo:.6f} > cliff_max={cliff_max:.6f}. "
        "Abort early for model_seed restart (empty-result)."
    )


def evaluate_run_cliff(
    run_dir: Path,
    *,
    cliff_max: float,
) -> dict[str, Any]:
    """Inspect a finished run directory for cliff status."""
    metrics_csv = Path(run_dir) / "logs" / "metrics.csv"
    best_val, best_ep = best_val_topo_from_metrics_csv(metrics_csv)
    passed = cliff_passed(best_val, cliff_max=cliff_max)
    return {
        "run_dir": str(run_dir),
        "best_val_topo_loss": best_val,
        "best_val_topo_epoch": best_ep,
        "val_topo_cliff_max": float(cliff_max),
        "val_topo_cliff_passed": passed,
    }


def write_cliff_report(run_dir: Path, report: dict[str, Any]) -> Path:
    path = Path(run_dir) / "logs" / "VAL_TOPO_CLIFF.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path
