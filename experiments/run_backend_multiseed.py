"""
Multi-seed backend comparison driver (experimental protocol wrapper).

Protocol (defaults can be overridden via CLI):
- YAML base config loaded by name (default ``reproduce``; path resolved by
  ``tda_ml.config.load_config`` together with the rest of the training stack).
- Training runs sequentially via ``tda_ml.main.main`` with overrides for ``epochs``,
  ``data.seed``, ``model.topology_loss.distance_backend``, and output root
  ``outputs.base_dir``.

Outputs (under ``--out-base``, default ``outputs/backend_compare``):
- ``progress_summary.csv``: one row per completed (backend, seed, epochs).
- ``backend_stats.csv``: per-backend mean and variability of MCC across seeds.

Metric definitions (from each run's ``logs/metrics.csv``, one row per epoch):
- ``final_*``: values from the **last** epoch row (training order in the file).
- ``best_*``: values from the epoch row with **maximum** ``val_mcc``. If several
  epochs tie for the maximum, Python's ``max`` keeps the **first** such row in
  file order. ``best_val_loss`` and ``best_val_recall`` are taken from that same
  row (they are **not** the loss/recall at an argmin-loss epoch).

Aggregation in ``backend_stats.csv`` (over distinct seeds per backend):
- ``mean_*``: arithmetic mean across completed runs.
- ``std_*``: **sample** standard deviation with divisor ``n - 1`` (same as
  ``statistics.stdev``). For ``n_completed < 2``, standard deviation is written
  as ``0.0`` as a placeholder (cross-seed variability is undefined); interpret
  using ``n_completed``.

Concurrency / filesystem:
- A lock file under ``--out-base`` serializes this driver for one output tree.
  Do not run two instances sharing the same ``--out-base``.
- Run directory is taken from ``train_main`` return value (``run_dir``); no
  filesystem glob on ``config_id`` prefix.

Privacy / version control:
- ``progress_summary.csv`` stores absolute ``run_dir`` paths; avoid committing
  filled CSVs from personal machines if paths must stay private.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from tda_ml.main import main as train_main
from tda_ml.config import deep_update, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

PROGRESS_HEADER = [
    "backend",
    "seed",
    "epochs",
    "run_dir",
    "best_val_mcc",
    "best_val_recall",
    "best_val_loss",
    "final_val_mcc",
    "final_val_recall",
    "final_val_loss",
]
REQUIRED_METRIC_COLUMNS = {"val_mcc", "val_recall", "val_loss"}


def _parse_float(
    row: dict[str, str],
    key: str,
    metrics_path: str,
    *,
    require_finite: bool = True,
) -> float:
    if key not in row:
        raise RuntimeError(f"Missing column '{key}' in metrics file: {metrics_path}")
    try:
        value = float(row[key])
    except ValueError as e:
        raise RuntimeError(
            f"Failed to parse '{key}' as float in metrics file: {metrics_path}"
        ) from e
    if require_finite and not math.isfinite(value):
        raise RuntimeError(f"Non-finite value for '{key}' in metrics file: {metrics_path}")
    if not require_finite and not math.isfinite(value):
        print(
            f"[WARN] Non-finite value for '{key}' in metrics file: {metrics_path}; "
            "keeping value for summary output."
        )
    return value


def parse_metrics_csv(metrics_path: str) -> dict[str, float]:
    """Parse epoch-wise metrics; see module docstring for *best* vs *final*."""
    with open(metrics_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_METRIC_COLUMNS - fieldnames
        if missing:
            raise RuntimeError(
                f"Missing required columns {sorted(missing)} in metrics file: {metrics_path}"
            )
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows found in metrics file: {metrics_path}")

    final = rows[-1]
    # Tie-break: first epoch row attaining max val_mcc (stable with max()).
    best = max(rows, key=lambda r: _parse_float(r, "val_mcc", metrics_path))
    return {
        "final_val_mcc": _parse_float(final, "val_mcc", metrics_path),
        "final_val_recall": _parse_float(final, "val_recall", metrics_path),
        "final_val_loss": _parse_float(
            final, "val_loss", metrics_path, require_finite=False
        ),
        "best_val_mcc": _parse_float(best, "val_mcc", metrics_path),
        "best_val_recall": _parse_float(best, "val_recall", metrics_path),
        "best_val_loss": _parse_float(best, "val_loss", metrics_path, require_finite=False),
    }


def ensure_csv_header(path: str, header: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        return

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        existing = next(reader, None)

    if existing is None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        return

    if existing != header:
        raise RuntimeError(
            "Unexpected CSV header in progress file. "
            f"Expected={header}, Found={existing}. "
            "Use a fresh --out-base directory or fix the CSV manually."
        )


def append_row(path: str, row: list[Any]) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


@contextmanager
def out_base_lock(out_base: str):
    lock_path = os.path.join(out_base, ".run_backend_multiseed.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as e:
        raise RuntimeError(
            "Another run_backend_multiseed process appears to be using this --out-base. "
            "Use a different --out-base or wait for the running process to finish."
        ) from e

    try:
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        yield
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def read_progress(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    valid_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=2):
        try:
            int(row["seed"])
            int(row["epochs"])
            float(row["best_val_mcc"])
            float(row["final_val_mcc"])
        except (KeyError, ValueError):
            print(f"[WARN] skip malformed progress row at line={idx}: {row}")
            continue
        valid_rows.append(row)
    return valid_rows


def sample_std_over_seeds(values: list[float]) -> float:
    """Sample standard deviation (divisor n-1); 0.0 if n < 2 (placeholder)."""
    if len(values) < 2:
        return 0.0
    return stdev(values)


def write_backend_stats(progress_csv: str, stats_csv: str) -> None:
    """Rewrite backend_stats.csv from progress_summary (deduped); see module docstring."""
    rows = read_progress(progress_csv)
    # Deduplicate retries/reruns: keep the last observed row per run key.
    unique_by_run: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in rows:
        key = (row["backend"], int(row["seed"]), int(row["epochs"]))
        unique_by_run[key] = row
    rows = list(unique_by_run.values())

    by_backend: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_backend.setdefault(r["backend"], []).append(r)

    header = [
        "backend",
        "n_completed",
        "mean_best_val_mcc",
        "std_best_val_mcc",
        "mean_final_val_mcc",
        "std_final_val_mcc",
    ]
    with open(stats_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for backend, rs in sorted(by_backend.items()):
            n = len(rs)
            best_vals = [float(x["best_val_mcc"]) for x in rs]
            final_vals = [float(x["final_val_mcc"]) for x in rs]
            best_std = sample_std_over_seeds(best_vals)
            final_std = sample_std_over_seeds(final_vals)
            writer.writerow(
                [
                    backend,
                    n,
                    mean(best_vals),
                    best_std,
                    mean(final_vals),
                    final_std,
                ]
            )


def run_one(
    base_config_name: str,
    backend: str,
    seed: int,
    epochs: int,
    out_base: str,
    progress_csv: str,
    w_topo: float | None = None,
) -> None:
    cfg = load_config(base_config_name, project_root=REPO_ROOT)
    config_id = f"backend_{backend}_seed{seed}"
    topo_cfg = cfg.get("model", {}).get("topology_loss", {})
    if "ellphi_differentiable" not in topo_cfg:
        raise ValueError(
            "model.topology_loss.ellphi_differentiable must be set explicitly "
            "in the base config; refusing silent true default"
        )
    ellphi_diff = bool(topo_cfg["ellphi_differentiable"])

    overrides: dict[str, Any] = {
        "meta": {"config_id": config_id},
        "training": {"epochs": epochs},
        "data": {"seed": seed},
        "model": {
            "topology_loss": {
                "distance_backend": backend,
                "ellphi_differentiable": ellphi_diff,
                # ellphi is geometry-only and rejects unsupported probability
                # weighting instead of silently ignoring it.
                "prob_weighting": backend != "ellphi",
            }
        },
        "outputs": {"base_dir": out_base},
    }
    if w_topo is not None:
        overrides["loss"] = {"w_topo": float(w_topo)}
    cfg = deep_update(cfg, overrides)

    print(f"[START] backend={backend} seed={seed} epochs={epochs}")
    result = train_main(config=cfg)
    run_dir = result.get("run_dir")
    if not run_dir:
        raise RuntimeError(
            f"train_main did not return run_dir for backend={backend} seed={seed}"
        )
    metrics_path = os.path.join(run_dir, "logs", "metrics.csv")
    metrics = parse_metrics_csv(metrics_path)

    append_row(
        progress_csv,
        [
            backend,
            seed,
            epochs,
            run_dir,
            metrics["best_val_mcc"],
            metrics["best_val_recall"],
            metrics["best_val_loss"],
            metrics["final_val_mcc"],
            metrics["final_val_recall"],
            metrics["final_val_loss"],
        ],
    )
    print(
        "[DONE] "
        f"backend={backend} seed={seed} best_val_mcc={metrics['best_val_mcc']:.4f} "
        f"final_val_mcc={metrics['final_val_mcc']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run multi-seed backend comparison and aggregate metrics. "
            "See module docstring for metric definitions and aggregation."
        )
    )
    p.add_argument("--base-config", type=str, default="reproduce")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    p.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=["mahalanobis", "ellphi"],
        choices=["mahalanobis", "ellphi"],
    )
    p.add_argument(
        "--out-base",
        type=str,
        default="outputs/backend_compare",
    )
    p.add_argument(
        "--rerun-completed",
        action="store_true",
        help=(
            "Force rerun even when (backend, seed, epochs) already exists in "
            "progress_summary.csv."
        ),
    )
    p.add_argument(
        "--w-topo",
        type=float,
        default=None,
        help="Override loss.w_topo (e.g. 0 for topology ablation).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_base, exist_ok=True)

    progress_csv = os.path.join(args.out_base, "progress_summary.csv")
    stats_csv = os.path.join(args.out_base, "backend_stats.csv")
    ensure_csv_header(
        progress_csv,
        PROGRESS_HEADER,
    )

    done = {
        (r["backend"], int(r["seed"]), int(r["epochs"]))
        for r in read_progress(progress_csv)
    }

    with out_base_lock(args.out_base):
        for backend in args.backends:
            for seed in args.seeds:
                key = (backend, seed, args.epochs)
                if key in done and not args.rerun_completed:
                    print(
                        "[SKIP] already completed "
                        f"backend={backend} seed={seed} epochs={args.epochs}"
                    )
                    continue
                if key in done and args.rerun_completed:
                    print(
                        "[RERUN] existing result found but rerunning due to "
                        f"--rerun-completed backend={backend} seed={seed} epochs={args.epochs}"
                    )
                run_one(
                    base_config_name=args.base_config,
                    backend=backend,
                    seed=seed,
                    epochs=args.epochs,
                    out_base=args.out_base,
                    progress_csv=progress_csv,
                    w_topo=args.w_topo,
                )
                write_backend_stats(progress_csv, stats_csv)

    write_backend_stats(progress_csv, stats_csv)
    print(f"Progress CSV: {progress_csv}")
    print(f"Backend stats: {stats_csv}")


if __name__ == "__main__":
    main()

