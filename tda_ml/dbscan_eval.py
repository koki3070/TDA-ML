"""Shared DBSCAN-based cloud evaluation (paper protocol).

Per cloud: ellphi/mahalanobis DBSCAN for outlier labels (MCC), and topo W-Dist
(learned ellipses vs clean teacher PD — same as ``TopologicalLoss``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from tda_ml.dbscan import compute_anisotropic_distance_matrix_np
from tda_ml.metrics import compute_recall_specificity_gmean_mcc
from tda_ml.numerical_eps import ZERO_PAD_ABS_SUM
from tda_ml.reproducibility import record_fallback, resolve_dbscan_grid, write_grid_log
from tda_ml.topo_wdist import TopoWdistOptions, compute_topo_wdist


@dataclass
class CloudMetrics:
    recall: float
    specificity: float
    gmean: float
    mcc: float
    wdist: float


@dataclass
class GridResult:
    """Aggregated metrics at the best DBSCAN hyperparameters."""

    recall: float
    specificity: float
    gmean: float
    mcc: float
    wdist: float
    eps: float
    min_samples: int
    n_clouds: int
    objective: str
    grid_log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PreparedCloud:
    """Per-cloud data with a cached distance matrix (grid-reusable)."""

    points: np.ndarray
    params: np.ndarray
    labels_gt: np.ndarray
    clean_pc: np.ndarray
    dist: np.ndarray


def valid_clean_inliers(clean_pc: np.ndarray) -> np.ndarray:
    """Drop zero-padding rows from the clean (ground-truth inlier) point cloud."""
    mask = np.abs(clean_pc).sum(axis=1) > ZERO_PAD_ABS_SUM
    return clean_pc[mask]


def dbscan_labels_to_outlier_pred(labels: np.ndarray) -> np.ndarray:
    """DBSCAN noise (-1) -> outlier (1); clustered points -> inlier (0)."""
    return (labels == -1).astype(np.int64)


def iter_cloud_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(points, params, labels_gt, clean_pc)`` per cloud in ``loader``."""
    model.eval()
    with torch.no_grad():
        for data, labels, clean_pc in loader:
            data = data.to(device, non_blocking=True)
            _, params = model(data)
            data_np = data.cpu().numpy()
            params_np = params.cpu().numpy()
            labels_np = labels.cpu().numpy()
            clean_np = clean_pc.cpu().numpy()
            for b in range(data_np.shape[0]):
                yield data_np[b], params_np[b], labels_np[b], clean_np[b]


def prepare_clouds(
    clouds: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    backend: str,
    metric: str = "max",
) -> list[PreparedCloud]:
    """Precompute distance matrices once per cloud (reused across the grid)."""
    prepared: list[PreparedCloud] = []
    for points, params, labels_gt, clean_pc in clouds:
        dist = compute_anisotropic_distance_matrix_np(
            points, params, metric=metric, backend=backend
        )
        prepared.append(
            PreparedCloud(
                points=points,
                params=params,
                labels_gt=labels_gt,
                clean_pc=clean_pc,
                dist=dist,
            )
        )
    return prepared


def _topo_wdist_for_prepared_cloud(
    cloud: PreparedCloud,
    *,
    topo_options: TopoWdistOptions | None,
) -> float:
    """Topo W-Dist for one cloud (independent of DBSCAN hyperparameters)."""
    if topo_options is None:
        raise ValueError("topo_options is required for topo W-Dist evaluation")
    return float(
        compute_topo_wdist(
            cloud.points,
            cloud.params,
            cloud.clean_pc,
            topo_options,
        )
    )


def _dbscan_classification_metrics(
    cloud: PreparedCloud,
    eps: float,
    min_samples: int,
) -> tuple[float, float, float, float]:
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(cloud.dist)
    pred = dbscan_labels_to_outlier_pred(labels)
    return compute_recall_specificity_gmean_mcc(cloud.labels_gt, pred)


def _eval_prepared_cloud(
    cloud: PreparedCloud,
    eps: float,
    min_samples: int,
    *,
    topo_options: TopoWdistOptions | None = None,
    wdist: float | None = None,
) -> CloudMetrics:
    recall, specificity, gmean, mcc = _dbscan_classification_metrics(
        cloud, eps, min_samples
    )
    if wdist is None:
        wdist = _topo_wdist_for_prepared_cloud(cloud, topo_options=topo_options)
    return CloudMetrics(recall, specificity, gmean, mcc, wdist)


def _aggregate(rows: list[CloudMetrics]) -> tuple[float, float, float, float, float]:
    if not rows:
        raise ValueError("No clouds to aggregate")
    wdists = [float(r.wdist) for r in rows]
    if any(not np.isfinite(w) for w in wdists):
        raise ValueError(
            f"cannot aggregate non-finite W-Dist ({wdists!r}); refusing silent nanmean"
        )
    return (
        float(np.mean([r.recall for r in rows])),
        float(np.mean([r.specificity for r in rows])),
        float(np.mean([r.gmean for r in rows])),
        float(np.mean([r.mcc for r in rows])),
        float(np.mean(wdists)),
    )


def grid_search_prepared(
    prepared: list[PreparedCloud],
    *,
    eps_values: list[float],
    min_samples_values: list[int],
    objective: str = "wdist",
    topo_options: TopoWdistOptions | None = None,
    allow_skip_degenerate_grid_cells: bool = False,
    manifest_ref: dict[str, Any] | None = None,
) -> GridResult:
    """Grid-search DBSCAN over cached clouds; pick the best by ``objective``."""
    if objective not in ("wdist", "mcc"):
        raise ValueError(f"objective must be 'wdist' or 'mcc'; got {objective!r}")
    if not prepared:
        raise ValueError("No clouds to evaluate")

    minimize = objective == "wdist"
    best_score = float("inf") if minimize else -1.0
    best: GridResult | None = None
    grid_log: list[dict[str, Any]] = []
    cloud_wdists = [
        _topo_wdist_for_prepared_cloud(cloud, topo_options=topo_options)
        for cloud in prepared
    ]

    for eps in eps_values:
        for min_samples in min_samples_values:
            rows: list[CloudMetrics] = []
            cell_error: str | None = None
            for cloud, wdist in zip(prepared, cloud_wdists, strict=True):
                try:
                    rows.append(
                        _eval_prepared_cloud(
                            cloud,
                            float(eps),
                            int(min_samples),
                            topo_options=topo_options,
                            wdist=wdist,
                        )
                    )
                except Exception as exc:
                    cell_error = f"{type(exc).__name__}: {exc}"
                    break
            log_entry: dict[str, Any] = {
                "eps": float(eps),
                "min_samples": int(min_samples),
            }
            if cell_error is not None:
                log_entry["status"] = "failed"
                log_entry["error"] = cell_error
                grid_log.append(log_entry)
                if allow_skip_degenerate_grid_cells:
                    if manifest_ref is not None:
                        record_fallback(
                            manifest_ref,
                            "dbscan_grid_cell_skip",
                            f"eps={eps} min_samples={min_samples}: {cell_error}",
                        )
                    continue
                raise RuntimeError(
                    "DBSCAN grid cell failed "
                    f"(eps={eps}, min_samples={min_samples}): {cell_error}. "
                    "Set reproducibility.allow_skip_degenerate_grid_cells=true to opt in "
                    "to skipping failed cells."
                )
            recall, specificity, gmean, mcc, wdist = _aggregate(rows)
            score = wdist if minimize else mcc
            log_entry.update(
                {
                    "status": "ok",
                    "mean_mcc": mcc,
                    "mean_wdist": wdist,
                    "n_clouds": len(rows),
                }
            )
            grid_log.append(log_entry)
            is_better = score < best_score if minimize else score > best_score
            if is_better:
                best_score = score
                best = GridResult(
                    recall=recall,
                    specificity=specificity,
                    gmean=gmean,
                    mcc=mcc,
                    wdist=wdist,
                    eps=float(eps),
                    min_samples=int(min_samples),
                    n_clouds=len(rows),
                    objective=objective,
                    grid_log=grid_log.copy(),
                )

    if best is None:
        raise RuntimeError(
            "DBSCAN grid search failed for all hyperparameters; "
            f"see grid_log ({len(grid_log)} cells)"
        )
    best.grid_log = grid_log
    return best


def evaluate_model_grid(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    config: dict[str, Any],
    backend: str,
    eps_values: list[float] | None = None,
    min_samples_values: list[int] | None = None,
    objective: str = "wdist",
    metric: str = "max",
    topo_options: TopoWdistOptions | None = None,
    allow_skip_degenerate_grid_cells: bool = False,
    grid_log_path: Path | str | None = None,
    manifest_ref: dict[str, Any] | None = None,
) -> GridResult:
    """Forward ``loader`` through ``model`` and grid-search DBSCAN by ``objective``."""
    eps_resolved, ms_resolved = resolve_dbscan_grid(
        config,
        eps_values=eps_values,
        min_samples_values=min_samples_values,
    )
    clouds = list(iter_cloud_predictions(model, loader, device))
    prepared = prepare_clouds(clouds, backend=backend, metric=metric)
    result = grid_search_prepared(
        prepared,
        eps_values=eps_resolved,
        min_samples_values=ms_resolved,
        objective=objective,
        topo_options=topo_options,
        allow_skip_degenerate_grid_cells=allow_skip_degenerate_grid_cells,
        manifest_ref=manifest_ref,
    )
    if grid_log_path is not None:
        write_grid_log(grid_log_path, result.grid_log)
    return result


def evaluate_model_fixed(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    backend: str,
    eps: float,
    min_samples: int,
    metric: str = "max",
    topo_options: TopoWdistOptions | None = None,
) -> GridResult:
    """Evaluate at a single fixed ``(eps, min_samples)`` (no grid search)."""
    clouds = list(iter_cloud_predictions(model, loader, device))
    prepared = prepare_clouds(clouds, backend=backend, metric=metric)
    rows = [
        _eval_prepared_cloud(
            c,
            float(eps),
            int(min_samples),
            topo_options=topo_options,
        )
        for c in prepared
    ]
    recall, specificity, gmean, mcc, wdist = _aggregate(rows)
    return GridResult(
        recall=recall,
        specificity=specificity,
        gmean=gmean,
        mcc=mcc,
        wdist=wdist,
        eps=float(eps),
        min_samples=int(min_samples),
        n_clouds=len(rows),
        objective="fixed",
    )
