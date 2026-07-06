"""Shared DBSCAN-based cloud evaluation (paper protocol).

Per cloud: ellphi/mahalanobis DBSCAN for outlier labels (MCC), and topo W-Dist
(learned ellipses vs clean teacher PD — same as ``TopologicalLoss``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from tda_ml.dbscan import compute_anisotropic_distance_matrix_np
from tda_ml.metrics import compute_recall_specificity_gmean_mcc_wdist
from tda_ml.topo_wdist import TopoWdistOptions

DEFAULT_EPS_VALUES: list[float] = list(np.linspace(0.15, 1.5, 15))
DEFAULT_MIN_SAMPLES_VALUES: list[int] = [3, 5, 7, 10, 15]


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
    mask = np.abs(clean_pc).sum(axis=1) > 1e-6
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


def _eval_prepared_cloud(
    cloud: PreparedCloud,
    eps: float,
    min_samples: int,
    *,
    topo_options: TopoWdistOptions | None = None,
) -> CloudMetrics:
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(cloud.dist)
    pred = dbscan_labels_to_outlier_pred(labels)
    recall, specificity, gmean, mcc, wdist = compute_recall_specificity_gmean_mcc_wdist(
        cloud.labels_gt,
        pred,
        points=cloud.points,
        params=cloud.params,
        clean_pc=cloud.clean_pc,
        topo_options=topo_options,
    )
    return CloudMetrics(recall, specificity, gmean, mcc, wdist)


def _aggregate(rows: list[CloudMetrics]) -> tuple[float, float, float, float, float]:
    return (
        float(np.mean([r.recall for r in rows])),
        float(np.mean([r.specificity for r in rows])),
        float(np.mean([r.gmean for r in rows])),
        float(np.mean([r.mcc for r in rows])),
        float(np.mean([r.wdist for r in rows])),
    )


def grid_search_prepared(
    prepared: list[PreparedCloud],
    *,
    eps_values: list[float],
    min_samples_values: list[int],
    objective: str = "wdist",
    topo_options: TopoWdistOptions | None = None,
) -> GridResult:
    """Grid-search DBSCAN over cached clouds; pick the best by ``objective``.

    ``objective='wdist'`` minimizes mean topo W-Dist (independent of DBSCAN hparams).
    ``objective='mcc'`` maximizes mean MCC. Returns aggregated metrics at the selected
    hyperparameters.
    """
    if objective not in ("wdist", "mcc"):
        raise ValueError(f"objective must be 'wdist' or 'mcc'; got {objective!r}")
    if not prepared:
        raise ValueError("No clouds to evaluate")

    minimize = objective == "wdist"
    best_score = float("inf") if minimize else -1.0
    best: GridResult | None = None

    for eps in eps_values:
        for min_samples in min_samples_values:
            rows: list[CloudMetrics] = []
            ok = True
            for cloud in prepared:
                try:
                    rows.append(
                        _eval_prepared_cloud(
                            cloud,
                            float(eps),
                            int(min_samples),
                            topo_options=topo_options,
                        )
                    )
                except Exception:  # noqa: BLE001 — skip degenerate hyperparameters
                    ok = False
                    break
            if not ok or not rows:
                continue
            recall, specificity, gmean, mcc, wdist = _aggregate(rows)
            score = wdist if minimize else mcc
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
                )

    if best is None:
        raise RuntimeError("DBSCAN grid search failed for all hyperparameters")
    return best


def evaluate_model_grid(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    backend: str,
    eps_values: list[float] | None = None,
    min_samples_values: list[int] | None = None,
    objective: str = "wdist",
    metric: str = "max",
    topo_options: TopoWdistOptions | None = None,
) -> GridResult:
    """Forward ``loader`` through ``model`` and grid-search DBSCAN by ``objective``."""
    clouds = list(iter_cloud_predictions(model, loader, device))
    prepared = prepare_clouds(clouds, backend=backend, metric=metric)
    return grid_search_prepared(
        prepared,
        eps_values=eps_values or DEFAULT_EPS_VALUES,
        min_samples_values=min_samples_values or DEFAULT_MIN_SAMPLES_VALUES,
        objective=objective,
        topo_options=topo_options,
    )


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
