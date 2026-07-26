from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, recall_score

from tda_ml.persistence import compute_w_distance
from tda_ml.topo_wdist import TopoWdistOptions, compute_topo_wdist


def compute_recall_specificity_gmean_mcc(
    labels_gt: Sequence[int] | np.ndarray,
    labels_pred: Sequence[int] | np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Return common binary metrics for 0=inlier and 1=outlier labels.

    Returns:
        recall, specificity, gmean, mcc
    """
    recall = recall_score(labels_gt, labels_pred, zero_division=0)
    tn, fp, _, _ = confusion_matrix(labels_gt, labels_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean = float((recall * specificity) ** 0.5)
    mcc = float(matthews_corrcoef(labels_gt, labels_pred))
    return float(recall), float(specificity), gmean, mcc


def compute_recall_specificity_gmean_mcc_wdist(
    labels_gt: Sequence[int] | np.ndarray,
    labels_pred: Sequence[int] | np.ndarray,
    *,
    points: np.ndarray | None = None,
    params: np.ndarray | None = None,
    clean_pc: np.ndarray | None = None,
    topo_options: TopoWdistOptions | None = None,
    gt_inliers: np.ndarray | None = None,
) -> tuple[float, float, float, float, float]:
    """
    Return the four classification metrics plus W-Dist.

    When ``params`` and ``clean_pc`` are provided, W-Dist is the ellipse-filtration
    distance (``compute_topo_wdist``): learned ellipses on the full cloud vs the
    clean teacher PD — same definition as ``TopologicalLoss``.

    When only ``gt_inliers`` is provided (legacy baselines), W-Dist uses Euclidean
    Alpha-complex PDs on DBSCAN-predicted inlier **point coordinates**.
    """
    recall, specificity, gmean, mcc = compute_recall_specificity_gmean_mcc(
        labels_gt, labels_pred
    )
    if points is not None and params is not None and clean_pc is not None:
        if topo_options is None:
            raise ValueError(
                "topo_options is required for ellipse-filtration topo W-Dist; "
                "refusing silent TopoWdistOptions defaults"
            )
        w_dist = float(compute_topo_wdist(points, params, clean_pc, topo_options))
    elif points is not None and gt_inliers is not None:
        pred_inliers = points[np.asarray(labels_pred) == 0]
        w_dist = float(compute_w_distance(pred_inliers, gt_inliers))
    else:
        raise ValueError(
            "W-Dist requires either (points, params, clean_pc) for ellipse-filtration "
            "topo W-Dist, or (points, gt_inliers) for legacy Euclidean Alpha W-Dist; "
            "refusing silent zero."
        )
    return recall, specificity, gmean, mcc, w_dist
