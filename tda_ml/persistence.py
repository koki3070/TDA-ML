"""Persistence diagrams and Wasserstein distances (Euclidean Alpha / Gudhi).

Used for the optional Euclidean baseline W-Dist path in ``metrics``.
The paper main-table topo W-Dist uses ellipse filtration via
``tda_ml.topo_wdist`` / ``TopologicalLoss``, not this module.
"""

import numpy as np
import torch
import gudhi
from gudhi.wasserstein import wasserstein_distance


def wasserstein_h1(
    pd_a,
    pd_b,
    order: float = 1.0,
    internal_p: float = 2.0,
) -> float:
    """
    Wasserstein distance between H1 persistence diagrams (finite points only).
    ``keep_essential_parts=False`` avoids ``+inf`` from mismatched essential counts.
    """
    x = np.asarray(pd_a, dtype=np.float64) if len(pd_a) else np.empty((0, 2))
    y = np.asarray(pd_b, dtype=np.float64) if len(pd_b) else np.empty((0, 2))
    return float(
        wasserstein_distance(
            x, y, order=order, internal_p=internal_p, keep_essential_parts=False
        )
    )


def compute_w_distance(points_pred, points_gt):
    """
    Wasserstein distance between H1 persistence diagrams of two point clouds.

    Args:
        points_pred: (N, 2) array or tensor
        points_gt: (M, 2) array or tensor

    Returns:
        1-Wasserstein distance (``order=1``, ``internal_p=2``) via Gudhi.
        Empty point clouds yield empty H1 diagrams; the distance is the
        standard optimal-transport cost (including matching to the diagonal),
        not a sentinel penalty.
    """
    if isinstance(points_pred, torch.Tensor):
        points_pred = points_pred.detach().cpu().numpy()
    if isinstance(points_gt, torch.Tensor):
        points_gt = points_gt.detach().cpu().numpy()

    def get_pd(pts):
        if len(pts) < 3:
            return []
        alpha_complex = gudhi.AlphaComplex(points=pts)
        simplex_tree = alpha_complex.create_simplex_tree()
        simplex_tree.compute_persistence()
        pd = simplex_tree.persistence_intervals_in_dimension(1)
        return pd

    pd_pred = get_pd(points_pred)
    pd_gt = get_pd(points_gt)

    return wasserstein_h1(pd_pred, pd_gt, order=1, internal_p=2)
