import torch
from sklearn.cluster import DBSCAN

from tda_ml.distance_backend import compute_ellphi_distance_matrix_np
from tda_ml.topology import compute_anisotropic_distance_matrix


def compute_anisotropic_distance_matrix_np(
    points, params, metric="max", probs=None, backend="mahalanobis"
):
    """
    Compute the anisotropic distance matrix between points as a NumPy array.
    Uses the centralized logic from tda_ml.topology for mathematical consistency.

    Args:
        points (torch.Tensor or np.ndarray): (N, 2) coordinates.
        params (torch.Tensor or np.ndarray): (N, 3) ellipse shape [a, b, theta] at each point.
        metric (str): Symmetrization strategy. 'max' or 'min'. Default 'max'.
        probs (torch.Tensor or np.ndarray, optional): (N,) Outlier probabilities (same
            convention as :func:`tda_ml.topology.compute_anisotropic_distance_matrix`).
            With inlier probability ``p_in,i = clamp(1 - prob_i, min=1e-4)``, squared
            distances are divided by ``p_in,i * p_in,j`` before symmetrization and
            square root, i.e. distances scale like ``1 / sqrt(p_in,i * p_in,j)``.
        backend (str): ``mahalanobis`` or ``ellphi`` (probs are ignored for ellphi)

    Returns:
        np.ndarray: (N, N) distance matrix.
    """
    b = backend.lower().strip()
    points_t = torch.as_tensor(points, dtype=torch.float32)
    params_t = torch.as_tensor(params, dtype=torch.float32)

    if b == "ellphi" and points_t.ndim != 2:
        raise ValueError("backend='ellphi' requires points with shape (N, 2).")

    if points_t.ndim == 2:
        pts_np = points_t.numpy()
        par_np = params_t.numpy()
        if b == "ellphi":
            return compute_ellphi_distance_matrix_np(pts_np, par_np)
        probs_t = torch.as_tensor(probs, dtype=torch.float32) if probs is not None else None
        points_t = points_t.unsqueeze(0)
        params_t = params_t.unsqueeze(0)
        if probs_t is not None:
            probs_t = probs_t.unsqueeze(0)
        with torch.no_grad():
            dist_matrix_t = compute_anisotropic_distance_matrix(
                points_t, params_t, probs=probs_t, symmetrize=metric
            )
        return dist_matrix_t.squeeze(0).cpu().numpy()

    probs_t = torch.as_tensor(probs, dtype=torch.float32) if probs is not None else None

    with torch.no_grad():
        dist_matrix_t = compute_anisotropic_distance_matrix(
            points_t, params_t, probs=probs_t, symmetrize=metric
        )

    return dist_matrix_t.squeeze(0).cpu().numpy()


def apply_anisotropic_dbscan(
    points, params, eps=0.5, min_samples=5, metric="max", probs=None, backend="mahalanobis"
):
    """
    Apply DBSCAN using anisotropic distance.
    
    Args:
        points: (N, 2) coordinates
        params: (N, 3) ellipse parameters [a, b, theta]
        eps: DBSCAN epsilon
        min_samples: DBSCAN min_samples
        metric: Symmetrization strategy ('max' or 'min')
        probs: (N,) outlier probabilities; when ``backend='mahalanobis'``, distances use
            the same inlier-pair weighting as :func:`tda_ml.topology.compute_anisotropic_distance_matrix`
            (see ``compute_anisotropic_distance_matrix_np``). Ignored for ``ellphi``.
        backend: ``mahalanobis`` or ``ellphi``

    Returns:
        labels: Cluster labels for each point (-1 for noise)
    """
    dist_matrix = compute_anisotropic_distance_matrix_np(
        points, params, metric=metric, probs=probs, backend=backend
    )
    
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(dist_matrix)
    
    return labels
