"""
Distance-matrix backends: differentiable anisotropic Mahalanobis and ellphi tangency distance.

The differentiable ellphi path connects ``ellphi.grad`` functions
(``coef_from_cov_grad`` / ``pdist_tangency_grad``) through a PyTorch
``autograd.Function`` wrapper (upstream: https://github.com/t-uda/ellphi).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from tda_ml.ellphi_torch import (
    _has_ellphi_grad_api,
    ellipse_params_to_centers_cov,
    pdist_tangency_matrix_differentiable,
)
from tda_ml.geometry import ellipse_params_to_centers_cov_numpy
from tda_ml.numerical_eps import NUMERICAL_EPS
from tda_ml.topology import compute_anisotropic_distance_matrix

try:
    import ellphi
except ImportError:
    ellphi = None  # type: ignore[misc, assignment]

try:
    from scipy.spatial.distance import squareform
except ImportError:
    squareform = None  # type: ignore[misc, assignment]

_ELLPHI_PROB_WARNED = False

DISTANCE_MODE_MAHALANOBIS = "mahalanobis"
DISTANCE_MODE_ELLPHI = "ellphi"


def normalize_topo_distance_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m == "mahalanobis":
        return DISTANCE_MODE_MAHALANOBIS
    if m == "ellphi":
        return DISTANCE_MODE_ELLPHI
    raise ValueError(f"Unknown topo distance mode: {mode!r}")


def rescale_distance_matrix(
    d_mat: torch.Tensor,
    *,
    scale_mode: str,
    eps_scale: float,
    clean_scale: float | None = None,
) -> torch.Tensor:
    """Align a predicted distance matrix to the teacher PD filtration units.

    ``clean_scale`` (m_e) is the median pairwise Euclidean distance of the teacher
    cloud for this sample. In ``median`` mode the prediction is scaled so its median
    matches m_e. ``clean_scale`` is required (missing scale hard-fails). Any other
    ``scale_mode`` applies the fixed scalar ``eps_scale`` (1.0 == no-op).
    """
    if scale_mode == "median":
        if clean_scale is None:
            raise RuntimeError(
                "scale_mode='median' requires clean_scale (teacher median Euclidean); "
                "refusing silent unit-median fallback."
            )
        off = d_mat[d_mat > 0]
        if off.numel() == 0:
            raise RuntimeError(
                "scale_mode='median' found no positive off-diagonal distances."
            )
        denom = torch.median(off).detach() + NUMERICAL_EPS
        return d_mat * (float(clean_scale) / denom)
    if eps_scale != 1.0:
        return d_mat * eps_scale
    return d_mat


def subsample_indices(
    n: int,
    max_points: int | None,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    """Random subsampling indices (legacy ``topo_loss_max_points``); ``None`` if no-op."""
    if max_points is None or n <= max_points:
        return None
    return torch.randperm(n, device=device)[:max_points]


def compute_ellphi_distance_matrix_np(points_np: np.ndarray, params_np: np.ndarray) -> np.ndarray:
    """
    Compute ellphi tangency distance matrix ``(N, N)`` in ``float64``.

    ``params`` has shape ``(N, 3) = [a, b, theta]`` at the points in ``points_np``.
    """
    if ellphi is None:
        raise ImportError("Distance backend 'ellphi' requires the ellphi package.")
    if squareform is None:
        raise ImportError("scipy is required to expand condensed pairwise distances.")

    centers, covs = ellipse_params_to_centers_cov_numpy(points_np, params_np)

    ec = ellphi.EllipseCloud.from_cov(
        np.ascontiguousarray(centers),
        np.ascontiguousarray(covs),
    )
    dist = ec.pdist_tangency()
    if dist.ndim == 1:
        dist = squareform(dist)
    return np.asarray(dist, dtype=np.float64)


def compute_distance_matrix_batch(
    points: torch.Tensor,
    params: torch.Tensor,
    *,
    probs: torch.Tensor | None,
    symmetrize: str,
    backend: str,
    ellphi_differentiable: bool = True,
) -> torch.Tensor:
    """
    Compute batched distance matrices with shape ``(B, N, N)``.

    backend:
      - ``mahalanobis``: ``compute_anisotropic_distance_matrix`` (differentiable)
      - ``ellphi``: tangency distance. If ``ellphi_differentiable=True`` and
        ``ellphi.grad`` is available, gradients flow to centers/covariances.
        Missing grad API is a hard-fail (no NumPy fallback).

    For ``ellphi``, ``probs``-based weighting is currently unsupported and ignored.
    """
    b = backend.lower().strip()
    if b not in ("mahalanobis", "ellphi"):
        raise ValueError(f"Unknown distance backend: {backend!r}. Use 'mahalanobis' or 'ellphi'.")

    if b == "mahalanobis":
        return compute_anisotropic_distance_matrix(
            points, params, probs=probs, symmetrize=symmetrize
        )

    global _ELLPHI_PROB_WARNED
    if probs is not None and not _ELLPHI_PROB_WARNED:
        warnings.warn(
            "distance_backend='ellphi' ignores outlier-probability weighting because it is not implemented yet.",
            UserWarning,
            stacklevel=2,
        )
        _ELLPHI_PROB_WARNED = True

    use_torch = ellphi_differentiable and _has_ellphi_grad_api()
    if ellphi_differentiable and not use_torch:
        raise RuntimeError(
            "distance_backend='ellphi' with ellphi_differentiable=True requires "
            "ellphi.grad (coef_from_cov_grad / pdist_tangency_grad). "
            "Install/update ellphi or set ellphi_differentiable=false explicitly."
        )

    batch_size = points.shape[0]
    mats: list[torch.Tensor] = []
    points_np = None
    params_np = None
    if not use_torch:
        # Convert once per batch to reduce repeated CPU/NumPy transfer overhead.
        points_np = points.detach().cpu().numpy()
        params_np = params.detach().cpu().numpy()

    for i in range(batch_size):
        if use_torch:
            c, cov = ellipse_params_to_centers_cov(points[i], params[i])
            mats.append(pdist_tangency_matrix_differentiable(c, cov))
        else:
            dm = compute_ellphi_distance_matrix_np(
                points_np[i],
                params_np[i],
            )
            mats.append(torch.from_numpy(dm).to(device=points.device, dtype=points.dtype))
    return torch.stack(mats, dim=0)


def compute_topo_distance_matrix(
    points: torch.Tensor,
    params: torch.Tensor,
    *,
    distance_mode: str = "mahalanobis",
    ellphi_backend: str = "auto",
) -> torch.Tensor:
    """
    Shared topology-loss entrypoint: map batched points/ellipse params to ``(B,N,N)`` distances.

    ``distance_mode`` is ``mahalanobis`` or ``ellphi``.
    Differentiable ellphi requires ``ellphi.grad``; otherwise hard-fail.
    """
    backend = normalize_topo_distance_mode(distance_mode)
    eb = str(ellphi_backend).strip().lower()
    ellphi_diff = eb in ("auto", "torch", "grad", "differentiable", "1", "true", "yes")
    return compute_distance_matrix_batch(
        points,
        params,
        probs=None,
        symmetrize="max",
        backend=backend,
        ellphi_differentiable=ellphi_diff,
    )


def mahalanobis_distance_matrix_batched(points: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Mahalanobis-style batched distance matrix without probability weighting."""
    return compute_anisotropic_distance_matrix(
        points, params, probs=None, symmetrize="max"
    )
