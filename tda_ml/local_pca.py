"""Local PCA ellipse parameters (paper §3.3.2 ideal ellipses, ADBSCAN baseline)."""

from __future__ import annotations

import math

import torch

from tda_ml.numerical_eps import EIGENVALUE_FLOOR, PCA_RIDGE_EPS

TEACHER_MODE_EUCLIDEAN = "euclidean"
TEACHER_MODE_LOCAL_PCA = "local_pca"


def normalize_teacher_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m in (TEACHER_MODE_EUCLIDEAN, "euclid"):
        return TEACHER_MODE_EUCLIDEAN
    if m in (TEACHER_MODE_LOCAL_PCA, "local-pca", "ideal", "d_ideal"):
        return TEACHER_MODE_LOCAL_PCA
    raise ValueError(
        f"teacher_mode must be 'euclidean' or 'local_pca', got {mode!r}"
    )


def local_pca_ellipse_params(
    points: torch.Tensor,
    *,
    k: int = 10,
    normalize_axes: bool = True,
    major_scale: float = 1.0,
) -> torch.Tensor:
    """
    Ideal ellipse parameters from local PCA only (no learned corrections).

    Args:
        points: ``(B, N, 2)`` or ``(N, 2)`` coordinates.
        k: number of Euclidean nearest neighbors (including self in the k-ball).
        normalize_axes: if True, divide by the major semi-axis so ``a=1`` (aspect ratio
            only). If False, use raw ``sqrt(eigenvalue)`` semi-axes ``(sqrt(l1), sqrt(l2))``.
        major_scale: positive multiplier applied after normalization/raw axes.
            Paper no_cls uses ``0.4`` so teacher major axes match the student /
            DBSCAN neighborhood scale (unit-normalized teachers are too large).

    Returns:
        ``(..., N, 3)`` with ``[a, b, theta]`` per point.
    """
    scale = float(major_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"major_scale must be a finite positive float, got {major_scale!r}")

    if points.ndim == 2:
        return local_pca_ellipse_params(
            points.unsqueeze(0),
            k=k,
            normalize_axes=normalize_axes,
            major_scale=scale,
        ).squeeze(0)

    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"points must be (B, N, 2) or (N, 2); got {tuple(points.shape)}")

    b, n, d = points.shape
    if n < 2:
        raise ValueError(f"Need at least 2 points for local PCA; got n={n}")
    k_eff = min(int(k), n)
    if k_eff < 2:
        raise ValueError(f"Effective k must be >= 2; got k={k}, n={n}")

    dist_sq = torch.cdist(points, points, p=2) ** 2
    _, idx = torch.topk(-dist_sq, k=k_eff, dim=-1)

    batch_idx = torch.arange(b, device=points.device).view(b, 1, 1).expand(b, n, k_eff)
    flat_x = points.view(b * n, d)
    flat_neighbors = flat_x[idx.reshape(b, -1) + (batch_idx.reshape(b, -1) * n), :]
    neighbors = flat_neighbors.view(b, n, k_eff, d)

    relative_coords = neighbors - points.unsqueeze(2)
    mean_neighbor = relative_coords.mean(dim=2, keepdim=True)
    centered = relative_coords - mean_neighbor
    cov = torch.matmul(centered.transpose(-1, -2), centered) / (k_eff - 1)

    cov32 = cov.float()
    eye2 = torch.eye(2, device=points.device, dtype=torch.float32)
    e, v = torch.linalg.eigh(cov32 + eye2 * PCA_RIDGE_EPS)

    v1 = v[:, :, :, 1]
    base_angle = torch.atan2(v1[:, :, 1], v1[:, :, 0])

    base_axes = torch.sqrt(torch.clamp(e, min=EIGENVALUE_FLOOR))
    base_axes = torch.flip(base_axes, dims=[-1])
    if normalize_axes:
        base_axes = base_axes / (
            base_axes.max(dim=-1, keepdim=True)[0] + EIGENVALUE_FLOOR
        )
    if scale != 1.0:
        base_axes = base_axes * scale

    return torch.cat([base_axes, base_angle.unsqueeze(-1)], dim=-1).to(dtype=points.dtype)
