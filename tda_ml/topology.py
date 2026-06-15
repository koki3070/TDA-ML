from __future__ import annotations

import torch


def compute_anisotropic_metric(
    axes: torch.Tensor,
    angles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the components of the local metric tensor M_i based on ellipse parameters.

    Mathematical Definition:
    M_i = R(theta_i) diag(a_i^-2, b_i^-2) R(theta_i)^T

    Args:
        axes (torch.Tensor): (..., 2) [a, b]
        angles (torch.Tensor): (..., 1) [theta]

    Returns:
        tuple: (m00, m11, m01) components of the metric tensors.
    """
    a = axes[..., 0] + 1e-6
    b = axes[..., 1] + 1e-6
    theta = angles.squeeze(-1)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    inv_a2 = 1.0 / (a**2 + 1e-8)
    inv_b2 = 1.0 / (b**2 + 1e-8)

    # Components of M = R diag(a^-2, b^-2) R^T
    m00 = cos_t**2 * inv_a2 + sin_t**2 * inv_b2
    m11 = sin_t**2 * inv_a2 + cos_t**2 * inv_b2
    m01 = cos_t * sin_t * (inv_a2 - inv_b2)

    return m00, m11, m01


def compute_anisotropic_distance_matrix(
    points: torch.Tensor,
    params: torch.Tensor,
    probs: torch.Tensor | None = None,
    symmetrize: str = "max",
) -> torch.Tensor:
    r"""
    Computes the anisotropic distance matrix between points in a vectorized manner.

    Mathematical Formulation:
    1. Distance from $x_i$ to $x_j$ under metric $M_i$:
       $d_{M_i}(x_i, x_j) = \sqrt{(x_j - x_i)^\top M_i (x_j - x_i)}$
    2. Symmetrized Distance $D_{ij}$:
       $D_{ij} = \max \{ d_{M_i}(x_i, x_j), d_{M_j}(x_j, x_i) \}$
    3. Probability weighting (when ``probs`` is set):
       Let $p_{\mathrm{in},i} = 1 - p_{\mathrm{out},i}$ with $p_{\mathrm{out},i}$ the predicted
       outlier probability at $i$. The implementation scales squared distances by
       $1 / (p_{\mathrm{in},i}\, p_{\mathrm{in},j})$ (elementwise product on the batch grid).
       This is the only probability-weighting mode implemented; there is no separate
       asymmetric $1/p_{\mathrm{in},i}$-only variant.

    Args:
        points (torch.Tensor): (B, N, 2) Coordinates of centers.
        params (torch.Tensor): (B, N, 3) or (B, N, 5).
            - (B, N, 3): compact ellipse parameters [a, b, theta]
            - (B, N, 5): full model head output [dx, dy, a, b, theta]
              (center offsets are currently ignored in distance computation)
        probs (torch.Tensor, optional): (B, N) Outlier probabilities in $[0,1]$.
            When provided, squared distances are divided by
            ``(1-probs).unsqueeze(2) * (1-probs).unsqueeze(1)`` (clamped).
        symmetrize (str): 'max' or 'min' for metric symmetrization along $i,j$. Default 'max'.

    Returns:
        torch.Tensor: (B, N, N) Symmetric distance matrices.
    """
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(
            f"points must have shape (B, N, 2), got shape {tuple(points.shape)}"
        )
    batch_size, num_points, _ = points.shape
    if params.ndim != 3 or params.shape[0] != batch_size or params.shape[1] != num_points:
        raise ValueError(
            "params must have shape (B, N, K) matching points; "
            f"got points {tuple(points.shape)}, params {tuple(params.shape)}"
        )
    if params.shape[-1] not in (3, 5):
        raise ValueError(
            "params must have shape (B, N, 3) or (B, N, 5); "
            f"got params {tuple(params.shape)}"
        )
    if probs is not None and probs.shape != (batch_size, num_points):
        raise ValueError(
            "probs must have shape (B, N) matching points; "
            f"got points {tuple(points.shape)}, probs {tuple(probs.shape)}"
        )

    centers = points
    metric_params = params if params.shape[-1] == 3 else params[..., 2:5]
    m00, m11, m01 = compute_anisotropic_metric(
        metric_params[..., 0:2], metric_params[..., 2:3]
    )

    # Compute relative coordinates (B, N, N, 2)
    # diff[b, i, j] = centers[b, j] - centers[b, i]
    diff = centers.unsqueeze(2) - centers.unsqueeze(1)
    dx = diff[..., 0]
    dy = diff[..., 1]

    # Compute d_Mi(xi, xj)^2 for all i, j
    # m00 shape is (B, N), need to broadcast to (B, N, N)
    dist_sq_ij = (
        m00.unsqueeze(2) * dx**2
        + 2 * m01.unsqueeze(2) * dx * dy
        + m11.unsqueeze(2) * dy**2
    )

    # Weight by inlier probabilities if provided: divide squared distance by p_in[i]*p_in[j].
    if probs is not None:
        inlier_probs = 1.0 - probs
        inlier_probs = torch.clamp(inlier_probs, min=1e-4)
        prob_matrix = inlier_probs.unsqueeze(2) * inlier_probs.unsqueeze(1)  # (B, N, N)
        dist_sq_ij = dist_sq_ij / (prob_matrix + 1e-8)

    # Symmetrization
    dist_sq_ji = dist_sq_ij.transpose(-1, -2)

    if symmetrize == "max":
        dist_sq = torch.max(dist_sq_ij, dist_sq_ji)
    elif symmetrize == "min":
        dist_sq = torch.min(dist_sq_ij, dist_sq_ji)
    else:
        raise ValueError(
            f"symmetrize must be 'max' or 'min', got {symmetrize!r}"
        )

    dist_sq = torch.clamp(dist_sq, min=0.0)
    return torch.sqrt(dist_sq)
