"""Place outliers along local-PCA first principal axes (tangent directions)."""

from __future__ import annotations

import torch

from tda_ml.local_pca import local_pca_ellipse_params
from tda_ml.numerical_eps import MIN_TANGENT_OUTLIER_SEPARATION


def sample_local_pca_tangent_outliers(
    inliers: torch.Tensor,
    num_outliers: int,
    *,
    k: int = 10,
    offset_min: float = 0.15,
    offset_max: float = 0.40,
    generator: torch.Generator | None = None,
    max_attempts: int = 200,
    box_min: float = -1.0,
    box_max: float = 1.0,
    min_separation: float = MIN_TANGENT_OUTLIER_SEPARATION,
    existing_points: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sample outliers by displacing random inliers along local PCA PC1.

    Local PCA is computed on ``inliers`` only (no isotropic jitter). Each outlier
    is ``p + s * u1`` where ``u1 = (cos θ, sin θ)`` is the major-axis direction
    at a randomly chosen inlier ``p``, and ``s`` has random sign with
    ``|s| ~ Uniform(offset_min, offset_max)``.

    Candidates that leave ``[box_min, box_max]^2`` or land within
    ``min_separation`` of the separation reference set or already-accepted
    outliers are retried: two nearly coincident centers make the ellphi
    tangency derivative w.r.t. the center undefined. The separation reference
    defaults to ``inliers``; pass ``existing_points`` (e.g. noisy inliers that
    enter the cloud) when PCA geometry and cloud geometry differ. Exhausting
    ``max_attempts`` hard-fails; candidates are never clipped into the box nor
    merged onto an existing point.
    """
    if num_outliers <= 0:
        return torch.empty(0, 2, dtype=inliers.dtype, device=inliers.device)
    if inliers.ndim != 2 or inliers.shape[-1] != 2:
        raise ValueError(f"inliers must be (N, 2); got {tuple(inliers.shape)}")
    n = int(inliers.shape[0])
    if n < 2:
        raise ValueError(f"Need at least 2 inliers for local PCA; got n={n}")
    if offset_min <= 0 or offset_max < offset_min:
        raise ValueError(
            f"Require 0 < offset_min <= offset_max; got {offset_min}, {offset_max}"
        )
    if min_separation < 0:
        raise ValueError(f"min_separation must be >= 0; got {min_separation}")
    sep_ref = inliers if existing_points is None else existing_points
    if sep_ref.ndim != 2 or sep_ref.shape[-1] != 2:
        raise ValueError(
            f"existing_points must be (N, 2); got {tuple(sep_ref.shape)}"
        )

    params = local_pca_ellipse_params(inliers, k=k, normalize_axes=True)  # (N, 3)
    theta = params[:, 2]
    direction = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    outliers = torch.empty(num_outliers, 2, dtype=inliers.dtype, device=inliers.device)
    for j in range(num_outliers):
        for _ in range(max_attempts):
            if generator is not None:
                idx = int(torch.randint(0, n, (1,), generator=generator).item())
                u = float(torch.rand(1, generator=generator).item())
                sign = 1.0 if float(torch.rand(1, generator=generator).item()) < 0.5 else -1.0
            else:
                idx = int(torch.randint(0, n, (1,)).item())
                u = float(torch.rand(1).item())
                sign = 1.0 if float(torch.rand(1).item()) < 0.5 else -1.0
            mag = offset_min + (offset_max - offset_min) * u
            cand = inliers[idx] + (sign * mag) * direction[idx]
            if not bool(torch.all((cand >= box_min) & (cand <= box_max)).item()):
                continue
            if min_separation > 0:
                d_ref = torch.linalg.norm(sep_ref - cand, dim=-1).min()
                if bool((d_ref < min_separation).item()):
                    continue
                if j > 0:
                    d_out = torch.linalg.norm(outliers[:j] - cand, dim=-1).min()
                    if bool((d_out < min_separation).item()):
                        continue
            outliers[j] = cand
            break
        else:
            raise RuntimeError(
                "Failed to sample an in-bounds, well-separated tangent outlier after "
                f"{max_attempts} attempts (outlier_index={j}, "
                f"box=[{box_min}, {box_max}], "
                f"offset=[{offset_min}, {offset_max}], "
                f"min_separation={min_separation})."
            )
    return outliers
