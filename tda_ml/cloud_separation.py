"""Enforce pairwise center separation so ellphi tangency derivatives stay defined."""

from __future__ import annotations

import torch

from tda_ml.numerical_eps import MIN_ELLPHI_CENTER_SEPARATION


def apply_noise_with_min_separation(
    base: torch.Tensor,
    noise_std: float,
    *,
    min_separation: float = MIN_ELLPHI_CENTER_SEPARATION,
    generator: torch.Generator | None = None,
    max_attempts: int = 200,
) -> torch.Tensor:
    """
    Apply isotropic Gaussian noise while keeping pairwise distances >= ``min_separation``.

    Points are accepted left-to-right. A candidate that lands within
    ``min_separation`` of any already-accepted point is rejected and re-noised.
    Exhausting ``max_attempts`` hard-fails (no silent merge / clip).

    This prevents near-coincident ellipse centers (common when MNIST clouds are
    padded by duplicating foreground pixels) from making the ellphi tangency
    derivative w.r.t. mu numerically undefined.
    """
    if base.ndim != 2 or base.shape[-1] != 2:
        raise ValueError(f"base must be (N, 2); got {tuple(base.shape)}")
    if noise_std < 0:
        raise ValueError(f"noise_std must be >= 0; got {noise_std}")
    if min_separation < 0:
        raise ValueError(f"min_separation must be >= 0; got {min_separation}")

    n = int(base.shape[0])
    out = torch.empty_like(base)
    for i in range(n):
        for _ in range(max_attempts):
            if noise_std > 0:
                if generator is not None:
                    noise = torch.randn(2, generator=generator, dtype=base.dtype) * noise_std
                else:
                    noise = torch.randn(2, dtype=base.dtype, device=base.device) * noise_std
            else:
                noise = torch.zeros(2, dtype=base.dtype, device=base.device)
            cand = base[i] + noise
            if min_separation > 0 and i > 0:
                dmin = torch.linalg.norm(out[:i] - cand, dim=-1).min()
                if bool((dmin < min_separation).item()):
                    continue
            out[i] = cand
            break
        else:
            raise RuntimeError(
                "Failed to place a well-separated inlier after "
                f"{max_attempts} noise draws (point_index={i}, "
                f"noise_std={noise_std}, min_separation={min_separation})."
            )
    return out


def sample_uniform_outliers_with_min_separation(
    num_outliers: int,
    existing: torch.Tensor,
    *,
    min_separation: float = MIN_ELLPHI_CENTER_SEPARATION,
    generator: torch.Generator | None = None,
    max_attempts: int = 200,
    box_min: float = -1.0,
    box_max: float = 1.0,
) -> torch.Tensor:
    """Sample uniform ``[-1,1]^2`` outliers that stay ``min_separation`` from ``existing``."""
    if num_outliers <= 0:
        return torch.empty(0, 2, dtype=existing.dtype, device=existing.device)
    if min_separation < 0:
        raise ValueError(f"min_separation must be >= 0; got {min_separation}")

    span = box_max - box_min
    outliers = torch.empty(num_outliers, 2, dtype=existing.dtype, device=existing.device)
    for j in range(num_outliers):
        for _ in range(max_attempts):
            if generator is not None:
                cand = torch.rand(2, generator=generator, dtype=existing.dtype) * span + box_min
            else:
                cand = torch.rand(2, dtype=existing.dtype, device=existing.device) * span + box_min
            if min_separation > 0:
                d_ex = torch.linalg.norm(existing - cand, dim=-1).min()
                if bool((d_ex < min_separation).item()):
                    continue
                if j > 0:
                    d_out = torch.linalg.norm(outliers[:j] - cand, dim=-1).min()
                    if bool((d_out < min_separation).item()):
                        continue
            outliers[j] = cand
            break
        else:
            raise RuntimeError(
                "Failed to sample a well-separated uniform outlier after "
                f"{max_attempts} attempts (outlier_index={j}, "
                f"min_separation={min_separation})."
            )
    return outliers
