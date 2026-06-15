from __future__ import annotations

import numpy as np
import torch


def ellipse_params_to_centers_cov_torch(
    points: torch.Tensor, params: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(N,2) points + (N,3|5) params -> centers (N,2), cov (N,2,2)."""
    if params.ndim != 2:
        raise ValueError(f"params must be rank-2, got shape {tuple(params.shape)}")
    if params.shape[1] == 3:
        compact = params
    elif params.shape[1] == 5:
        compact = params[:, 2:5]
    else:
        raise ValueError(
            "params must have shape (N, 3) [a,b,theta] or (N, 5) [dx,dy,a,b,theta]; "
            f"got shape {tuple(params.shape)}"
        )
    centers = points
    a, b, th = compact[:, 0], compact[:, 1], compact[:, 2]
    cos_t, sin_t = torch.cos(th), torch.sin(th)
    r00 = cos_t**2 * a**2 + sin_t**2 * b**2
    r01 = cos_t * sin_t * (a**2 - b**2)
    r11 = sin_t**2 * a**2 + cos_t**2 * b**2
    cov = torch.stack(
        [torch.stack([r00, r01], dim=-1), torch.stack([r01, r11], dim=-1)],
        dim=-2,
    )
    return centers, cov


def ellipse_params_to_centers_cov_numpy(
    points_np: np.ndarray, params_np: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy equivalent of ``ellipse_params_to_centers_cov_torch``."""
    points_np = np.asarray(points_np, dtype=np.float64)
    params_np = np.asarray(params_np, dtype=np.float64)
    if params_np.ndim != 2:
        raise ValueError(f"params must be rank-2, got shape {params_np.shape}")
    if params_np.shape[1] == 3:
        compact = params_np
    elif params_np.shape[1] == 5:
        compact = params_np[:, 2:5]
    else:
        raise ValueError(
            "params must have shape (N, 3) [a,b,theta] or (N, 5) [dx,dy,a,b,theta]; "
            f"got shape {params_np.shape}"
        )
    centers = points_np
    a, b, th = compact[:, 0], compact[:, 1], compact[:, 2]
    cos_t, sin_t = np.cos(th), np.sin(th)
    r00 = cos_t**2 * a**2 + sin_t**2 * b**2
    r01 = cos_t * sin_t * (a**2 - b**2)
    r11 = sin_t**2 * a**2 + cos_t**2 * b**2

    cov = np.zeros((points_np.shape[0], 2, 2), dtype=np.float64)
    cov[:, 0, 0] = r00
    cov[:, 0, 1] = r01
    cov[:, 1, 0] = r01
    cov[:, 1, 1] = r11
    return centers, cov
