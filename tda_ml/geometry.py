from __future__ import annotations

import numpy as np
import torch


def ellipse_params_to_centers_cov_torch(
    points: torch.Tensor, params: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(N,2) points + (N,3) [a,b,theta] -> centers (N,2), cov (N,2,2)."""
    centers = points
    a, b, th = params[:, 0], params[:, 1], params[:, 2]
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
    centers = points_np
    a, b, th = params_np[:, 0], params_np[:, 1], params_np[:, 2]
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
