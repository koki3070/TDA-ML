"""
PyTorch bridge for ellphi tangency distance matrices.

This module wires ``ellphi.grad`` VJP callbacks into ``autograd.Function`` so
ellphi tangency distances can participate in gradient-based training.
Reference: https://github.com/t-uda/ellphi
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from tda_ml.geometry import ellipse_params_to_centers_cov_torch


@lru_cache(maxsize=1)
def _has_ellphi_grad_api() -> bool:
    try:
        from ellphi.grad import coef_from_cov_grad  # noqa: F401
        from ellphi.grad import pdist_tangency_grad  # noqa: F401
    except ImportError:
        return False
    return True


def ellipse_params_to_centers_cov(
    points: torch.Tensor, params: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map ``(N,2)`` points and ``(N,3)`` [a,b,theta] to differentiable centers/covariances."""
    return ellipse_params_to_centers_cov_torch(points, params)


def _condensed_gradient_from_full(g: np.ndarray) -> np.ndarray:
    """Pull square-form upstream gradients back to scipy condensed layout.

    squareform duplicates each d_ij into D_ij and D_ji; sum gradients from both.
    """
    n = g.shape[0]
    iu = np.triu_indices(n, k=1)
    return g[iu] + g[(iu[1], iu[0])]


class _EllphiPdistMatrix(torch.autograd.Function):
    """Autograd wrapper for tangency distance matrix: centers/cov -> square ``(N,N)`` matrix."""

    @staticmethod
    def forward(ctx, centers: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        from ellphi.grad import coef_from_cov_grad, pdist_tangency_grad
        from scipy.spatial.distance import squareform

        ctx.save_for_backward(centers, cov)
        device, dtype = centers.device, centers.dtype
        x_np = centers.detach().cpu().numpy().astype(np.float64)
        c_np = cov.detach().cpu().numpy().astype(np.float64)

        coefs, vjp_cov = coef_from_cov_grad(x_np, c_np)
        if np.isnan(coefs).any():
            raise RuntimeError(
                "ellphi coef_from_cov_grad returned NaN; degenerate ellipse geometry"
            )

        try:
            dists, vjp_pdist = pdist_tangency_grad(coefs)
        except (ZeroDivisionError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                "ellphi pdist_tangency_grad failed; degenerate ellipse geometry"
            ) from exc

        full = squareform(dists)
        ctx.vjp_pdist = vjp_pdist
        ctx.vjp_cov = vjp_cov
        ctx.device = device
        ctx.dtype = dtype
        return torch.as_tensor(full, device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        centers, cov = ctx.saved_tensors
        g = grad_output.detach().cpu().numpy().astype(np.float64)
        g_cond = _condensed_gradient_from_full(g)
        try:
            grad_coefs = ctx.vjp_pdist(g_cond)
            grad_x, grad_cov_np = ctx.vjp_cov(grad_coefs)
        except (ZeroDivisionError, ValueError, RuntimeError) as exc:
            raise RuntimeError("ellphi VJP backward failed") from exc

        dev, dt = ctx.device, ctx.dtype
        return (
            torch.as_tensor(grad_x, device=dev, dtype=dt),
            torch.as_tensor(grad_cov_np, device=dev, dtype=dt),
        )


def pdist_tangency_matrix_differentiable(
    centers: torch.Tensor, cov: torch.Tensor
) -> torch.Tensor:
    """Return differentiable tangency distance matrix ``(N,N)`` via ``ellphi.grad``."""
    return _EllphiPdistMatrix.apply(centers, cov)
