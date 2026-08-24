"""Clean (teacher) persistence diagrams for the topology loss."""

from __future__ import annotations

import torch
from torch_topological.nn import VietorisRipsComplex

from tda_ml.distance_backend import compute_distance_matrix_batch
from tda_ml.local_pca import (
    TEACHER_MODE_EUCLIDEAN,
    local_pca_ellipse_params,
    normalize_teacher_mode,
)
from tda_ml.numerical_eps import ZERO_PAD_ABS_SUM


def _median_offdiag(d_mat: torch.Tensor) -> float:
    off = d_mat[d_mat > 0]
    if off.numel() == 0:
        raise RuntimeError(
            "Teacher distance matrix has no positive off-diagonal entries; "
            "cannot compute median scale."
        )
    return float(torch.median(off).item())


def compute_clean_teacher_batch(
    clean_points: torch.Tensor,
    vr_complex: VietorisRipsComplex,
    *,
    teacher_mode: str = TEACHER_MODE_EUCLIDEAN,
    distance_backend: str = "ellphi",
    ellphi_differentiable: bool = False,
    local_pca_k: int = 10,
    local_pca_normalize_axes: bool = True,
    max_points: int | None = None,
    need_clean_scales: bool = False,
) -> tuple[list, list[float] | None]:
    """
    Build teacher PD info (and optional per-sample scale targets) for a batch.

    ``teacher_mode``:
      - ``euclidean``: VR on raw clean coordinates (legacy default).
      - ``local_pca``: ideal local-PCA ellipses + ``distance_backend`` distance matrix → VR
        (paper §3.3.2 ``D_ideal``; same filtration units as the prediction side).

    When ``need_clean_scales`` is True (``topo_scale_mode='median'``), returns per-sample
    medians of the teacher filtration: Euclidean pairwise median (euclidean mode) or
    median off-diagonal entries of ``D_ideal`` (local_pca mode).
    """
    mode = normalize_teacher_mode(teacher_mode)
    backend = distance_backend.lower().strip()
    clean_pd_info: list = []
    clean_scales: list[float] | None = [] if need_clean_scales else None

    with torch.no_grad():
        for j in range(clean_points.shape[0]):
            pts = clean_points[j]
            valid_mask = torch.abs(pts).sum(dim=1) > ZERO_PAD_ABS_SUM
            pts = pts[valid_mask]
            if pts.shape[0] < 2:
                raise RuntimeError(
                    f"Clean teacher cloud has {pts.shape[0]} valid inlier points after "
                    "zero-padding mask; need at least 2 for persistence."
                )

            if max_points is not None and pts.shape[0] > max_points:
                idx = torch.randperm(pts.shape[0], device=pts.device)[:max_points]
                pts = pts[idx]

            if mode == TEACHER_MODE_EUCLIDEAN:
                clean_pd_info.append(vr_complex(pts))
                if clean_scales is not None:
                    eu = torch.pdist(pts)
                    clean_scales.append(
                        float(torch.median(eu).item()) if eu.numel() > 0 else None
                    )
                    if clean_scales[-1] is None:
                        raise RuntimeError(
                            "Euclidean teacher cloud has no pairwise distances for scale."
                        )
                continue

            ideal_params = local_pca_ellipse_params(
                pts.unsqueeze(0),
                k=local_pca_k,
                normalize_axes=local_pca_normalize_axes,
            )
            d_batch = compute_distance_matrix_batch(
                pts.unsqueeze(0),
                ideal_params,
                probs=None,
                symmetrize="max",
                backend=backend,
                ellphi_differentiable=ellphi_differentiable,
            )
            d_mat = d_batch[0]
            clean_pd_info.append(vr_complex(d_mat, treat_as_distances=True))
            if clean_scales is not None:
                clean_scales.append(_median_offdiag(d_mat))

    return clean_pd_info, clean_scales
