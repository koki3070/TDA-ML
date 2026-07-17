"""Ellipse-filtration W-Dist (prediction PD vs clean teacher PD).

Same definition as ``TopologicalLoss``: predicted ellipses on the (noisy) cloud
vs teacher PD from ``compute_clean_teacher_batch`` (e.g. local PCA + ellphi).
DBSCAN is **not** involved in this metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch_topological.nn import VietorisRipsComplex, WassersteinDistance

from tda_ml.distance_backend import (
    compute_distance_matrix_batch,
    rescale_distance_matrix,
    subsample_indices,
)
from tda_ml.persistence_dimensions import (
    normalize_homology_dimensions,
    select_persistence_dimensions,
)
from tda_ml.teacher_pd import compute_clean_teacher_batch


@dataclass(frozen=True)
class TopoWdistOptions:
    """Eval options; method-defining fields have no silent defaults."""

    teacher_mode: str
    distance_backend: str
    homology_dimensions: tuple[int, ...]
    prob_weighting: bool
    teacher_local_pca_k: int = 10
    teacher_local_pca_normalize_axes: bool = True
    eps_scale: float = 1.0
    scale_mode: str = "fixed"
    max_points: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "homology_dimensions",
            normalize_homology_dimensions(self.homology_dimensions),
        )
        object.__setattr__(
            self,
            "teacher_mode",
            str(self.teacher_mode).strip().lower(),
        )
        object.__setattr__(
            self,
            "distance_backend",
            str(self.distance_backend).strip().lower(),
        )
        object.__setattr__(self, "prob_weighting", bool(self.prob_weighting))
        object.__setattr__(
            self,
            "scale_mode",
            str(self.scale_mode).strip().lower(),
        )
        if self.scale_mode not in ("fixed", "median"):
            raise ValueError(
                f"scale_mode must be 'fixed' or 'median', got {self.scale_mode!r}"
            )


def topo_wdist_options_from_config(config: dict[str, Any]) -> TopoWdistOptions:
    """Build options from config; missing method fields hard-fail."""
    loss_cfg = config.get("loss") or {}
    training_cfg = config.get("training") or {}
    topo_cfg = (config.get("model") or {}).get("topology_loss") or {}

    missing_topo = [
        key
        for key in ("distance_backend", "homology_dimensions", "prob_weighting")
        if key not in topo_cfg
    ]
    if missing_topo:
        raise ValueError(
            "topo W-Dist requires explicit model.topology_loss fields "
            f"{missing_topo}; refusing silent defaults"
        )

    teacher_mode = loss_cfg.get("teacher_mode", training_cfg.get("teacher_mode"))
    if teacher_mode is None:
        raise ValueError(
            "topo W-Dist requires explicit loss.teacher_mode "
            "(or training.teacher_mode); refusing silent euclidean default"
        )

    max_pts = training_cfg.get("topo_loss_max_points", loss_cfg.get("topo_loss_max_points"))
    return TopoWdistOptions(
        teacher_mode=str(teacher_mode).strip().lower(),
        distance_backend=str(topo_cfg["distance_backend"]).lower().strip(),
        teacher_local_pca_k=int(
            loss_cfg.get(
                "teacher_local_pca_k",
                training_cfg.get("teacher_local_pca_k", 10),
            )
        ),
        teacher_local_pca_normalize_axes=bool(
            loss_cfg.get(
                "teacher_local_pca_normalize_axes",
                training_cfg.get("teacher_local_pca_normalize_axes", True),
            )
        ),
        eps_scale=float(
            loss_cfg.get("topo_eps_scale", training_cfg.get("topo_eps_scale", 1.0))
        ),
        scale_mode=str(
            loss_cfg.get("topo_scale_mode", training_cfg.get("topo_scale_mode", "fixed"))
        ).strip().lower(),
        max_points=int(max_pts) if max_pts is not None else None,
        prob_weighting=bool(topo_cfg["prob_weighting"]),
        homology_dimensions=normalize_homology_dimensions(
            topo_cfg["homology_dimensions"]
        ),
    )


def _subsample_cloud(
    points: torch.Tensor,
    params: torch.Tensor,
    max_points: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = subsample_indices(points.shape[0], max_points, device=points.device)
    if idx is None:
        return points, params
    return points[idx], params[idx]


def compute_topo_wdist(
    points: np.ndarray,
    params: np.ndarray,
    clean_pc: np.ndarray,
    options: TopoWdistOptions,
) -> float:
    """
    Wasserstein-2 distance over configured homology dimensions.

    ``points`` / ``params``: full noisy cloud and learned ellipses (DBSCAN unused).
    ``clean_pc``: padded clean inlier coordinates for the teacher PD.
    ``options`` is required (no silent TopoWdistOptions defaults).
    """
    if options is None:
        raise ValueError(
            "compute_topo_wdist requires explicit TopoWdistOptions; "
            "refusing silent defaults"
        )
    opts = options
    pts = torch.as_tensor(points, dtype=torch.float32)
    par = torch.as_tensor(params[..., :3], dtype=torch.float32)
    clean = torch.as_tensor(clean_pc, dtype=torch.float32)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (N, 2); got {pts.shape}")
    if par.shape[0] != pts.shape[0]:
        raise ValueError(f"params length {par.shape[0]} != points {pts.shape[0]}")

    vr = VietorisRipsComplex(dim=1)
    wdist_fn = WassersteinDistance(q=2)

    with torch.no_grad():
        clean_batch = clean.unsqueeze(0)
        clean_pd_info, clean_scales = compute_clean_teacher_batch(
            clean_batch,
            vr,
            teacher_mode=opts.teacher_mode,
            distance_backend=opts.distance_backend,
            ellphi_differentiable=False,
            local_pca_k=opts.teacher_local_pca_k,
            local_pca_normalize_axes=opts.teacher_local_pca_normalize_axes,
            max_points=opts.max_points,
            need_clean_scales=(opts.scale_mode == "median"),
        )

        pts_i, par_i = _subsample_cloud(pts, par, opts.max_points)
        d_batch = compute_distance_matrix_batch(
            pts_i.unsqueeze(0),
            par_i.unsqueeze(0),
            probs=None,
            symmetrize="max",
            backend=opts.distance_backend,
            ellphi_differentiable=False,
        )
        clean_scale_i = clean_scales[0] if clean_scales is not None else None
        d_mat = rescale_distance_matrix(
            d_batch[0],
            scale_mode=opts.scale_mode,
            eps_scale=opts.eps_scale,
            clean_scale=clean_scale_i,
        )
        pd_pred = vr(d_mat, treat_as_distances=True)
        pd_pred_selected = select_persistence_dimensions(
            pd_pred, opts.homology_dimensions
        )
        clean_pd_selected = select_persistence_dimensions(
            clean_pd_info[0], opts.homology_dimensions
        )
        w2_sq = wdist_fn(pd_pred_selected, clean_pd_selected) ** 2
        value = float(w2_sq.item())
        if not np.isfinite(value):
            raise RuntimeError("non-finite topo W-Dist")
        return value
