import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_topological.nn import VietorisRipsComplex, WassersteinDistance

from tda_ml.distance_backend import (
    compute_distance_matrix_batch,
    rescale_distance_matrix,
    subsample_indices,
)
from tda_ml.numerical_eps import NUMERICAL_EPS
from tda_ml.reproducibility import record_fallback

logger = logging.getLogger(__name__)


class ClassificationLoss(nn.Module):
    """
    Standard Binary Cross-Entropy with Logits for inlier/outlier classification.
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, labels):
        return self.loss_fn(logits.squeeze(-1), labels.float())

class SizeRegularizationLoss(nn.Module):
    """
    Penalizes ellipse scale.

    Modes:
    - quadratic (default): ``w_major * a^2 + w_minor * b^2`` on every point (always on).
    - barrier: ``w * relu(a^2 + b^2 - radius^2)^2`` — zero gradient below ``barrier_radius``.
    - power: ``w * (||axes||^2 / ref)^gamma`` — always on; gradient grows with scale
      (small ellipses: weak pull; large ellipses: stronger than quadratic when gamma > 1).
    - softplus: ``w * softplus(beta * (||axes||^2 / ref - 1))^2`` — smooth ramp centred
      at ``ref``; no hard dead zone, but gentle below reference scale.
    """

    def __init__(
        self,
        w_major: float = 0.1,
        w_minor: float = 0.1,
        mode: str = "quadratic",
        barrier_radius: float = 1.5,
        size_ref: float = 1.34,
        size_power: float = 1.5,
        size_softplus_beta: float = 8.0,
    ):
        super().__init__()
        self.w_major = w_major
        self.w_minor = w_minor
        self.mode = mode
        self.barrier_radius = float(barrier_radius)
        self.size_ref = float(size_ref)
        self.size_power = float(size_power)
        self.size_softplus_beta = float(size_softplus_beta)

    def _weight(self) -> float:
        return 0.5 * (self.w_major + self.w_minor)

    def forward(self, params):
        axes = params[..., 0:2]
        major_axis = axes.max(dim=-1)[0]
        minor_axis = axes.min(dim=-1)[0]
        sq_norm = major_axis**2 + minor_axis**2
        ref = max(self.size_ref, NUMERICAL_EPS)
        weight = self._weight()

        if self.mode == "barrier":
            excess = F.relu(sq_norm - self.barrier_radius**2)
            return weight * (excess**2).mean()
        if self.mode == "power":
            scaled = sq_norm / ref
            return weight * scaled.pow(self.size_power).mean()
        if self.mode == "softplus":
            # centred at ref: ~0 below ref, ~quadratic in excess above ref
            x = self.size_softplus_beta * (sq_norm / ref - 1.0)
            return weight * F.softplus(x).pow(2).mean()
        loss = (self.w_major * (major_axis**2) + self.w_minor * (minor_axis**2)).mean()
        return loss

class AnisotropyPenaltyLoss(nn.Module):
    """
    Shapes the ellipse aspect ratio. The direction of the effect depends on ``mode``.

    Modes:
    - linear: Penalizes aspect ratio (major/minor) directly -> drives ellipses toward circles.
    - barrier: Penalizes aspect ratio squared only above ``barrier_threshold``
      -> free below the threshold, strong push back above it.
    - elongate: Minimizes minor/major (circularity) -> rewards elongation along the
      local principal direction; circle (ratio=1) is the worst case. This reproduces the
      legacy ``aniso_mode: elongate`` behavior that yielded data-aligned ellipses.
    - elongate_barrier: ``elongate`` below the aspect-ratio ceiling, plus ``barrier`` on
      excess above ``barrier_threshold`` (keep tangent elongation, cap needle-like tails).
    """
    def __init__(self, weight=0.01, mode='linear', barrier_threshold=6.0):
        super().__init__()
        self.weight = weight
        self.mode = mode
        self.barrier_threshold = barrier_threshold

    def forward(self, params):
        if abs(self.weight) < 1e-9:
            return torch.tensor(0.0, device=params.device)
            
        axes = params[..., 0:2]
        major_axis = axes.max(dim=-1)[0]
        minor_axis = axes.min(dim=-1)[0]

        if self.mode == 'barrier':
            aspect_ratios = major_axis / (minor_axis + NUMERICAL_EPS)
            barrier_term = F.relu(aspect_ratios - self.barrier_threshold).pow(2).mean()
            loss = 10.0 * barrier_term
        elif self.mode == 'elongate_barrier':
            aspect_ratios = major_axis / (minor_axis + NUMERICAL_EPS)
            elongate = (minor_axis / (major_axis + NUMERICAL_EPS)).mean()
            barrier_term = F.relu(aspect_ratios - self.barrier_threshold).pow(2).mean()
            loss = elongate + 10.0 * barrier_term
        elif self.mode == 'elongate':
            loss = (minor_axis / (major_axis + NUMERICAL_EPS)).mean()
        else:
            aspect_ratios = major_axis / (minor_axis + NUMERICAL_EPS)
            loss = aspect_ratios.mean()
            
        return self.weight * loss

class MinBRegularizationLoss(nn.Module):
    """Penalize minor axis falling below ``target`` (legacy ``lambda_min_b`` / ``min_b_target``)."""

    def __init__(self, weight: float = 0.0, target: float = 0.2):
        super().__init__()
        self.weight = weight
        self.target = target

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        if self.weight <= 0:
            return torch.tensor(0.0, device=params.device)
        axes = params[..., 0:2]
        minor_axis = axes.min(dim=-1)[0]
        return self.weight * F.relu(self.target - minor_axis).mean()

class TopologicalLoss(nn.Module):
    """
    Computes the Topological Loss between the predicted anisotropic filtration
    and the clean ground truth persistence diagram using Wasserstein distance.

    distance_backend:
      - ``mahalanobis``: differentiable anisotropic distance
      - ``ellphi``: tangency distance. With ``ellphi_differentiable=True``,
        gradients can flow to ellipse parameters via ``ellphi.grad`` (numerical
        singularities may still produce NaN/zero gradients).

    ellphi_differentiable:
      If ``False``, use NumPy ``EllipseCloud.pdist_tangency`` only (no gradients).
    """
    def __init__(
        self,
        weight=0.1,
        distance_backend: str = "mahalanobis",
        ellphi_differentiable: bool = True,
        prob_weighting: bool = True,
        eps_scale: float = 1.0,
        scale_mode: str = "fixed",
        max_points: int | None = None,
        strict_topo_samples: bool = True,
        manifest_ref: dict | None = None,
    ):
        super().__init__()
        self.weight = weight
        self.distance_backend = distance_backend.lower().strip()
        self.ellphi_differentiable = ellphi_differentiable
        # When False, outlier-probability weighting of the distance matrix is
        # disabled (probs=None). Only affects the ``mahalanobis`` backend; ``ellphi``
        # never uses probs. Useful for a fair backend ablation against ellphi.
        self.prob_weighting = prob_weighting
        # Filtration-unit alignment between the predicted distance matrix (Mahalanobis
        # or ellphi tangency units) and the Euclidean clean (teacher) PD. Legacy
        # ``topo_eps_scale`` (v73 default 0.7022) multiplied D by a scalar; ``ellphi``
        # tangency distances live on a different scale than Euclidean, so without this
        # the topology loss is dominated by scale rather than shape.
        #   - scale_mode="fixed":  D <- D * eps_scale  (scalar; eps_scale=1.0 == no-op)
        #   - scale_mode="median": bring the *prediction* onto the teacher's Euclidean
        #     scale: D <- D * (m_e / median(offdiag D)), where m_e is the median pairwise
        #     Euclidean distance of the clean cloud (provided by the trainer as
        #     ``clean_scales``). The teacher PD is left untouched. If m_e is unavailable,
        #     fall back to D <- D / median(offdiag D) (per-sample unit median).
        self.eps_scale = float(eps_scale)
        self.scale_mode = str(scale_mode).strip().lower()
        if self.scale_mode not in ("fixed", "median"):
            raise ValueError(
                f"scale_mode must be 'fixed' or 'median', got {scale_mode!r}"
            )
        # Legacy ``topo_loss_max_points``: subsample points before VR/Wasserstein.
        self.max_points = int(max_points) if max_points is not None else None
        if self.max_points is not None and self.max_points < 2:
            raise ValueError(f"max_points must be >= 2, got {self.max_points}")
        self.strict_topo_samples = bool(strict_topo_samples)
        self.manifest_ref = manifest_ref
        self.vr_complex = VietorisRipsComplex(dim=1)
        self.wasserstein = WassersteinDistance(q=2)

    def _rescale_distance_matrix(self, d_mat: torch.Tensor, clean_scale=None) -> torch.Tensor:
        """See :func:`tda_ml.distance_backend.rescale_distance_matrix`."""
        return rescale_distance_matrix(
            d_mat,
            scale_mode=self.scale_mode,
            eps_scale=self.eps_scale,
            clean_scale=clean_scale,
        )

    def _subsample_points(
        self,
        points_i: torch.Tensor,
        params_i: torch.Tensor,
        logits_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = subsample_indices(points_i.shape[0], self.max_points, device=points_i.device)
        if idx is None:
            return points_i, params_i, logits_i
        return points_i[idx], params_i[idx], logits_i[idx]

    def forward(self, points, params, logits, clean_pd_info, clean_scales=None):
        if self.weight <= 0:
            return torch.tensor(0.0, device=points.device)

        batch_size = points.shape[0]

        total_loss = 0.0
        valid_samples = 0
        topo_failures: list[tuple[int, str]] = []

        for i in range(batch_size):
            pts_i, par_i, logits_i = self._subsample_points(
                points[i], params[i], logits[i]
            )
            probs_i = (
                torch.sigmoid(logits_i).squeeze(-1) if self.prob_weighting else None
            )
            d_batch = compute_distance_matrix_batch(
                pts_i.unsqueeze(0),
                par_i.unsqueeze(0),
                probs=probs_i.unsqueeze(0) if probs_i is not None else None,
                symmetrize="max",
                backend=self.distance_backend,
                ellphi_differentiable=self.ellphi_differentiable,
            )
            clean_scale_i = clean_scales[i] if clean_scales is not None else None
            d_mat = self._rescale_distance_matrix(d_batch[0], clean_scale=clean_scale_i)
            try:
                pd_pred_info = self.vr_complex(d_mat, treat_as_distances=True)
                loss_sample = self.wasserstein(pd_pred_info, clean_pd_info[i]) ** 2

                if torch.isnan(loss_sample):
                    msg = f"nan or inf Wasserstein loss at batch_index={i}"
                    if self.strict_topo_samples:
                        raise RuntimeError(f"TopologicalLoss: {msg}")
                    topo_failures.append((i, msg))
                    continue
                total_loss += loss_sample
                valid_samples += 1
            except RuntimeError:
                raise
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc} at batch_index={i}"
                if self.strict_topo_samples:
                    raise RuntimeError(f"TopologicalLoss: {msg}") from exc
                topo_failures.append((i, msg))
                continue

        if topo_failures:
            batch_skipped = batch_size - valid_samples
            first_i, first_msg = topo_failures[0]
            if self.manifest_ref is not None:
                record_fallback(
                    self.manifest_ref,
                    "topo_sample_skip",
                    f"skipped {batch_skipped}/{batch_size} items "
                    f"(first batch_index={first_i}: {first_msg})",
                )
            logger.warning(
                "TopologicalLoss: skipped %d/%d batch items (first batch_index=%s: %s)",
                batch_skipped,
                batch_size,
                first_i,
                first_msg,
            )

        if valid_samples == 0:
            raise RuntimeError(
                "TopologicalLoss: all batch items failed; no valid Wasserstein terms"
            )

        return self.weight * (total_loss / valid_samples)
