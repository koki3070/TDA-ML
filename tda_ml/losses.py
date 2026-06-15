import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_topological.nn import VietorisRipsComplex, WassersteinDistance

from tda_ml.distance_backend import compute_distance_matrix_batch
from tda_ml.topology import compute_anisotropic_distance_matrix

# Distance-mode aliases for topology loss (kept for config/test compatibility).
logger = logging.getLogger(__name__)

DISTANCE_MODE_MAHALANOBIS = "mahalanobis"
DISTANCE_MODE_ELLPHI = "ellphi"


def normalize_topo_distance_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m == "mahalanobis":
        return DISTANCE_MODE_MAHALANOBIS
    if m == "ellphi":
        return DISTANCE_MODE_ELLPHI
    raise ValueError(f"Unknown topo distance mode: {mode!r}")


def compute_topo_distance_matrix(
    points: torch.Tensor,
    params: torch.Tensor,
    *,
    distance_mode: str = "mahalanobis",
    ellphi_backend: str = "auto",
) -> torch.Tensor:
    """
    Shared topology-loss entrypoint: map batched points/ellipse params to ``(B,N,N)`` distances.

    ``distance_mode`` is ``mahalanobis`` or ``ellphi``.
    When ``ellphi_backend='auto'``, a differentiable ellphi path is preferred and
    falls back to NumPy when unavailable.
    """
    backend = normalize_topo_distance_mode(distance_mode)
    eb = str(ellphi_backend).strip().lower()
    ellphi_diff = eb in ("auto", "torch", "grad", "differentiable", "1", "true", "yes")
    return compute_distance_matrix_batch(
        points,
        params,
        probs=None,
        symmetrize="max",
        backend=backend,
        ellphi_differentiable=ellphi_diff,
    )


def mahalanobis_distance_matrix_batched(points: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Mahalanobis-style batched distance matrix without probability weighting."""
    return compute_anisotropic_distance_matrix(
        points, params, probs=None, symmetrize="max"
    )


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
    Penalizes the size of estimated ellipses to prevent over-expansion.
    
    Formula: L = lambda_major * a^2 + lambda_minor * b^2
    """
    def __init__(self, w_major=0.1, w_minor=0.1):
        super().__init__()
        self.w_major = w_major
        self.w_minor = w_minor

    def forward(self, params):
        axes = params[..., 2:4]
        major_axis = axes.max(dim=-1)[0]
        minor_axis = axes.min(dim=-1)[0]
        loss = (self.w_major * (major_axis**2) + self.w_minor * (minor_axis**2)).mean()
        return loss

class AnisotropyPenaltyLoss(nn.Module):
    """
    Prevents ellipses from becoming too elongated by penalizing high aspect ratios.
    
    Modes:
    - linear: Penalizes aspect ratio (major/minor) directly.
    - barrier: Penalizes aspect ratio squared only above a certain threshold.
    """
    def __init__(self, weight=0.01, mode='linear', barrier_threshold=6.0):
        super().__init__()
        self.weight = weight
        self.mode = mode
        self.barrier_threshold = barrier_threshold

    def forward(self, params):
        if abs(self.weight) < 1e-9:
            return torch.tensor(0.0, device=params.device)
            
        axes = params[..., 2:4]
        major_axis = axes.max(dim=-1)[0]
        minor_axis = axes.min(dim=-1)[0]
        
        aspect_ratios = major_axis / (minor_axis + 1e-6)
        
        if self.mode == 'barrier':
            barrier_term = F.relu(aspect_ratios - self.barrier_threshold).pow(2).mean()
            loss = 10.0 * barrier_term
        else:
            loss = aspect_ratios.mean()
            
        return self.weight * loss

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
    ):
        super().__init__()
        self.weight = weight
        self.distance_backend = distance_backend.lower().strip()
        self.ellphi_differentiable = ellphi_differentiable
        self.vr_complex = VietorisRipsComplex(dim=1)
        self.wasserstein = WassersteinDistance(q=2)

    def forward(self, points, params, logits, clean_pd_info):
        if self.weight <= 0:
            return torch.tensor(0.0, device=points.device)

        batch_size = points.shape[0]
        probs_outlier = torch.sigmoid(logits).squeeze(-1)

        D_prime = compute_distance_matrix_batch(
            points,
            params,
            probs=probs_outlier,
            symmetrize="max",
            backend=self.distance_backend,
            ellphi_differentiable=self.ellphi_differentiable,
        )
        
        total_loss = 0.0
        valid_samples = 0
        topo_failures: list[tuple[int, str]] = []

        for i in range(batch_size):
            d_mat = D_prime[i]
            try:
                pd_pred_info = self.vr_complex(d_mat, treat_as_distances=True)
                loss_sample = self.wasserstein(pd_pred_info, clean_pd_info[i]) ** 2

                if not torch.isnan(loss_sample):
                    total_loss += loss_sample
                    valid_samples += 1
                else:
                    topo_failures.append((i, "nan or inf Wasserstein loss"))
            except Exception as exc:
                topo_failures.append((i, f"{type(exc).__name__}: {exc}"))
                continue

        if topo_failures:
            batch_skipped = batch_size - valid_samples
            first_i, first_msg = topo_failures[0]
            logger.warning(
                "TopologicalLoss: skipped %d/%d batch items (first batch_index=%s: %s)",
                batch_skipped,
                batch_size,
                first_i,
                first_msg,
            )

        if valid_samples == 0:
            return torch.tensor(0.0, device=points.device, requires_grad=True)

        return self.weight * (total_loss / valid_samples)
