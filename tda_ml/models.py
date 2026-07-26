import torch
import torch.nn as nn

from tda_ml.numerical_eps import EIGENVALUE_FLOOR, PCA_RIDGE_EPS

class DecoupledGeometricEncoder(nn.Module):
    r"""
    Decoupled encoder: local neighborhood PCA / eigendecomposition plus separate MLP heads.

    ``k`` sets the number of Euclidean nearest neighbors used for each local covariance.
    ``AnisotropicOutlierClassifier`` constructs this encoder with ``k=10``.
    The constructor default ``k=5`` applies only when you build ``DecoupledGeometricEncoder`` directly.

    Mathematical Architecture:
    1. Local PCA: For each point $x_i$, compute covariance $C_i$ of its $k$-nearest neighbors.
       $$C_i = \frac{1}{k-1} \sum_{j \in KNN(i)} (x_j - \bar{x}_i)(x_j - \bar{x}_i)^\top$$
    2. Canonical Basis: Extract local orientation $\theta_{base}$ and eccentricities using eigen-decomposition.
       $$C_i v = \lambda v \implies \theta_{base} = \text{atan2}(v_{1,y}, v_{1,x})$$
    3. Feature Learning: Process canonical coordinates through MLPs to learn structural corrections.
    """
    def __init__(self, in_dim=2, local_dim=64, k=5):
        if in_dim != 2:
            raise ValueError(
                f"DecoupledGeometricEncoder only supports 2D point clouds (in_dim=2), got in_dim={in_dim}"
            )
        super().__init__()
        self.k = k
        self.local_dim = local_dim
        
        self.local_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, local_dim),
            nn.ReLU()
        )
        
        self.cls_branch = nn.Sequential(
            nn.Linear(in_dim + local_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.topo_branch = nn.Sequential(
            nn.Linear(local_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

    def forward(self, x):
        B, N, D = x.shape
        
        dist_sq = torch.cdist(x, x, p=2) ** 2
        _, idx = torch.topk(-dist_sq, k=self.k, dim=-1)
        
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, self.k)
        flat_x = x.view(B*N, D)
        flat_neighbors = flat_x[idx.view(B, -1) + (batch_idx.view(B, -1) * N), :]
        neighbors = flat_neighbors.view(B, N, self.k, D)
        
        current_points = x.unsqueeze(2)
        relative_coords = neighbors - current_points
        
        mean_neighbor = relative_coords.mean(dim=2, keepdim=True)
        centered = relative_coords - mean_neighbor
        
        cov = torch.matmul(centered.transpose(-1, -2), centered) / (self.k - 1)

        # eigh is not implemented for float16; local PCA in float32 is standard.
        cov32 = cov.float()
        eye2 = torch.eye(2, device=x.device, dtype=torch.float32)
        e, v = torch.linalg.eigh(cov32 + eye2 * PCA_RIDGE_EPS)
        
        v1 = v[:, :, :, 1]
        base_angle = torch.atan2(v1[:, :, 1], v1[:, :, 0]).unsqueeze(-1)
        
        base_axes = torch.sqrt(torch.clamp(e, min=EIGENVALUE_FLOOR))
        base_axes = torch.flip(base_axes, dims=[-1])
        
        base_axes = base_axes / (
            base_axes.max(dim=-1, keepdim=True)[0] + EIGENVALUE_FLOOR
        )
        
        local_coords_canonical = torch.matmul(centered, v)
        
        local_feats = self.local_mlp(local_coords_canonical).max(dim=2)[0]
        
        cls_input = torch.cat([x, local_feats], dim=-1)
        cls_features = self.cls_branch(cls_input)
        
        topo_features = self.topo_branch(local_feats)
            
        return cls_features, topo_features, base_angle, base_axes

class AnisotropicOutlierClassifier(nn.Module):
    r"""
    Predicts outlier probabilities and anisotropic ellipse parameters.

    Uses ``DecoupledGeometricEncoder(..., k=10)`` for local neighborhoods unless you replace
    ``self.encoder`` after construction.

    The model transforms baseline PCA ellipses into "learned" ellipses anchored at each input point:
    1. Outlier Logit: $l_i = f_{cls}(z_i)$
    2. Ellipse Axes: $a_i = a_{i, base} \cdot \exp(\Delta a_i)$, $b_i = b_{i, base} \cdot \exp(\Delta b_i)$
    3. Ellipse Angle: $\theta_i = \theta_{i, base} + \tanh(\Delta \theta_i) \cdot \frac{\pi}{2}$

    ``ellipse_param_dim`` must be ``3`` (``[a, b, theta]`` per point). Five-dimensional
    ellipse outputs are not implemented in this PR.

    ``forward`` returns ellipse parameters with shape ``(B, N, ellipse_param_dim)``.
    """
    def __init__(self, point_dim=2, feature_dim=128, ellipse_param_dim: int = 3):
        super().__init__()
        if ellipse_param_dim != 3:
            raise ValueError(
                "AnisotropicOutlierClassifier supports ellipse_param_dim=3 ([a, b, theta]) only; "
                f"got ellipse_param_dim={ellipse_param_dim}. Five-dimensional outputs are not implemented."
            )
        self.ellipse_param_dim = ellipse_param_dim

        self.encoder = DecoupledGeometricEncoder(in_dim=point_dim, local_dim=64, k=10)
        
        head_in_dim = 128

        self.classification_head = nn.Sequential(
            nn.Linear(head_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.topology_head = nn.Sequential(
            nn.Linear(head_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, ellipse_param_dim),
        )
        nn.init.constant_(self.topology_head[-1].bias[0], -1.0)
        nn.init.constant_(self.topology_head[-1].bias[1], -1.0)

    def forward(self, x):
        cls_feats, topo_feats, base_angle, base_axes = self.encoder(x)

        outlier_logits = self.classification_head(cls_feats)
        raw = self.topology_head(topo_feats)

        axes_scale = torch.exp(raw[:, :, 0:2])
        axes = axes_scale * base_axes
        angle_delta = torch.tanh(raw[:, :, 2:3]) * (torch.pi / 2)
        angle = base_angle + angle_delta
        ellipse_params = torch.cat([axes, angle], dim=2)

        return outlier_logits, ellipse_params
