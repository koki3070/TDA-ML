"""Gradient health for anisotropic distance matrices (issue #61)."""

import torch

from tda_ml.topology import compute_anisotropic_distance_matrix


def test_anisotropic_distance_diagonal_is_exactly_zero():
    batch_size, num_points = 1, 5
    points = torch.randn(batch_size, num_points, 2, dtype=torch.float32)
    params = torch.tensor(
        [
            [
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    probs = torch.full((batch_size, num_points), 0.2, dtype=torch.float32)

    dist = compute_anisotropic_distance_matrix(
        points, params, probs=probs, symmetrize="max"
    )
    diag = dist.diagonal(dim1=-2, dim2=-1)
    assert torch.all(diag == 0.0)
    assert torch.all(dist[..., ~torch.eye(num_points, dtype=torch.bool)] > 0)


def test_anisotropic_distance_backward_finite_at_zero_diagonal():
    """Self-distance diagonal is zero; backward must not emit inf/nan gradients."""
    batch_size, num_points = 1, 5
    points = torch.randn(batch_size, num_points, 2, dtype=torch.float32)
    params = torch.tensor(
        [
            [
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    probs = torch.full((batch_size, num_points), 0.2, dtype=torch.float32)

    dist = compute_anisotropic_distance_matrix(
        points, params, probs=probs, symmetrize="max"
    )
    assert torch.all(dist.diagonal(dim1=-2, dim2=-1) == 0.0)

    dist.sum().backward()
    assert params.grad is not None
    assert torch.isfinite(params.grad).all()


def test_anisotropic_distance_backward_no_sqrt_anomaly_on_diagonal():
    with torch.autograd.detect_anomaly():
        points = torch.randn(1, 8, 2, dtype=torch.float32, requires_grad=True)
        params = torch.rand(1, 8, 3, dtype=torch.float32, requires_grad=True)
        probs = torch.full((1, 8), 0.3, dtype=torch.float32)
        dist = compute_anisotropic_distance_matrix(
            points, params, probs=probs, symmetrize="max"
        )
        dist.sum().backward()
    assert torch.isfinite(points.grad).all()
    assert torch.isfinite(params.grad).all()


def test_anisotropic_distance_backward_finite_with_coincident_points():
    """Off-diagonal dist_sq is zero when two centers coincide; backward must stay finite."""
    points = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    params = torch.tensor(
        [[[1.0, 0.5, 0.0], [1.0, 0.5, 0.0], [1.0, 0.5, 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )

    dist = compute_anisotropic_distance_matrix(points, params, symmetrize="max")
    assert dist[0, 0, 2].item() == 0.0
    assert dist[0, 2, 0].item() == 0.0

    dist.sum().backward()
    assert torch.isfinite(points.grad).all()
    assert torch.isfinite(params.grad).all()


def test_anisotropic_distance_backward_finite_with_close_points():
    """Near-coincident pairs have small but positive off-diagonal distances."""
    base = torch.randn(1, 6, 2, dtype=torch.float32)
    points = base.clone().requires_grad_(True)
    with torch.no_grad():
        points[0, 1] = points[0, 0] + 1e-4

    params = torch.rand(1, 6, 3, dtype=torch.float32, requires_grad=True)
    probs = torch.full((1, 6), 0.25, dtype=torch.float32)

    dist = compute_anisotropic_distance_matrix(
        points, params, probs=probs, symmetrize="max"
    )
    offdiag = ~torch.eye(6, dtype=torch.bool)
    assert torch.all(dist[0, offdiag] > 0)

    dist.sum().backward()
    assert torch.isfinite(points.grad).all()
    assert torch.isfinite(params.grad).all()
