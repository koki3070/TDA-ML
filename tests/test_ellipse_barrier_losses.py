"""Tests for threshold (barrier) ellipse regularizers."""

import unittest

import torch

from tda_ml.losses import AnisotropyPenaltyLoss, SizeRegularizationLoss


class TestEllipseBarrierLosses(unittest.TestCase):
    def test_size_barrier_inactive_below_radius(self):
        loss_fn = SizeRegularizationLoss(
            w_major=0.5, w_minor=0.5, mode="barrier", barrier_radius=2.0
        )
        # major=minor=1 -> ||(a,b)||^2 = 2 < 4
        params = torch.tensor([[[1.0, 1.0, 0.0]]])
        self.assertAlmostEqual(float(loss_fn(params).item()), 0.0, places=6)

    def test_size_barrier_active_above_radius(self):
        loss_fn = SizeRegularizationLoss(
            w_major=0.5, w_minor=0.5, mode="barrier", barrier_radius=1.0
        )
        params = torch.tensor([[[2.0, 2.0, 0.0]]])
        val = float(loss_fn(params).item())
        self.assertGreater(val, 0.0)

    def test_elongate_barrier_keeps_elongate_below_threshold(self):
        loss_fn = AnisotropyPenaltyLoss(
            weight=1.0, mode="elongate_barrier", barrier_threshold=8.0
        )
        thin = torch.tensor([[[2.0, 0.5, 0.0]]])  # ratio=4
        roundish = torch.tensor([[[1.0, 0.8, 0.0]]])  # ratio=1.25
        self.assertLess(float(loss_fn(thin).item()), float(loss_fn(roundish).item()))

    def test_elongate_barrier_penalizes_above_threshold(self):
        loss_fn = AnisotropyPenaltyLoss(
            weight=1.0, mode="elongate_barrier", barrier_threshold=3.0
        )
        ok = torch.tensor([[[3.0, 1.0, 0.0]]])  # ratio=3
        bad = torch.tensor([[[6.0, 1.0, 0.0]]])  # ratio=6
        self.assertLess(float(loss_fn(ok).item()), float(loss_fn(bad).item()))

    def test_size_power_small_gradient_at_small_scale(self):
        ref = 1.34
        loss_fn = SizeRegularizationLoss(
            w_major=0.5, w_minor=0.5, mode="power", size_ref=ref, size_power=1.5
        )
        small = torch.tensor([[[0.2, 0.15, 0.0]]], requires_grad=True)
        large = torch.tensor([[[1.0, 0.9, 0.0]]], requires_grad=True)
        loss_fn(small).backward()
        g_small = small.grad[..., 0:2].abs().max().item()
        small.grad = None
        loss_fn(large).backward()
        g_large = large.grad[..., 0:2].abs().max().item()
        self.assertLess(g_small, g_large)

    def test_size_softplus_smooth_nonzero_below_ref(self):
        ref = 1.34
        loss_fn = SizeRegularizationLoss(
            w_major=0.5,
            w_minor=0.5,
            mode="softplus",
            size_ref=ref,
            size_softplus_beta=8.0,
        )
        below = torch.tensor([[[0.5, 0.4, 0.0]]])  # sq < ref
        above = torch.tensor([[[1.0, 1.0, 0.0]]])  # sq > ref
        self.assertGreater(float(loss_fn(below).item()), 0.0)
        self.assertLess(float(loss_fn(below).item()), float(loss_fn(above).item()))


if __name__ == "__main__":
    unittest.main()
