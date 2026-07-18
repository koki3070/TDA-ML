"""Unit tests for local-PCA tangent outlier sampling."""

from __future__ import annotations

import unittest

import torch

from tda_ml.tangent_outliers import sample_local_pca_tangent_outliers


class TestTangentOutliers(unittest.TestCase):
    def test_count_and_box(self):
        g = torch.Generator().manual_seed(0)
        # Horizontal line → PC1 roughly along x.
        inliers = torch.stack(
            [torch.linspace(-0.5, 0.5, 40), torch.zeros(40)], dim=1
        )
        outliers = sample_local_pca_tangent_outliers(
            inliers,
            8,
            k=8,
            offset_min=0.2,
            offset_max=0.35,
            generator=g,
        )
        self.assertEqual(tuple(outliers.shape), (8, 2))
        self.assertTrue(torch.all(outliers >= -1.0).item())
        self.assertTrue(torch.all(outliers <= 1.0).item())

    def test_deterministic(self):
        inliers = torch.randn(30, 2)
        g1 = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(7)
        a = sample_local_pca_tangent_outliers(inliers, 5, generator=g1)
        b = sample_local_pca_tangent_outliers(inliers, 5, generator=g2)
        self.assertTrue(torch.allclose(a, b))

    def test_min_separation_enforced(self):
        # Outliers must be at least ``min_separation`` from every inlier and from
        # each other so the ellphi tangency derivative w.r.t. mu stays defined.
        g = torch.Generator().manual_seed(3)
        inliers = torch.stack(
            [torch.linspace(-0.5, 0.5, 40), torch.zeros(40)], dim=1
        )
        min_sep = 0.05
        outliers = sample_local_pca_tangent_outliers(
            inliers, 12, k=8, offset_min=0.2, offset_max=0.5,
            generator=g, min_separation=min_sep,
        )
        d_inl = torch.cdist(outliers, inliers).min()
        self.assertGreaterEqual(float(d_inl), min_sep)
        d_out = torch.cdist(outliers, outliers)
        d_out.fill_diagonal_(float("inf"))
        self.assertGreaterEqual(float(d_out.min()), min_sep)

    def test_stroke_clearance_enforced(self):
        # Near-tangent mode: outliers must stay >= stroke_clearance from every
        # inlier so the outlier label never contradicts the geometry.
        g = torch.Generator().manual_seed(5)
        inliers = torch.stack(
            [torch.linspace(-0.5, 0.5, 40), torch.zeros(40)], dim=1
        )
        clearance = 0.08
        outliers = sample_local_pca_tangent_outliers(
            inliers, 12, k=8, offset_min=0.15, offset_max=0.40,
            generator=g, angle_jitter_deg=30.0, stroke_clearance=clearance,
            max_attempts=400,
        )
        d_inl = torch.cdist(outliers, inliers).min()
        self.assertGreaterEqual(float(d_inl), clearance)

    def test_zero_jitter_matches_pure_tangent_stream(self):
        # angle_jitter_deg=0 must not consume extra RNG draws, so the sampled
        # outliers are byte-identical to the historical pure-tangent mode.
        inliers = torch.randn(30, 2)
        g1 = torch.Generator().manual_seed(11)
        g2 = torch.Generator().manual_seed(11)
        a = sample_local_pca_tangent_outliers(inliers, 5, generator=g1)
        b = sample_local_pca_tangent_outliers(
            inliers, 5, generator=g2, angle_jitter_deg=0.0, stroke_clearance=0.0
        )
        self.assertTrue(torch.equal(a, b))

    def test_invalid_jitter_and_clearance_rejected(self):
        inliers = torch.randn(30, 2)
        with self.assertRaises(ValueError):
            sample_local_pca_tangent_outliers(inliers, 5, angle_jitter_deg=-1.0)
        with self.assertRaises(ValueError):
            sample_local_pca_tangent_outliers(inliers, 5, angle_jitter_deg=91.0)
        with self.assertRaises(ValueError):
            sample_local_pca_tangent_outliers(inliers, 5, stroke_clearance=-0.1)

    def test_min_separation_unsatisfiable_hard_fails(self):
        # A tiny box with a large min_separation cannot be satisfied -> hard-fail
        # rather than emitting near-coincident (degenerate) outliers.
        g = torch.Generator().manual_seed(0)
        inliers = torch.stack(
            [torch.linspace(-0.02, 0.02, 20), torch.zeros(20)], dim=1
        )
        with self.assertRaises(RuntimeError):
            sample_local_pca_tangent_outliers(
                inliers, 20, k=5, offset_min=0.01, offset_max=0.02,
                generator=g, min_separation=0.5, box_min=-0.05, box_max=0.05,
            )


if __name__ == "__main__":
    unittest.main()
