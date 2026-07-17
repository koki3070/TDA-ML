"""Unit tests for ellphi-safe pairwise cloud separation."""

from __future__ import annotations

import unittest

import torch

from tda_ml.cloud_separation import (
    apply_noise_with_min_separation,
    sample_uniform_outliers_with_min_separation,
)
from tda_ml.numerical_eps import MIN_ELLPHI_CENTER_SEPARATION


class TestCloudSeparation(unittest.TestCase):
    def test_noise_enforces_min_separation_on_duplicates(self):
        # Padding-style duplicates: identical base points must still separate.
        g = torch.Generator().manual_seed(0)
        base = torch.zeros(30, 2)
        base[:, 0] = torch.linspace(-0.5, 0.5, 15).repeat_interleave(2)
        min_sep = 0.05
        out = apply_noise_with_min_separation(
            base, noise_std=0.08, min_separation=min_sep, generator=g
        )
        d = torch.cdist(out, out)
        d.fill_diagonal_(float("inf"))
        self.assertGreaterEqual(float(d.min()), min_sep)

    def test_noise_unsatisfiable_hard_fails(self):
        g = torch.Generator().manual_seed(0)
        base = torch.zeros(10, 2)
        with self.assertRaises(RuntimeError):
            apply_noise_with_min_separation(
                base,
                noise_std=0.0,
                min_separation=0.1,
                generator=g,
                max_attempts=5,
            )

    def test_uniform_outliers_separated(self):
        g = torch.Generator().manual_seed(1)
        existing = torch.stack(
            [torch.linspace(-0.8, 0.8, 40), torch.zeros(40)], dim=1
        )
        min_sep = MIN_ELLPHI_CENTER_SEPARATION
        outliers = sample_uniform_outliers_with_min_separation(
            15, existing, min_separation=min_sep, generator=g
        )
        d_ex = torch.cdist(outliers, existing).min()
        self.assertGreaterEqual(float(d_ex), min_sep)
        d_out = torch.cdist(outliers, outliers)
        d_out.fill_diagonal_(float("inf"))
        self.assertGreaterEqual(float(d_out.min()), min_sep)

    def test_deterministic(self):
        base = torch.randn(20, 2)
        g1 = torch.Generator().manual_seed(9)
        g2 = torch.Generator().manual_seed(9)
        a = apply_noise_with_min_separation(base, 0.01, generator=g1)
        b = apply_noise_with_min_separation(base, 0.01, generator=g2)
        self.assertTrue(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
