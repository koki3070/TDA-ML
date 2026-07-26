"""Tests for ellipse-filtration topo W-Dist (eval metric)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from tda_ml.local_pca import local_pca_ellipse_params
from tda_ml.topo_wdist import TopoWdistOptions, compute_topo_wdist


class TestTopoWdist(unittest.TestCase):
    def _circle(self, n: int = 12) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float32)

    def test_identical_clouds_near_zero_mahalanobis(self):
        clean = self._circle(14)
        noisy = clean.copy()
        params = local_pca_ellipse_params(torch.from_numpy(noisy).unsqueeze(0)).squeeze(0).numpy()
        opts = TopoWdistOptions(
            teacher_mode="local_pca",
            distance_backend="mahalanobis",
            homology_dimensions=(0, 1),
            prob_weighting=False,
        )
        w = compute_topo_wdist(noisy, params, clean, opts)
        self.assertTrue(np.isfinite(w))
        self.assertLess(w, 0.05)

    def test_different_clouds_positive(self):
        clean = self._circle(14)
        noisy = clean + 0.3 * np.random.default_rng(0).standard_normal(clean.shape).astype(
            np.float32
        )
        params = local_pca_ellipse_params(torch.from_numpy(noisy).unsqueeze(0)).squeeze(0).numpy()
        opts = TopoWdistOptions(
            teacher_mode="local_pca",
            distance_backend="mahalanobis",
            homology_dimensions=(0, 1),
            prob_weighting=False,
        )
        w_same = compute_topo_wdist(clean, params[: clean.shape[0]], clean, opts)
        w_diff = compute_topo_wdist(noisy, params, clean, opts)
        self.assertGreater(w_diff, w_same)


if __name__ == "__main__":
    unittest.main()
