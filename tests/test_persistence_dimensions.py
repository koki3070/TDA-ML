"""Tests for explicit H0/H1 selection in topology losses."""

from __future__ import annotations

import unittest

import torch
from torch_topological.nn import VietorisRipsComplex, WassersteinDistance

from tda_ml.persistence_dimensions import (
    normalize_homology_dimensions,
    select_persistence_dimensions,
)
from tda_ml.losses import TopologicalLoss
from tda_ml.topo_wdist import topo_wdist_options_from_config


class TestPersistenceDimensions(unittest.TestCase):
    def setUp(self):
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        self.info = VietorisRipsComplex(dim=1)(points)

    def test_default_selects_h0_and_h1(self):
        self.assertEqual(normalize_homology_dimensions(None), (0, 1))
        selected = select_persistence_dimensions(self.info, None)
        self.assertEqual([item.dimension for item in selected], [0, 1])

    def test_h1_only_selects_one_diagram(self):
        selected = select_persistence_dimensions(self.info, [1])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].dimension, 1)

    def test_invalid_dimensions_hard_fail(self):
        with self.assertRaises(ValueError):
            normalize_homology_dimensions([])
        with self.assertRaises(ValueError):
            normalize_homology_dimensions([1, 1])
        with self.assertRaises(ValueError):
            normalize_homology_dimensions([2])

    def test_wasserstein_h1_only_differs_from_h0_h1(self):
        # Use two actual point clouds for a stable comparison.
        x = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        y = torch.tensor(
            [[0.0, 0.0], [1.2, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        vr = VietorisRipsComplex(dim=1)
        px, py = vr(x), vr(y)
        wd = WassersteinDistance(q=2)
        both = wd(px, py)
        h1 = wd(
            select_persistence_dimensions(px, [1]),
            select_persistence_dimensions(py, [1]),
        )
        self.assertTrue(torch.isfinite(both))
        self.assertTrue(torch.isfinite(h1))
        self.assertNotEqual(float(both), float(h1))

    def test_topo_wdist_options_read_h1_only(self):
        cfg = {"model": {"topology_loss": {"homology_dimensions": [1]}}}
        opts = topo_wdist_options_from_config(cfg)
        self.assertEqual(opts.homology_dimensions, (1,))

    def test_topological_loss_runs_with_h1_only(self):
        theta = torch.linspace(0.0, 2.0 * torch.pi, 9)[:-1]
        points = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1).unsqueeze(0)
        params = torch.zeros(1, 8, 3)
        params[..., :2] = 1.0
        logits = torch.zeros(1, 8, 1)
        clean = [VietorisRipsComplex(dim=1)(points[0])]
        loss_fn = TopologicalLoss(
            weight=1.0,
            distance_backend="mahalanobis",
            prob_weighting=False,
            homology_dimensions=[1],
        )
        loss = loss_fn(points, params, logits, clean)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss_fn.homology_dimensions, (1,))


if __name__ == "__main__":
    unittest.main()
