"""Regression tests for training topo distance (ellphi-only PD filtration)."""

import unittest

import torch

from tda_ml.dbscan import compute_anisotropic_distance_matrix_np
from tda_ml.distance_backend import (
    DISTANCE_MODE_ELLPHI,
    compute_distance_matrix_batch,
    compute_topo_distance_matrix,
    normalize_topo_distance_mode,
)


class TestTopoDistanceMode(unittest.TestCase):
    def test_normalize_ellphi(self):
        self.assertEqual(normalize_topo_distance_mode("ellphi"), DISTANCE_MODE_ELLPHI)
        self.assertEqual(normalize_topo_distance_mode("Ellphi"), DISTANCE_MODE_ELLPHI)

    def test_normalize_mahalanobis_hard_fails(self):
        with self.assertRaisesRegex(ValueError, "not supported for training PD"):
            normalize_topo_distance_mode("mahalanobis")

    def test_ellphi_probability_weighting_hard_fails(self):
        points = torch.zeros(1, 3, 2)
        params = torch.ones(1, 3, 3)
        probs = torch.full((1, 3), 0.5)
        with self.assertRaisesRegex(RuntimeError, "does not implement probability weighting"):
            compute_distance_matrix_batch(
                points,
                params,
                probs=probs,
                symmetrize="max",
                backend="ellphi",
            )

    def test_ellphi_dbscan_probability_weighting_hard_fails(self):
        points = torch.zeros(3, 2)
        params = torch.ones(3, 3)
        probs = torch.full((3,), 0.5)
        with self.assertRaisesRegex(RuntimeError, "does not implement probability weighting"):
            compute_anisotropic_distance_matrix_np(
                points,
                params,
                probs=probs,
                backend="ellphi",
            )

    def test_ellphi_shape(self):
        torch.manual_seed(0)
        b, n = 2, 9
        pt = torch.randn(b, n, 2)
        par = torch.randn(b, n, 3)
        par[:, :, 0:2] = par[:, :, 0:2].abs() + 0.1
        d = compute_topo_distance_matrix(pt, par, distance_mode="ellphi")
        self.assertEqual(d.shape, (b, n, n))

    def test_ellphi_finite_and_grad(self):
        torch.manual_seed(0)
        b, n = 1, 7
        pt = torch.randn(b, n, 2, requires_grad=True)
        par = torch.randn(b, n, 3)
        par[:, :, 0:2] = par[:, :, 0:2].abs() + 0.15
        par.requires_grad_(True)
        d_e = compute_topo_distance_matrix(pt, par, distance_mode="ellphi", ellphi_backend="auto")
        self.assertEqual(d_e.shape, (b, n, n))
        self.assertTrue(torch.isfinite(d_e).all())
        loss = d_e.sum()
        loss.backward()
        self.assertIsNotNone(pt.grad)
        self.assertIsNotNone(par.grad)

    def test_normalize_invalid_mode_raises_value_error(self):
        with self.assertRaises(ValueError):
            normalize_topo_distance_mode("unknown-mode")


class TestTopoEpsScale(unittest.TestCase):
    """eps_scale / scale_mode (filtration-unit alignment) behavior."""

    def _inputs(self):
        torch.manual_seed(1)
        b, n = 2, 8
        pt = torch.randn(b, n, 2)
        par = torch.randn(b, n, 3)
        par[:, :, 0:2] = par[:, :, 0:2].abs() + 0.2
        logits = torch.zeros(b, n, 1)
        return pt, par, logits

    def _clean_pd(self, pt):
        from torch_topological.nn import VietorisRipsComplex

        vr = VietorisRipsComplex(dim=1)
        return [vr(pt[i]) for i in range(pt.shape[0])]

    def test_invalid_scale_mode_raises(self):
        from tda_ml.losses import TopologicalLoss

        with self.assertRaises(ValueError):
            TopologicalLoss(scale_mode="bogus", homology_dimensions=[0, 1])

    def test_eps_scale_default_is_noop(self):
        """eps_scale=1.0 (fixed) must not change the loss vs. an explicit 1.0."""
        from tda_ml.losses import TopologicalLoss

        pt, par, logits = self._inputs()
        clean = self._clean_pd(pt)
        base = TopologicalLoss(
            weight=1.0,
            distance_backend="ellphi",
            prob_weighting=False,
            homology_dimensions=[0, 1],
        )
        same = TopologicalLoss(
            weight=1.0,
            distance_backend="ellphi",
            prob_weighting=False,
            eps_scale=1.0,
            homology_dimensions=[0, 1],
        )
        l0 = base(pt, par, logits, clean).item()
        l1 = same(pt, par, logits, clean).item()
        self.assertAlmostEqual(l0, l1, places=6)

    def test_eps_scale_changes_loss(self):
        """A non-unit eps_scale rescales the predicted filtration and changes the loss."""
        from tda_ml.losses import TopologicalLoss

        pt, par, logits = self._inputs()
        clean = self._clean_pd(pt)
        base = TopologicalLoss(
            weight=1.0,
            distance_backend="ellphi",
            prob_weighting=False,
            homology_dimensions=[0, 1],
        )
        scaled = TopologicalLoss(
            weight=1.0,
            distance_backend="ellphi",
            prob_weighting=False,
            eps_scale=0.3,
            homology_dimensions=[0, 1],
        )
        l0 = base(pt, par, logits, clean).item()
        ls = scaled(pt, par, logits, clean).item()
        self.assertGreater(abs(l0 - ls), 1e-6)

    def test_median_mode_runs_and_grads(self):
        from tda_ml.losses import TopologicalLoss

        pt, par, logits = self._inputs()
        par = par.clone().requires_grad_(True)
        clean = self._clean_pd(pt)
        loss_fn = TopologicalLoss(
            weight=1.0,
            distance_backend="ellphi",
            prob_weighting=False,
            scale_mode="median",
            homology_dimensions=[0, 1],
        )
        clean_scales = [float(torch.pdist(pt[i]).median()) for i in range(pt.shape[0])]
        loss = loss_fn(pt, par, logits, clean, clean_scales=clean_scales)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(par.grad)

    def test_median_aligns_prediction_to_teacher_scale(self):
        """median mode must rescale the prediction so its median matches m_e."""
        from tda_ml.losses import TopologicalLoss

        pt, par, logits = self._inputs()
        loss_fn = TopologicalLoss(
            weight=1.0,
            distance_backend="ellphi",
            prob_weighting=False,
            scale_mode="median",
            homology_dimensions=[0, 1],
        )
        i = 0
        D = compute_distance_matrix_batch(
            pt, par, probs=None, symmetrize="max", backend="ellphi"
        )
        m_e = float(torch.pdist(pt[i]).median())
        rescaled = loss_fn._rescale_distance_matrix(D[i], clean_scale=m_e)
        off = rescaled[rescaled > 0]
        self.assertAlmostEqual(float(off.median()), m_e, places=4)

    def test_ellphi_degenerate_extreme_params_hard_fail(self):
        """極小/極大軸で ellphi が NaN になる場合は黙って有限値にせず hard-fail。"""
        pt = torch.tensor(
            [[[1e-9, -1e-9], [1.0, 2.0], [3.0, -1.0], [0.5, 0.1], [-2.0, 1.5]]],
            dtype=torch.float64,
        )
        par = torch.tensor(
            [
                [
                    [1e-8, 1e8, 0.0],
                    [2e-8, 5e7, 0.3],
                    [1e7, 2e-7, -0.4],
                    [5e-8, 8e7, 0.7],
                    [2e7, 3e-8, -1.0],
                ]
            ],
            dtype=torch.float64,
        )
        with self.assertRaisesRegex(RuntimeError, "degenerate ellipse|NaN"):
            compute_topo_distance_matrix(pt, par, distance_mode="ellphi")


class TestTopoSubsampling(unittest.TestCase):
    def test_topo_max_points_subsamples(self):
        from tda_ml.losses import TopologicalLoss

        torch.manual_seed(0)
        b, n = 1, 20
        pt = torch.randn(b, n, 2)
        par = torch.randn(b, n, 3)
        par[:, :, 0:2] = par[:, :, 0:2].abs() + 0.2
        logits = torch.zeros(b, n, 1)
        from torch_topological.nn import VietorisRipsComplex

        vr = VietorisRipsComplex(dim=1)
        clean = [vr(pt[0, :12])]
        loss_fn = TopologicalLoss(
            weight=1.0,
            distance_backend="ellphi",
            prob_weighting=False,
            max_points=12,
            homology_dimensions=[0, 1],
        )
        loss = loss_fn(pt, par, logits, clean)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
