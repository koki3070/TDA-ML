"""Regression tests for training.topo_distance_mode (mahalanobis vs ellphi)."""

import unittest
import torch

from tda_ml.distance_backend import (
    DISTANCE_MODE_ELLPHI,
    DISTANCE_MODE_MAHALANOBIS,
    compute_topo_distance_matrix,
    mahalanobis_distance_matrix_batched,
    normalize_topo_distance_mode,
)


class TestTopoDistanceMode(unittest.TestCase):
    def test_normalize_aliases(self):
        self.assertEqual(normalize_topo_distance_mode("Mahalanobis"), DISTANCE_MODE_MAHALANOBIS)
        self.assertEqual(normalize_topo_distance_mode("ellphi"), DISTANCE_MODE_ELLPHI)

    def test_mahalanobis_shape(self):
        torch.manual_seed(0)
        b, n = 2, 9
        pt = torch.randn(b, n, 2)
        par = torch.randn(b, n, 3)
        par[:, :, 0:2] = par[:, :, 0:2].abs() + 0.1
        d = compute_topo_distance_matrix(pt, par, distance_mode="mahalanobis")
        self.assertEqual(d.shape, (b, n, n))

    def test_ellphi_finite_and_grad(self):
        torch.manual_seed(0)
        b, n = 1, 7
        pt = torch.randn(b, n, 2, requires_grad=True)
        par = torch.randn(b, n, 3)
        par[:, :, 0:2] = par[:, :, 0:2].abs() + 0.15
        par.requires_grad_(True)
        d_e = compute_topo_distance_matrix(pt, par, distance_mode="ellphi", ellphi_backend="auto")
        d_m = mahalanobis_distance_matrix_batched(pt, par)
        self.assertEqual(d_e.shape, (b, n, n))
        self.assertTrue(torch.isfinite(d_e).all())
        loss = d_e.sum()
        loss.backward()
        self.assertIsNotNone(pt.grad)
        self.assertIsNotNone(par.grad)
        # Forward definitions differ; should not be identical in general
        self.assertGreater((d_e - d_m).abs().mean().item(), 1e-6)

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
            TopologicalLoss(scale_mode="bogus")

    def test_eps_scale_default_is_noop(self):
        """eps_scale=1.0 (fixed) must not change the loss vs. an explicit 1.0."""
        from tda_ml.losses import TopologicalLoss
        pt, par, logits = self._inputs()
        clean = self._clean_pd(pt)
        base = TopologicalLoss(weight=1.0, distance_backend="mahalanobis", prob_weighting=False)
        same = TopologicalLoss(weight=1.0, distance_backend="mahalanobis",
                               prob_weighting=False, eps_scale=1.0)
        l0 = base(pt, par, logits, clean).item()
        l1 = same(pt, par, logits, clean).item()
        self.assertAlmostEqual(l0, l1, places=6)

    def test_eps_scale_changes_loss(self):
        """A non-unit eps_scale rescales the predicted filtration and changes the loss."""
        from tda_ml.losses import TopologicalLoss
        pt, par, logits = self._inputs()
        clean = self._clean_pd(pt)
        base = TopologicalLoss(weight=1.0, distance_backend="mahalanobis", prob_weighting=False)
        scaled = TopologicalLoss(weight=1.0, distance_backend="mahalanobis",
                                 prob_weighting=False, eps_scale=0.3)
        l0 = base(pt, par, logits, clean).item()
        ls = scaled(pt, par, logits, clean).item()
        self.assertGreater(abs(l0 - ls), 1e-6)

    def test_median_mode_runs_and_grads(self):
        from tda_ml.losses import TopologicalLoss
        pt, par, logits = self._inputs()
        par = par.clone().requires_grad_(True)
        clean = self._clean_pd(pt)
        loss_fn = TopologicalLoss(weight=1.0, distance_backend="mahalanobis",
                                  prob_weighting=False, scale_mode="median")
        # clean_scales (m_e) provided by the trainer; loss brings prediction onto it.
        clean_scales = [float(torch.pdist(pt[i]).median()) for i in range(pt.shape[0])]
        loss = loss_fn(pt, par, logits, clean, clean_scales=clean_scales)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(par.grad)

    def test_median_aligns_prediction_to_teacher_scale(self):
        """median mode must rescale the prediction so its median matches m_e,
        leaving the teacher untouched."""
        from tda_ml.losses import TopologicalLoss
        pt, par, logits = self._inputs()
        loss_fn = TopologicalLoss(weight=1.0, distance_backend="mahalanobis",
                                  prob_weighting=False, scale_mode="median")
        i = 0
        from tda_ml.distance_backend import compute_distance_matrix_batch
        D = compute_distance_matrix_batch(pt, par, probs=None, symmetrize="max",
                                          backend="mahalanobis")
        m_e = float(torch.pdist(pt[i]).median())
        rescaled = loss_fn._rescale_distance_matrix(D[i], clean_scale=m_e)
        off = rescaled[rescaled > 0]
        # After alignment the predicted median equals the teacher Euclidean median.
        self.assertAlmostEqual(float(off.median()), m_e, places=4)

    def test_mahalanobis_extreme_params_remain_finite(self):
        """極小/極大軸長でも距離行列が非有限値にならないことを確認。"""
        b, n = 1, 5
        pt = torch.tensor(
            [[[1e-9, -1e-9], [1.0, 2.0], [3.0, -1.0], [0.5, 0.1], [-2.0, 1.5]]],
            dtype=torch.float64,
        )
        par = torch.tensor(
            [[[1e-8, 1e8, 0.0],
              [2e-8, 5e7, 0.3],
              [1e7, 2e-7, -0.4],
              [5e-8, 8e7, 0.7],
              [2e7, 3e-8, -1.0]]],
            dtype=torch.float64,
        )
        d = compute_topo_distance_matrix(pt, par, distance_mode="mahalanobis")
        self.assertEqual(d.shape, (b, n, n))
        self.assertTrue(torch.isfinite(d).all())


class TestMinBAndTopoSubsampling(unittest.TestCase):
    def test_min_b_penalizes_small_minor_axis(self):
        from tda_ml.losses import MinBRegularizationLoss
        loss_fn = MinBRegularizationLoss(weight=1.0, target=0.5)
        params = torch.tensor([[[0.8, 0.1, 0.0], [0.6, 0.4, 0.0]]])
        loss = loss_fn(params)
        self.assertGreater(loss.item(), 0.0)

    def test_min_b_zero_weight_is_noop(self):
        from tda_ml.losses import MinBRegularizationLoss
        loss_fn = MinBRegularizationLoss(weight=0.0, target=0.5)
        params = torch.tensor([[[0.8, 0.1, 0.0]]])
        self.assertEqual(loss_fn(params).item(), 0.0)

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
            distance_backend="mahalanobis",
            prob_weighting=False,
            max_points=12,
        )
        loss = loss_fn(pt, par, logits, clean)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
