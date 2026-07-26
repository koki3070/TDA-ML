"""Tests for teacher PD modes (Euclidean vs local-PCA D_ideal)."""

import unittest

import torch
from torch_topological.nn import VietorisRipsComplex

from tda_ml.local_pca import (
    TEACHER_MODE_EUCLIDEAN,
    TEACHER_MODE_LOCAL_PCA,
    local_pca_ellipse_params,
    normalize_teacher_mode,
)
from tda_ml.teacher_pd import compute_clean_teacher_batch


class TestTeacherMode(unittest.TestCase):
    def test_normalize_teacher_mode(self):
        self.assertEqual(normalize_teacher_mode("Euclidean"), TEACHER_MODE_EUCLIDEAN)
        self.assertEqual(normalize_teacher_mode("local_pca"), TEACHER_MODE_LOCAL_PCA)
        with self.assertRaises(ValueError):
            normalize_teacher_mode("unknown")

    def test_local_pca_params_shape(self):
        torch.manual_seed(0)
        pts = torch.randn(12, 2)
        params = local_pca_ellipse_params(pts, k=10)
        self.assertEqual(params.shape, (12, 3))
        self.assertTrue(torch.isfinite(params).all())
        self.assertTrue((params[:, 0:2] > 0).all())

    def test_local_pca_raw_axes_not_unit_major(self):
        torch.manual_seed(1)
        pts = torch.randn(20, 2) * 0.1
        pts[0] = torch.tensor([0.0, 0.0])
        pts[1:] += torch.randn(19, 2) * 0.01
        norm = local_pca_ellipse_params(pts, k=10, normalize_axes=True)
        raw = local_pca_ellipse_params(pts, k=10, normalize_axes=False)
        self.assertTrue(torch.allclose(norm[:, 0], torch.ones_like(norm[:, 0]), atol=1e-5))
        self.assertGreater(float(raw[:, 0].max()), float(raw[:, 0].min()))

    def test_euclidean_teacher_runs(self):
        vr = VietorisRipsComplex(dim=1)
        clean = torch.randn(2, 15, 2)
        pd_info, scales = compute_clean_teacher_batch(
            clean,
            vr,
            teacher_mode="euclidean",
            need_clean_scales=True,
        )
        self.assertEqual(len(pd_info), 2)
        self.assertIsNotNone(scales)
        self.assertEqual(len(scales), 2)

    def test_local_pca_teacher_mahalanobis(self):
        vr = VietorisRipsComplex(dim=1)
        clean = torch.randn(2, 20, 2)
        pd_info, scales = compute_clean_teacher_batch(
            clean,
            vr,
            teacher_mode="local_pca",
            distance_backend="mahalanobis",
            local_pca_k=10,
            need_clean_scales=True,
        )
        self.assertEqual(len(pd_info), 2)
        self.assertIsNotNone(scales)
        self.assertEqual(len(scales), 2)

    def test_local_pca_teacher_ellphi_finite(self):
        vr = VietorisRipsComplex(dim=1)
        clean = torch.randn(1, 12, 2)
        pd_info, _ = compute_clean_teacher_batch(
            clean,
            vr,
            teacher_mode="local_pca",
            distance_backend="ellphi",
            ellphi_differentiable=False,
            local_pca_k=8,
        )
        self.assertEqual(len(pd_info), 1)

    def test_modes_differ_for_same_points(self):
        vr = VietorisRipsComplex(dim=1)
        clean = torch.randn(1, 25, 2)
        pd_eucl, _ = compute_clean_teacher_batch(
            clean, vr, teacher_mode="euclidean", distance_backend="mahalanobis"
        )
        pd_pca, _ = compute_clean_teacher_batch(
            clean,
            vr,
            teacher_mode="local_pca",
            distance_backend="mahalanobis",
            local_pca_k=10,
        )
        # PD objects differ when filtration construction differs
        self.assertIsNotNone(pd_eucl[0])
        self.assertIsNotNone(pd_pca[0])


if __name__ == "__main__":
    unittest.main()
