import unittest

import torch

from tda_ml.topology import compute_anisotropic_distance_matrix


class TestTopology(unittest.TestCase):
    def _make_batch(self):
        points = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float64,
        )
        params = torch.tensor(
            [[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]],
            dtype=torch.float64,
        )
        return points, params

    def test_zero_diagonal_same_point(self):
        points, params = self._make_batch()
        dist = compute_anisotropic_distance_matrix(points, params, symmetrize="max")
        n = points.shape[1]
        for i in range(n):
            self.assertAlmostEqual(dist[0, i, i].item(), 0.0, places=8)

    def test_symmetry_max_and_min(self):
        points, params = self._make_batch()
        for mode in ("max", "min"):
            with self.subTest(symmetrize=mode):
                dist = compute_anisotropic_distance_matrix(
                    points, params, symmetrize=mode
                )
                self.assertTrue(
                    torch.allclose(dist, dist.transpose(-1, -2), atol=1e-8)
                )

    def test_probs_changes_off_diagonal(self):
        points, params = self._make_batch()
        dist_none = compute_anisotropic_distance_matrix(points, params)
        probs = torch.tensor([[0.0, 0.5, 0.0]], dtype=torch.float64)
        dist_probs = compute_anisotropic_distance_matrix(
            points, params, probs=probs
        )
        mask = ~torch.eye(points.shape[1], dtype=torch.bool)
        self.assertFalse(
            torch.allclose(
                dist_none[0][mask],
                dist_probs[0][mask],
                atol=1e-8,
            )
        )
        self.assertAlmostEqual(dist_probs[0, 0, 0].item(), 0.0, places=8)

    def test_invalid_symmetrize_raises(self):
        points, params = self._make_batch()
        with self.assertRaises(ValueError):
            compute_anisotropic_distance_matrix(
                points, params, symmetrize="mean"
            )

    def test_accepts_full_model_params_dim5(self):
        points, params3 = self._make_batch()
        # Full model layout: [dx, dy, a, b, theta]
        offsets = torch.zeros((points.shape[0], points.shape[1], 2), dtype=points.dtype)
        params5 = torch.cat([offsets, params3], dim=-1)
        dist3 = compute_anisotropic_distance_matrix(points, params3, symmetrize="max")
        dist5 = compute_anisotropic_distance_matrix(points, params5, symmetrize="max")
        self.assertTrue(torch.allclose(dist3, dist5, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
