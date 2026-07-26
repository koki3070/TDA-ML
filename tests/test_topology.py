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

    def test_diagonal_is_exactly_zero(self):
        points, params = self._make_batch()
        dist = compute_anisotropic_distance_matrix(points, params, symmetrize="max")
        n = points.shape[1]
        for i in range(n):
            self.assertEqual(dist[0, i, i].item(), 0.0)

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
        self.assertEqual(dist_probs[0, 0, 0].item(), 0.0)

    def test_invalid_symmetrize_raises(self):
        points, params = self._make_batch()
        with self.assertRaises(ValueError):
            compute_anisotropic_distance_matrix(
                points, params, symmetrize="mean"
            )


if __name__ == "__main__":
    unittest.main()
