import unittest

import numpy as np

from tda_ml.persistence import compute_w_distance


class TestPersistenceWasserstein(unittest.TestCase):
    def _circle_points(self, n: int = 20) -> np.ndarray:
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([np.cos(theta), np.sin(theta)], axis=1)

    def test_empty_pred_returns_finite_wasserstein_not_sentinel(self):
        circle = self._circle_points()
        w = compute_w_distance([], circle)
        self.assertTrue(np.isfinite(w))
        self.assertLess(w, 10.0)

    def test_empty_gt_symmetric_with_empty_pred(self):
        circle = self._circle_points()
        w_pred_empty = compute_w_distance([], circle)
        w_gt_empty = compute_w_distance(circle, [])
        self.assertAlmostEqual(w_pred_empty, w_gt_empty, places=6)

    def test_both_empty_point_clouds(self):
        self.assertEqual(compute_w_distance([], []), 0.0)


if __name__ == "__main__":
    unittest.main()
