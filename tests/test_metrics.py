import unittest

import numpy as np

from tda_ml.metrics import (
    compute_recall_specificity_gmean_mcc,
    compute_recall_specificity_gmean_mcc_wdist,
)
from tda_ml.persistence import compute_w_distance


class TestMetrics(unittest.TestCase):
    def test_compute_recall_specificity_gmean_mcc(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 1, 0, 1, 0, 1]

        recall, specificity, gmean, mcc = compute_recall_specificity_gmean_mcc(
            y_true, y_pred
        )

        self.assertAlmostEqual(recall, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(specificity, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(gmean, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(mcc, 1.0 / 3.0, places=6)

    def test_compute_metrics_all_inlier_zero_division_path(self):
        """positive class 不在時は recall=0 を返し、ゼロ除算を起こさない。"""
        y_true = np.array([0, 0, 0, 0], dtype=int)
        y_pred = np.array([0, 0, 0, 0], dtype=int)

        recall, specificity, gmean, mcc = compute_recall_specificity_gmean_mcc(
            y_true, y_pred
        )
        self.assertEqual(recall, 0.0)
        self.assertEqual(specificity, 1.0)
        self.assertEqual(gmean, 0.0)
        self.assertEqual(mcc, 0.0)

    def test_compute_metrics_nan_inf_labels_raise_value_error(self):
        """NaN/Inf を含むラベルは sklearn 側で ValueError として扱う。"""
        y_pred = np.array([0, 1, 0, 1], dtype=int)
        with self.assertRaises(ValueError):
            compute_recall_specificity_gmean_mcc(np.array([0, 1, np.nan, 0]), y_pred)
        with self.assertRaises(ValueError):
            compute_recall_specificity_gmean_mcc(np.array([0, 1, np.inf, 0]), y_pred)

    def test_compute_metrics_with_wdist_empty_pred_inliers(self):
        y_true = np.array([0, 0, 1, 1], dtype=int)
        y_pred = np.array([1, 1, 1, 1], dtype=int)
        points = np.array([[0.0, 0.0], [0.5, 0.2], [1.0, 1.0], [-1.0, -1.0]])
        gt_inliers = self._circle_gt_inliers()

        _, _, _, _, wdist = compute_recall_specificity_gmean_mcc_wdist(
            y_true, y_pred, points=points, gt_inliers=gt_inliers
        )
        expected = compute_w_distance(np.empty((0, 2)), gt_inliers)
        self.assertAlmostEqual(wdist, expected, places=6)
        self.assertTrue(np.isfinite(wdist))

    def test_wdist_missing_inputs_hard_fail(self):
        y_true = np.array([0, 1], dtype=int)
        y_pred = np.array([0, 1], dtype=int)
        with self.assertRaises(ValueError):
            compute_recall_specificity_gmean_mcc_wdist(y_true, y_pred)

    @staticmethod
    def _circle_gt_inliers(n: int = 12) -> np.ndarray:
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([np.cos(theta), np.sin(theta)], axis=1)

    def test_compute_metrics_wdist_extreme_scale_inputs(self):
        """極小/極大スケールの点群でも指標計算が有限値で返る。"""
        y_true = np.array([0, 0, 1, 1], dtype=int)
        y_pred = np.array([0, 1, 0, 1], dtype=int)
        points = np.array(
            [[1e-12, -1e-12], [2e6, -3e6], [1e6, 1e6], [-1e6, 1.5e6]],
            dtype=np.float64,
        )
        gt_inliers = np.array([[1e-12, -1e-12], [2e6, -3e6]], dtype=np.float64)

        recall, specificity, gmean, mcc, wdist = compute_recall_specificity_gmean_mcc_wdist(
            y_true, y_pred, points=points, gt_inliers=gt_inliers
        )
        for value in (recall, specificity, gmean, mcc, wdist):
            self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
