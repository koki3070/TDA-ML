"""DBSCAN grid should compute topo W-Dist once per cloud, not per (eps, min_samples)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from tda_ml.dbscan_eval import PreparedCloud, grid_search_prepared
from tda_ml.topo_wdist import TopoWdistOptions


def _toy_cloud() -> PreparedCloud:
    n = 12
    rng = np.random.default_rng(0)
    points = rng.normal(size=(n, 2))
    params = np.column_stack([np.full(n, 0.3), np.full(n, 0.2), np.zeros(n)])
    labels_gt = np.array([0] * 10 + [1] * 2, dtype=np.int64)
    clean_pc = np.vstack([points[:10], np.zeros((90, 2))])
    dist = np.abs(points[:, None, :] - points[None, :, :]).sum(axis=-1)
    np.fill_diagonal(dist, 0.0)
    return PreparedCloud(points, params, labels_gt, clean_pc, dist)


class TestDbscanGridWdistCache(unittest.TestCase):
    def test_topo_wdist_computed_once_per_cloud(self) -> None:
        clouds = [_toy_cloud(), _toy_cloud()]
        opts = TopoWdistOptions(teacher_mode="euclidean", distance_backend="mahalanobis")
        with patch(
            "tda_ml.dbscan_eval.compute_topo_wdist",
            side_effect=[0.5, 0.6],
        ) as mock_wdist:
            grid_search_prepared(
                clouds,
                eps_values=[0.2, 0.4],
                min_samples_values=[3, 5],
                objective="mcc",
                topo_options=opts,
            )
        self.assertEqual(mock_wdist.call_count, 2)

    def test_wdist_constant_across_grid_for_same_cloud(self) -> None:
        cloud = _toy_cloud()
        opts = TopoWdistOptions(teacher_mode="euclidean", distance_backend="mahalanobis")
        with patch(
            "tda_ml.dbscan_eval.compute_topo_wdist",
            return_value=0.42,
        ):
            result = grid_search_prepared(
                [cloud],
                eps_values=[0.2, 0.4],
                min_samples_values=[3],
                objective="mcc",
                topo_options=opts,
            )
        self.assertAlmostEqual(result.wdist, 0.42)


if __name__ == "__main__":
    unittest.main()
