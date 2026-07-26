import unittest

import numpy as np
import torch

from tda_ml.geometry import (
    ellipse_params_to_centers_cov_numpy,
    ellipse_params_to_centers_cov_torch,
)


class TestGeometry(unittest.TestCase):
    def test_numpy_torch_centers_cov_consistency(self):
        points = np.array([[0.0, 0.0], [1.0, -1.0]], dtype=np.float64)
        params = np.array(
            [
                [1.5, 0.5, 0.3],
                [0.7, 0.4, -0.8],
            ],
            dtype=np.float64,
        )

        c_np, cov_np = ellipse_params_to_centers_cov_numpy(points, params)
        c_t, cov_t = ellipse_params_to_centers_cov_torch(
            torch.tensor(points, dtype=torch.float64),
            torch.tensor(params, dtype=torch.float64),
        )

        self.assertTrue(np.allclose(c_np, c_t.numpy(), atol=1e-8))
        self.assertTrue(np.allclose(cov_np, cov_t.numpy(), atol=1e-8))

    def test_zero_axes_covariance_is_finite_and_symmetric(self):
        """半径0の退化楕円でも計算が破綻せず、共分散は対称を保つ。"""
        points = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64)
        params = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 4]],
            dtype=np.float64,
        )
        _, cov_np = ellipse_params_to_centers_cov_numpy(points, params)
        self.assertTrue(np.isfinite(cov_np).all())
        self.assertTrue(np.allclose(cov_np[:, 0, 1], cov_np[:, 1, 0], atol=1e-12))

    def test_extreme_axes_stays_finite(self):
        """極小/極大軸長を混在させても有限値の共分散を返す。"""
        points = np.array([[1e-9, -1e-9], [1e3, -1e3]], dtype=np.float64)
        params = np.array(
            [[1e-9, 1e9, 0.1], [1e8, 2e-8, -0.7]],
            dtype=np.float64,
        )
        _, cov_np = ellipse_params_to_centers_cov_numpy(points, params)
        _, cov_t = ellipse_params_to_centers_cov_torch(
            torch.tensor(points, dtype=torch.float64),
            torch.tensor(params, dtype=torch.float64),
        )
        self.assertTrue(np.isfinite(cov_np).all())
        self.assertTrue(torch.isfinite(cov_t).all())

    def test_nan_inf_inputs_propagate_non_finite_values(self):
        """NaN/Inf 入力時は非有限値が伝播する（現行仕様の破綻シグナル）。"""
        points = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        params = np.array(
            [[np.nan, 1.0, 0.0], [np.inf, 1.0, 0.0]],
            dtype=np.float64,
        )
        _, cov_np = ellipse_params_to_centers_cov_numpy(points, params)
        _, cov_t = ellipse_params_to_centers_cov_torch(
            torch.tensor(points, dtype=torch.float64),
            torch.tensor(params, dtype=torch.float64),
        )
        self.assertFalse(np.isfinite(cov_np).all())
        self.assertFalse(torch.isfinite(cov_t).all())


if __name__ == "__main__":
    unittest.main()
