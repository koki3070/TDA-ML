"""Unit tests for ellphi_torch helper utilities."""

import unittest

import numpy as np

from tda_ml.ellphi_torch import _condensed_gradient_from_full


class TestEllphiTorchHelpers(unittest.TestCase):
    def test_condensed_gradient_matches_expected_order(self):
        g = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [10.0, 0.0, 12.0, 13.0],
                [20.0, 21.0, 0.0, 23.0],
                [30.0, 31.0, 32.0, 0.0],
            ],
            dtype=np.float64,
        )
        # scipy condensed order for n=4: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        expected = np.array(
            [
                1.0 + 10.0,
                2.0 + 20.0,
                3.0 + 30.0,
                12.0 + 21.0,
                13.0 + 31.0,
                23.0 + 32.0,
            ],
            dtype=np.float64,
        )
        actual = _condensed_gradient_from_full(g)
        self.assertTrue(np.allclose(actual, expected))

    def test_condensed_gradient_sum_symmetric_entries_n3(self):
        """L(D)=D[0,1]+D[1,0] with unit upstream grad on both off-diagonals."""
        g = np.zeros((3, 3), dtype=np.float64)
        g[0, 1] = 1.0
        g[1, 0] = 1.0
        actual = _condensed_gradient_from_full(g)
        np.testing.assert_allclose(actual, np.array([2.0, 0.0, 0.0]))

    def test_condensed_gradient_frobenius_half_sum_squares_n3(self):
        """L(D)=0.5*sum(D^2) for symmetric D from d=(2,3,5)."""
        from scipy.spatial.distance import squareform

        d = np.array([2.0, 3.0, 5.0], dtype=np.float64)
        dist = squareform(d)
        g = dist.copy()
        actual = _condensed_gradient_from_full(g)
        np.testing.assert_allclose(actual, np.array([4.0, 6.0, 10.0]))


if __name__ == "__main__":
    unittest.main()
