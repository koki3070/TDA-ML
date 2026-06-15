"""Regression tests for ``tda_ml.torch_util.maybe_backward`` (used by the TD distance bench).

The benchmark driver is ``scripts/bench_td_distance_mahalanobis_vs_ellphi.py``; these tests
pin the small autograd guard without invoking the full benchmark.
"""

import unittest

import torch

from tda_ml.torch_util import maybe_backward


class TestBenchTdDistanceMahalanobisVsEllphi(unittest.TestCase):
    def test_maybe_backward_runs_for_grad_tensor(self) -> None:
        x = torch.tensor([2.0], requires_grad=True)
        loss = (x * x).sum()

        ran_backward = maybe_backward(loss)

        self.assertTrue(ran_backward)
        self.assertIsNotNone(x.grad)
        self.assertAlmostEqual(x.grad.item(), 4.0)

    def test_maybe_backward_skips_for_no_grad_tensor(self) -> None:
        loss = torch.tensor(3.0)

        ran_backward = maybe_backward(loss)

        self.assertFalse(ran_backward)


if __name__ == "__main__":
    unittest.main()
