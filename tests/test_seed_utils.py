"""Tests for global seed initialization utility."""

import random
import unittest

import numpy as np
import torch

from tda_ml.seed_utils import set_global_seed


class TestSeedUtils(unittest.TestCase):
    def test_set_global_seed_makes_generators_reproducible(self):
        set_global_seed(123)
        py_1 = random.random()
        np_1 = np.random.rand()
        torch_1 = torch.rand(3)

        set_global_seed(123)
        py_2 = random.random()
        np_2 = np.random.rand()
        torch_2 = torch.rand(3)

        self.assertEqual(py_1, py_2)
        self.assertEqual(np_1, np_2)
        self.assertTrue(torch.equal(torch_1, torch_2))


if __name__ == "__main__":
    unittest.main()
