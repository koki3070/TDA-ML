"""Tests for global seed initialization utility."""

import random
import unittest

import numpy as np
import torch

from tda_ml.seed_utils import (
    cliff_init_model_seeds,
    resolve_model_seed,
    set_global_seed,
)


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

    def test_resolve_model_seed_defaults_to_data_seed(self):
        self.assertEqual(resolve_model_seed({"data": {"seed": 42}}), 42)
        self.assertEqual(
            resolve_model_seed(
                {"data": {"seed": 42}, "training": {"model_seed": None}}
            ),
            42,
        )
        self.assertEqual(
            resolve_model_seed(
                {"data": {"seed": 42}, "training": {"model_seed": 999}}
            ),
            999,
        )

    def test_cliff_init_model_seeds_schedule(self):
        seeds = cliff_init_model_seeds(123, n_restarts=3)
        # order: offset 0, then 7 (clipped out), then 1 → [0,1,2] for n=3
        self.assertEqual(seeds, [123, 123 + 100_003, 123 + 2 * 100_003])
        seeds8 = cliff_init_model_seeds(123, n_restarts=8)
        self.assertEqual(seeds8[0], 123)
        self.assertEqual(seeds8[1], 123 + 7 * 100_003)  # probe-validated early
        self.assertEqual(len(set(seeds8)), 8)
        with self.assertRaises(ValueError):
            cliff_init_model_seeds(1, n_restarts=0)


if __name__ == "__main__":
    unittest.main()
