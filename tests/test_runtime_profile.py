"""Tests for runtime environment profile generation."""

import unittest

import torch

from tda_ml.runtime_profile import build_runtime_profile


class TestRuntimeProfile(unittest.TestCase):
    def test_build_runtime_profile_contains_fixed_fields(self):
        config = {
            "device": "cpu",
            "performance": {
                "enable_tf32": False,
                "cudnn_benchmark": False,
                "matmul_precision": "medium",
                "use_amp": False,
                "amp_dtype": "bfloat16",
            },
            "reproducibility": {"deterministic_algorithms": True},
        }
        profile = build_runtime_profile(
            config=config,
            device=torch.device("cpu"),
            num_workers=2,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=4,
            use_amp_effective=False,
            amp_dtype_effective="bfloat16",
        )

        self.assertEqual(profile["device_requested"], "cpu")
        self.assertEqual(profile["device_effective"], "cpu")
        self.assertEqual(profile["num_workers"], 2)
        self.assertFalse(profile["enable_tf32_config"])
        self.assertFalse(profile["cudnn_benchmark_config"])
        self.assertEqual(profile["amp_dtype_effective"], "bfloat16")
        self.assertTrue(profile["deterministic_algorithms"])


if __name__ == "__main__":
    unittest.main()
