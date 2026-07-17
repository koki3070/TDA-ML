"""Tests for Optuna tune trial config (local_pca teacher)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from tune_elongate_wdist import build_trial_config  # noqa: E402


class TestTuneTrialConfig(unittest.TestCase):
    def test_local_pca_tune_base(self):
        cfg = build_trial_config(
            "elongate_n100_no_cls_tune_local_pca",
            w_aniso=0.1,
            w_size=0.2,
            w_topo=0.3,
            lr=1e-4,
            backend="ellphi",
            tune_epochs=20,
            out_base="outputs/test_tune",
            trial_number=0,
        )
        self.assertEqual(cfg["loss"]["w_class"], 0.0)
        self.assertEqual(cfg["loss"]["teacher_mode"], "local_pca")
        self.assertEqual(cfg["training"]["selection"]["metric"], "val_topo")
        self.assertEqual(cfg["model"]["topology_loss"]["distance_backend"], "ellphi")
        self.assertFalse(cfg["model"]["topology_loss"]["prob_weighting"])

    def test_power_tune_preserves_h1_hard_fail_stack(self):
        cfg = build_trial_config(
            "elongate_n100_no_cls_tune_local_pca_ellphi_power_mcc",
            w_aniso=0.1,
            w_size=0.2,
            w_topo=0.3,
            lr=1e-4,
            backend="ellphi",
            tune_epochs=20,
            out_base="outputs/test_tune_power",
            trial_number=0,
            size_mode="power",
        )
        self.assertEqual(cfg["model"]["topology_loss"]["homology_dimensions"], [1])
        self.assertEqual(cfg["loss"]["aniso_mode"], "elongate")
        self.assertEqual(cfg["loss"]["size_mode"], "power")


if __name__ == "__main__":
    unittest.main()
