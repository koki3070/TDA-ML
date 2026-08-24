"""Tests for Optuna tune trial config (local_pca teacher)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from tune_mcc import build_trial_config as build_mcc_trial  # noqa: E402
from tune_wdist import build_trial_config as build_wdist_trial  # noqa: E402


class TestTuneTrialConfig(unittest.TestCase):
    def test_paper_declares_topology_contract(self):
        from tda_ml.config import load_config

        cfg = load_config("paper_n100_o20_nocls_h1_ellphi_lpca_power")
        self.assertEqual(cfg["model"]["topology_loss"]["homology_dimensions"], [1])
        self.assertFalse(cfg["model"]["topology_loss"]["prob_weighting"])
        self.assertEqual(cfg["loss"]["teacher_mode"], "local_pca")
        self.assertEqual(cfg["model"]["topology_loss"]["distance_backend"], "ellphi")
        self.assertEqual(cfg["data"]["max_points"], 100)
        self.assertEqual(cfg["data"]["num_outliers"], 20)
        self.assertEqual(cfg["data"]["dataset_type"], "mnist")
        self.assertEqual(cfg["data"]["outlier_mode"], "uniform")

    def test_canonical_power_tune_base(self):
        cfg = build_wdist_trial(
            "tune_n100_o20_nocls_h1_ellphi_lpca_power",
            w_aniso=0.1,
            w_size=0.2,
            w_topo=0.3,
            lr=1e-4,
            backend="ellphi",
            tune_epochs=20,
            out_base="outputs/test_tune",
            trial_number=0,
            size_mode="power",
        )
        self.assertEqual(cfg["loss"]["w_class"], 0.0)
        self.assertEqual(cfg["loss"]["teacher_mode"], "local_pca")
        self.assertEqual(cfg["model"]["topology_loss"]["homology_dimensions"], [1])
        self.assertEqual(cfg["loss"]["size_mode"], "power")
        self.assertEqual(cfg["training"]["selection"]["metric"], "val_topo")
        self.assertEqual(cfg["model"]["topology_loss"]["distance_backend"], "ellphi")
        self.assertFalse(cfg["model"]["topology_loss"]["prob_weighting"])

    def test_power_tune_preserves_h1_hard_fail_stack(self):
        cfg = build_wdist_trial(
            "tune_n100_o20_nocls_h1_ellphi_lpca_power",
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
        self.assertEqual(cfg["model"]["topology_loss"]["distance_backend"], "ellphi")

    def test_wdist_builder_overrides_divergent_yaml_homology_and_aniso(self):
        from copy import deepcopy

        from tda_ml.config import load_config

        divergent = load_config(
            "tune_n100_o20_nocls_h1_ellphi_lpca_power",
            project_root=REPO,
        )
        divergent["model"]["topology_loss"]["homology_dimensions"] = [0, 1]
        divergent["loss"]["aniso_mode"] = "linear"

        with mock.patch(
            "tune_wdist.load_config",
            side_effect=lambda *a, **k: deepcopy(divergent),
        ):
            # aniso_mode is now mirrored from the base config declaration, so a
            # non-paper variant must hard-fail instead of being silently forced.
            with self.assertRaisesRegex(ValueError, "not a declared paper variant"):
                build_wdist_trial(
                    "ignored",
                    w_aniso=0.1,
                    w_size=0.2,
                    w_topo=0.3,
                    lr=1e-4,
                    backend="ellphi",
                    tune_epochs=5,
                    out_base="outputs/x",
                    trial_number=1,
                    size_mode="power",
                )

        divergent["loss"]["aniso_mode"] = "elongate"
        with mock.patch(
            "tune_wdist.load_config",
            side_effect=lambda *a, **k: deepcopy(divergent),
        ):
            cfg = build_wdist_trial(
                "ignored",
                w_aniso=0.1,
                w_size=0.2,
                w_topo=0.3,
                lr=1e-4,
                backend="ellphi",
                tune_epochs=5,
                out_base="outputs/x",
                trial_number=1,
                size_mode="power",
            )
        self.assertEqual(cfg["model"]["topology_loss"]["homology_dimensions"], [1])
        self.assertEqual(cfg["loss"]["aniso_mode"], "elongate")

    def test_wdist_builder_mirrors_barrier_variant_from_base_config(self):
        cfg = build_wdist_trial(
            "methods_n100_o20_nocls_h1_ellphi_lpca_power_neartangent_barrier",
            w_aniso=0.1,
            w_size=0.2,
            w_topo=0.3,
            lr=1e-4,
            backend="ellphi",
            tune_epochs=5,
            out_base="outputs/x",
            trial_number=3,
            size_mode="power",
        )
        self.assertEqual(cfg["loss"]["aniso_mode"], "elongate_barrier")
        self.assertEqual(cfg["loss"]["aniso_barrier_threshold"], 6.0)
        self.assertEqual(cfg["model"]["topology_loss"]["homology_dimensions"], [1])

    def test_mcc_builder_forces_homology_h1(self):
        cfg = build_mcc_trial(
            "tune_n100_o20_nocls_h1_ellphi_lpca_power",
            w_aniso=0.1,
            w_size=0.2,
            w_topo=0.3,
            lr=1e-4,
            backend="ellphi",
            tune_epochs=5,
            out_base="outputs/x",
            trial_number=2,
            size_mode="power",
            size_ref=1.34,
            size_power=1.5,
        )
        self.assertEqual(cfg["model"]["topology_loss"]["homology_dimensions"], [1])
        self.assertEqual(cfg["loss"]["aniso_mode"], "elongate")


if __name__ == "__main__":
    unittest.main()
