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

CONTRACT_HOMOLOGY = [0, 1]
TUNE_RINGS = "tune_rings"
PAPER_RINGS = "paper_rings"


class TestTuneTrialConfig(unittest.TestCase):
    def test_paper_declares_topology_contract(self):
        from tda_ml.config import load_config

        cfg = load_config(PAPER_RINGS)
        self.assertEqual(cfg["model"]["topology_loss"]["homology_dimensions"], CONTRACT_HOMOLOGY)
        self.assertFalse(cfg["model"]["topology_loss"]["prob_weighting"])
        self.assertEqual(cfg["loss"]["teacher_mode"], "local_pca")
        self.assertEqual(cfg["model"]["topology_loss"]["distance_backend"], "ellphi")
        self.assertEqual(cfg["data"]["max_points"], 100)
        self.assertEqual(cfg["data"]["num_outliers"], 20)
        self.assertEqual(cfg["data"]["dataset_type"], "thin_rings")
        self.assertEqual(cfg["data"]["outlier_mode"], "ring_radial")
        self.assertEqual(cfg["loss"]["teacher_local_pca_major_scale"], 0.083)
        self.assertTrue(cfg["training"]["require_val_topo_cliff"])

    def test_canonical_power_tune_base(self):
        cfg = build_wdist_trial(
            TUNE_RINGS,
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
        self.assertEqual(cfg["model"]["topology_loss"]["homology_dimensions"], CONTRACT_HOMOLOGY)
        self.assertEqual(cfg["loss"]["size_mode"], "power")
        self.assertEqual(cfg["training"]["selection"]["metric"], "val_topo")
        self.assertEqual(cfg["model"]["topology_loss"]["distance_backend"], "ellphi")
        self.assertFalse(cfg["model"]["topology_loss"]["prob_weighting"])
        self.assertEqual(cfg["data"]["outlier_mode"], "ring_radial")

    def test_power_tune_preserves_contract_hard_fail_stack(self):
        cfg = build_wdist_trial(
            TUNE_RINGS,
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
        self.assertEqual(cfg["model"]["topology_loss"]["homology_dimensions"], CONTRACT_HOMOLOGY)
        self.assertEqual(cfg["loss"]["aniso_mode"], "elongate_barrier")
        self.assertEqual(cfg["loss"]["aniso_barrier_threshold"], 6.0)
        self.assertEqual(cfg["loss"]["size_mode"], "power")
        self.assertEqual(cfg["model"]["topology_loss"]["distance_backend"], "ellphi")

    def test_wdist_builder_rejects_divergent_yaml_homology_and_aniso(self):
        from copy import deepcopy

        from tda_ml.config import load_config

        divergent = load_config(
            TUNE_RINGS,
            project_root=REPO,
        )
        # Divergent from rings contract homology [0,1].
        divergent["model"]["topology_loss"]["homology_dimensions"] = [1]
        divergent["loss"]["aniso_mode"] = "linear"
        divergent["loss"].pop("aniso_barrier_threshold", None)

        with mock.patch(
            "tune_wdist.load_config",
            side_effect=lambda *a, **k: deepcopy(divergent),
        ):
            # Do not silently restamp homology or aniso onto a divergent YAML.
            with self.assertRaisesRegex(ValueError, "Rings no_cls"):
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

        # Plain elongate is a MNIST paper ablation, not the rings contract.
        divergent["loss"]["aniso_mode"] = "elongate"
        with mock.patch(
            "tune_wdist.load_config",
            side_effect=lambda *a, **k: deepcopy(divergent),
        ):
            with self.assertRaisesRegex(ValueError, "Rings no_cls"):
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

    def test_wdist_builder_mirrors_barrier_from_canonical_yaml(self):
        cfg = build_wdist_trial(
            TUNE_RINGS,
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
        self.assertEqual(cfg["model"]["topology_loss"]["homology_dimensions"], CONTRACT_HOMOLOGY)

    def test_mcc_builder_forces_contract_homology(self):
        cfg = build_mcc_trial(
            TUNE_RINGS,
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
        self.assertEqual(cfg["model"]["topology_loss"]["homology_dimensions"], CONTRACT_HOMOLOGY)
        self.assertEqual(cfg["loss"]["aniso_mode"], "elongate_barrier")
        self.assertEqual(cfg["loss"]["aniso_barrier_threshold"], 6.0)


if __name__ == "__main__":
    unittest.main()
