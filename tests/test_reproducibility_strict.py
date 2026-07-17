"""Strict reproducibility defaults (hard-fail on degenerate cases)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tda_ml.dbscan_eval import PreparedCloud, grid_search_prepared
from tda_ml.reproducibility import (
    assert_loss_config_explicit,
    dbscan_grid_from_config,
    reproducibility_settings,
    resolve_dbscan_grid,
)
from tda_ml.topo_wdist import TopoWdistOptions


class TestReproducibilityConfig(unittest.TestCase):
    def test_base_config_has_explicit_dbscan_grid(self):
        from tda_ml.config import load_config

        cfg = load_config("dev")
        eps, ms = dbscan_grid_from_config(cfg)
        self.assertEqual(len(eps), 15)
        self.assertEqual(ms, [3, 5, 7, 10, 15])

    def test_base_config_has_baseline_grids(self):
        from tda_ml.config import load_config
        from tda_ml.reproducibility import baseline_grids_from_config

        cfg = load_config("dev")
        grids = baseline_grids_from_config(cfg)
        self.assertIn("contamination_values", grids)
        self.assertIn("lof_n_neighbors", grids)

    def test_missing_dbscan_grid_raises(self):
        with self.assertRaises(ValueError):
            dbscan_grid_from_config({})

    def test_strict_defaults(self):
        settings = reproducibility_settings({})
        self.assertTrue(settings["strict_topo_samples"])
        self.assertFalse(settings["allow_nan_batch_skip"])
        self.assertFalse(settings["allow_empty_cloud_fallback"])
        self.assertFalse(settings["allow_legacy_loss_keys"])

    def test_resolve_dbscan_grid_explicit_override(self):
        from tda_ml.config import load_config

        cfg = load_config("dev")
        eps, ms = resolve_dbscan_grid(cfg, eps_values=[0.2], min_samples_values=[3])
        self.assertEqual(eps, [0.2])
        self.assertEqual(ms, [3])

    def test_assert_loss_config_requires_explicit_weights(self):
        with self.assertRaises(ValueError):
            assert_loss_config_explicit({"loss": {"w_topo": 0.1}, "training": {"lr": 1e-4}})

    def test_classify_tune_objective(self):
        from tda_ml.preflight import classify_tune_objective

        self.assertEqual(
            classify_tune_objective("val_topo_wdist_min_at_val_topo_best_ckpt"),
            "wdist",
        )
        self.assertEqual(classify_tune_objective("val_dbscan_mcc_max"), "mcc")
        self.assertEqual(
            classify_tune_objective(
                "val_dbscan_mcc_max_at_val_topo_best_ckpt"
            ),
            "mcc",
        )
        self.assertEqual(
            classify_tune_objective(
                "val_dbscan_mcc_max",
                objective_kind="mcc",
            ),
            "mcc",
        )
        self.assertEqual(
            classify_tune_objective("val_topo_wdist_min"),
            "wdist",
        )
        with self.assertRaises(ValueError):
            classify_tune_objective("custom_objective_with_wdist_substring")
        with self.assertRaises(ValueError):
            classify_tune_objective("legacy_name", objective_kind="mcc")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            classify_tune_objective(
                "val_topo_wdist_min",
                objective_kind="mcc",
            )

    def test_selection_default_is_val_topo(self):
        from tda_ml.model_selection import selection_settings_from_config

        settings = selection_settings_from_config({})
        self.assertEqual(settings.metric, "val_topo")


class TestPreflightTuneJson(unittest.TestCase):
    def test_wdist_json_passes_without_mcc_expectation(self):
        from tda_ml.preflight import preflight_tune_json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.json"
            path.write_text(
                '{"objective":"val_topo_wdist_min","best_params":{"w_topo":0.1,'
                '"w_aniso":0.1,"w_size":0.1,"lr":1e-4}}\n',
                encoding="utf-8",
            )
            payload = preflight_tune_json(path)
            self.assertEqual(payload["_objective_kind"], "wdist")

    def test_legacy_tune_json_without_paper_contract_raises(self):
        from tda_ml.preflight import (
            PAPER_NO_CLS_CONTRACT,
            preflight_tune_json,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.json"
            path.write_text(
                '{"objective":"val_topo_wdist_min","objective_kind":"wdist",'
                '"best_params":{"w_topo":0.1,"w_aniso":0.1,"w_size":0.1,"lr":1e-4}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Re-tune with the H1-only stack"):
                preflight_tune_json(
                    path,
                    expected_contract=PAPER_NO_CLS_CONTRACT,
                )

    def test_tune_json_paper_contract_must_match(self):
        import json

        from tda_ml.preflight import (
            PAPER_NO_CLS_CONTRACT,
            preflight_tune_json,
        )

        payload = {
            "objective": "val_dbscan_mcc_max_at_val_topo_best_ckpt",
            "objective_kind": "mcc",
            "best_params": {
                "w_topo": 0.1,
                "w_aniso": 0.1,
                "w_size": 0.1,
                "lr": 1e-4,
            },
            **PAPER_NO_CLS_CONTRACT,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.json"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            loaded = preflight_tune_json(
                path,
                expected_contract=PAPER_NO_CLS_CONTRACT,
            )
            self.assertEqual(loaded["_objective_kind"], "mcc")

    def test_objective_kind_must_agree_with_whitelist_name(self):
        from tda_ml.preflight import preflight_tune_json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.json"
            path.write_text(
                '{"objective":"custom_objective","objective_kind":"wdist",'
                '"best_params":{"w_topo":0.1,"w_aniso":0.1,"w_size":0.1,"lr":1e-4}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unrecognized tune objective"):
                preflight_tune_json(path)


class TestPaperNoClsContract(unittest.TestCase):
    @staticmethod
    def _valid_config() -> dict:
        return {
            "model": {
                "topology_loss": {
                    "homology_dimensions": [1],
                    "prob_weighting": False,
                    "distance_backend": "ellphi",
                }
            },
            "loss": {
                "teacher_mode": "local_pca",
                "aniso_mode": "elongate",
                "size_mode": "power",
                "w_class": 0.0,
                "teacher_local_pca_k": 10,
                "teacher_local_pca_normalize_axes": True,
            },
        }

    def test_explicit_contract_passes(self):
        from tda_ml.preflight import (
            PAPER_NO_CLS_CONTRACT,
            assert_paper_no_cls_contract,
        )

        self.assertEqual(
            assert_paper_no_cls_contract(self._valid_config()),
            PAPER_NO_CLS_CONTRACT,
        )

    def test_missing_homology_dimensions_raises(self):
        from tda_ml.preflight import assert_paper_no_cls_contract

        config = self._valid_config()
        del config["model"]["topology_loss"]["homology_dimensions"]
        with self.assertRaisesRegex(ValueError, "homology_dimensions"):
            assert_paper_no_cls_contract(config)

    def test_wrong_teacher_mode_raises(self):
        from tda_ml.preflight import assert_paper_no_cls_contract

        config = self._valid_config()
        config["loss"]["teacher_mode"] = "euclidean"
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            assert_paper_no_cls_contract(config)


class TestResolveValTopoCheckpoint(unittest.TestCase):
    def test_missing_best_model_raises(self):
        from tda_ml.checkpoint_io import resolve_val_topo_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                resolve_val_topo_checkpoint(Path(tmp))


class TestDbscanGridStrict(unittest.TestCase):
    def _toy_cloud(self) -> PreparedCloud:
        n = 12
        import numpy as np

        rng = np.random.default_rng(0)
        points = rng.normal(size=(n, 2))
        params = np.column_stack([np.full(n, 0.3), np.full(n, 0.2), np.zeros(n)])
        labels_gt = np.array([0] * 10 + [1] * 2, dtype=np.int64)
        clean_pc = np.vstack([points[:10], np.zeros((90, 2))])
        dist = np.abs(points[:, None, :] - points[None, :, :]).sum(axis=-1)
        np.fill_diagonal(dist, 0.0)
        return PreparedCloud(points, params, labels_gt, clean_pc, dist)

    def test_failed_cell_raises_when_skip_not_allowed(self):
        cloud = self._toy_cloud()
        opts = TopoWdistOptions(
            teacher_mode="euclidean",
            distance_backend="mahalanobis",
            homology_dimensions=(0, 1),
            prob_weighting=False,
        )
        with patch(
            "tda_ml.dbscan_eval._dbscan_classification_metrics",
            side_effect=RuntimeError("degenerate"),
        ):
            with self.assertRaises(RuntimeError):
                grid_search_prepared(
                    [cloud],
                    eps_values=[0.2],
                    min_samples_values=[3],
                    objective="mcc",
                    topo_options=opts,
                    allow_skip_degenerate_grid_cells=False,
                )


class TestTopologicalLossStrict(unittest.TestCase):
    def test_nan_wasserstein_raises_when_strict(self):
        from tda_ml.losses import TopologicalLoss
        from torch_topological.nn import VietorisRipsComplex

        torch.manual_seed(0)
        b, n = 1, 6
        pts = torch.randn(b, n, 2)
        par = torch.randn(b, n, 3)
        par[:, :, 0:2] = par[:, :, 0:2].abs() + 0.2
        logits = torch.zeros(b, n, 1)
        vr = VietorisRipsComplex(dim=1)
        clean = [vr(pts[0])]

        loss_fn = TopologicalLoss(
            weight=1.0,
            distance_backend="mahalanobis",
            prob_weighting=False,
            homology_dimensions=[0, 1],
            strict_topo_samples=True,
        )

        class BrokenWasserstein(torch.nn.Module):
            def forward(self, a, b):
                return torch.tensor(float("nan"), device=pts.device)

        loss_fn.wasserstein = BrokenWasserstein()
        with self.assertRaises(RuntimeError):
            loss_fn(pts, par, logits, clean)


if __name__ == "__main__":
    unittest.main()
