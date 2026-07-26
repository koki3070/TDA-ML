"""Regression tests for paper multiseed / aggregate / production helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from aggregate_paper_multiseed import discover_seed_metrics  # noqa: E402
from paper_run_freshness import inspect_seed_metrics  # noqa: E402
from run_paper_30ep import (  # noqa: E402
    TAG_MCC,
    TAG_WDIST,
    infer_tag,
    load_tune_weights,
)
from tda_ml.preflight import PAPER_NO_CLS_CONTRACT, preflight_training_config  # noqa: E402


class TestAggregateDiscover(unittest.TestCase):
    def test_duplicate_seed_hard_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, mcc in enumerate([0.1, 0.9]):
                logs = root / f"paper_s42_run{i}" / "logs"
                logs.mkdir(parents=True)
                (logs / "run_manifest.json").write_text(
                    json.dumps({"seed": 42, "tune_json": "/x.json"}),
                    encoding="utf-8",
                )
                (logs / "paper_metrics_test_foo.json").write_text(
                    json.dumps(
                        {"mcc": mcc, "gmean": 0.5, "wdist": 1.0, "recall": 0.5}
                    ),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(ValueError, "Duplicate metrics"):
                discover_seed_metrics(root, "**/paper_metrics_test_*.json")

    def test_missing_manifest_seed_hard_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "paper_s42_x" / "logs"
            logs.mkdir(parents=True)
            (logs / "paper_metrics_test_foo.json").write_text(
                json.dumps({"mcc": 0.1, "gmean": 0.5, "wdist": 1.0}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Missing run_manifest"):
                discover_seed_metrics(root, "**/paper_metrics_test_*.json")


class TestLoadTuneWeights(unittest.TestCase):
    def test_size_params_propagated_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.json"
            path.write_text(
                json.dumps(
                    {
                        "best_params": {
                            "w_topo": 1.0,
                            "w_aniso": 2.0,
                            "w_size": 3.0,
                            "lr": 1e-3,
                            "size_ref": 9.9,
                            "size_power": 0.25,
                        }
                    }
                ),
                encoding="utf-8",
            )
            weights = load_tune_weights(
                path, size_ref_default=1.34, size_power_default=1.5
            )
            self.assertEqual(weights["size_ref"], 9.9)
            self.assertEqual(weights["size_power"], 0.25)
            self.assertEqual(weights["size_from_tune"], 1.0)


class TestInferTag(unittest.TestCase):
    def test_explicit_tag_must_match_objective(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.json"
            path.write_text(
                json.dumps(
                    {
                        "objective": "val_topo_wdist_min_at_val_topo_best_ckpt",
                        "best_params": {
                            "w_topo": 1,
                            "w_aniso": 1,
                            "w_size": 1,
                            "lr": 1e-3,
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(infer_tag(path, None), TAG_WDIST)
            with self.assertRaisesRegex(ValueError, "does not match"):
                infer_tag(path, TAG_MCC)


class TestPreflightHomology(unittest.TestCase):
    def test_invalid_homology_rejected(self):
        from tda_ml.config import deep_update, load_config

        cfg = load_config(
            "paper_n100_o20_nocls_h1_ellphi_lpca_power",
            project_root=REPO,
        )
        bad = deep_update(
            cfg,
            {
                "model": {"topology_loss": {"homology_dimensions": [2, 2]}},
                "loss": {"teacher_mode": "nonsense"},
            },
        )
        with self.assertRaises(ValueError):
            preflight_training_config(bad, project_root=REPO)


class TestFreshness(unittest.TestCase):
    def test_missing_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, paths = inspect_seed_metrics(
                out_base=Path(tmp),
                seed=42,
                tag="wdist",
                tune_json=Path(tmp) / "missing.json",
                expected_revision="deadbeef",
            )
            self.assertEqual(status, "missing")
            self.assertEqual(paths, [])

    def test_stale_revision_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tune = root / "best.json"
            tune.write_text("{}", encoding="utf-8")
            logs = root / "paper_s42_x" / "logs"
            logs.mkdir(parents=True)
            (logs / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "seed": 42,
                        "tune_json": str(tune.resolve()),
                        "source_revision": "oldrev",
                    }
                ),
                encoding="utf-8",
            )
            (
                logs / "paper_metrics_test_wdist.json"
            ).write_text("{}", encoding="utf-8")
            status, _ = inspect_seed_metrics(
                out_base=root,
                seed=42,
                tag="wdist",
                tune_json=tune,
                expected_revision="newrev",
            )
            self.assertEqual(status, "stale")


class TestPaperContractExpanded(unittest.TestCase):
    def test_contract_includes_w_class_and_pca(self):
        self.assertEqual(PAPER_NO_CLS_CONTRACT["w_class"], 0.0)
        self.assertEqual(PAPER_NO_CLS_CONTRACT["teacher_local_pca_k"], 10)
        self.assertTrue(PAPER_NO_CLS_CONTRACT["teacher_local_pca_normalize_axes"])


if __name__ == "__main__":
    unittest.main()
