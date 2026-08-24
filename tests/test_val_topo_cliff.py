"""Tests for val_topo cliff gate and rings aggregate contract wiring."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from aggregate_paper_multiseed import (  # noqa: E402
    apply_cliff_policy,
    resolve_aggregate_contract,
    resolve_cliff_policy,
)
from tda_ml.preflight import RINGS_NO_CLS_CONTRACT  # noqa: E402
from tda_ml.val_topo_cliff import (  # noqa: E402
    RINGS_VAL_TOPO_CLIFF_MAX,
    ValTopoCliffError,
    assert_val_topo_cliff,
    best_val_topo_from_metrics_csv,
    cliff_max_from_config,
    cliff_passed,
)


class TestValTopoCliff(unittest.TestCase):
    def test_cliff_max_requires_explicit_threshold(self):
        with self.assertRaisesRegex(ValueError, "val_topo_cliff_max"):
            cliff_max_from_config({"training": {"require_val_topo_cliff": True}})

    def test_cliff_max_none_when_disabled_or_absent(self):
        self.assertIsNone(cliff_max_from_config({}))
        self.assertIsNone(
            cliff_max_from_config({"training": {"require_val_topo_cliff": False}})
        )

    def test_assert_passes_and_fails(self):
        assert_val_topo_cliff(0.1, cliff_max=0.3, context="ok")
        with self.assertRaises(ValTopoCliffError):
            assert_val_topo_cliff(0.45, cliff_max=0.3, context="bad")

    def test_cliff_deadline_helper(self):
        from tda_ml.val_topo_cliff import (
            cliff_deadline_epoch_from_config,
            maybe_raise_cliff_deadline,
        )

        self.assertIsNone(cliff_deadline_epoch_from_config({}))
        self.assertEqual(
            cliff_deadline_epoch_from_config(
                {
                    "training": {
                        "require_val_topo_cliff": True,
                        "val_topo_cliff_max": 0.3,
                        "val_topo_cliff_deadline_epoch": 25,
                    }
                }
            ),
            25,
        )
        maybe_raise_cliff_deadline(
            epoch=24,
            best_val_topo=0.45,
            cliff_max=0.3,
            deadline_epoch=25,
            context="early",
        )
        with self.assertRaises(ValTopoCliffError):
            maybe_raise_cliff_deadline(
                epoch=25,
                best_val_topo=0.45,
                cliff_max=0.3,
                deadline_epoch=25,
                context="late",
            )

    def test_metrics_csv_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f, fieldnames=["epoch", "val_topo_loss", "train_loss"]
                )
                w.writeheader()
                w.writerow({"epoch": 1, "val_topo_loss": 0.5, "train_loss": 1.0})
                w.writerow({"epoch": 2, "val_topo_loss": 0.12, "train_loss": 1.0})
                w.writerow({"epoch": 3, "val_topo_loss": 0.2, "train_loss": 1.0})
            best, ep = best_val_topo_from_metrics_csv(path)
            self.assertAlmostEqual(best, 0.12)
            self.assertEqual(ep, 2)
            self.assertTrue(cliff_passed(best, cliff_max=RINGS_VAL_TOPO_CLIFF_MAX))


class TestAggregateContract(unittest.TestCase):
    def test_auto_rings_from_tune_payload(self):
        name, contract = resolve_aggregate_contract(
            experiment_contract="auto",
            aniso_variant="elongate_barrier",
            tune_payload={"dataset_type": "thin_rings"},
        )
        self.assertEqual(name, "rings")
        self.assertEqual(contract, RINGS_NO_CLS_CONTRACT)

    def test_auto_missing_dataset_type_hard_fails(self):
        with self.assertRaisesRegex(ValueError, "missing dataset_type"):
            resolve_aggregate_contract(
                experiment_contract="auto",
                aniso_variant="elongate_barrier",
                tune_payload={},
            )

    def test_auto_legacy_tangent_without_dataset_type_is_mnist(self):
        name, _ = resolve_aggregate_contract(
            experiment_contract="auto",
            aniso_variant="elongate_barrier",
            tune_payload={"tangent_pca_k": 10},
        )
        self.assertEqual(name, "mnist")

    def test_auto_cliff_policy(self):
        self.assertEqual(
            resolve_cliff_policy(seed_cliff_policy="auto", contract_name="rings"),
            "strict",
        )
        self.assertEqual(
            resolve_cliff_policy(seed_cliff_policy="auto", contract_name="mnist"),
            "off",
        )

    def test_exclude_failed_aggregates_survivors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            by_seed = {}
            for seed, best in [(42, 0.09), (123, 0.45), (456, 0.08)]:
                run = root / f"paper_s{seed}"
                logs = run / "logs"
                logs.mkdir(parents=True)
                with (logs / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["epoch", "val_topo_loss"])
                    w.writeheader()
                    w.writerow({"epoch": 1, "val_topo_loss": best})
                by_seed[seed] = {
                    "seed": seed,
                    "run_dir": str(run),
                    "mcc": 0.5 if best < 0.3 else 0.0,
                    "gmean": 0.5,
                    "wdist": 3.0 if best < 0.3 else 14.0,
                    "tune_json": "/tmp/x.json",
                }
            mean_seeds, failed, _ = apply_cliff_policy(
                method="proposed_wdist_30ep",
                by_seed=by_seed,
                expected_seeds=[42, 123, 456],
                policy="exclude-failed",
                cliff_max=0.3,
            )
            self.assertEqual(mean_seeds, [42, 456])
            self.assertEqual(failed, [123])

            with self.assertRaisesRegex(ValueError, "cliff failed"):
                apply_cliff_policy(
                    method="proposed_wdist_30ep",
                    by_seed=by_seed,
                    expected_seeds=[42, 123, 456],
                    policy="strict",
                    cliff_max=0.3,
                )


if __name__ == "__main__":
    unittest.main()
