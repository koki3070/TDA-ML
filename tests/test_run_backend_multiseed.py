"""Tests for experiments/run_backend_multiseed.py (protocol helpers)."""

from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

import run_backend_multiseed as rbm  # noqa: E402


class TestRunBackendMultiseed(unittest.TestCase):
    def test_parse_metrics_csv_best_and_final(self) -> None:
        rows = [
            {"val_mcc": "0.1", "val_recall": "0.2", "val_loss": "1.0"},
            {"val_mcc": "0.5", "val_recall": "0.3", "val_loss": "0.9"},
            {"val_mcc": "0.4", "val_recall": "0.8", "val_loss": "0.7"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["val_mcc", "val_recall", "val_loss"])
            writer.writeheader()
            writer.writerows(rows)
            path = f.name
        try:
            m = rbm.parse_metrics_csv(path)
            self.assertAlmostEqual(m["final_val_mcc"], 0.4)
            self.assertAlmostEqual(m["final_val_recall"], 0.8)
            self.assertAlmostEqual(m["final_val_loss"], 0.7)
            self.assertAlmostEqual(m["best_val_mcc"], 0.5)
            self.assertAlmostEqual(m["best_val_recall"], 0.3)
            self.assertAlmostEqual(m["best_val_loss"], 0.9)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_parse_metrics_csv_mcc_tie_breaks_on_first_epoch(self) -> None:
        rows = [
            {"val_mcc": "0.5", "val_recall": "0.1", "val_loss": "2.0"},
            {"val_mcc": "0.5", "val_recall": "0.9", "val_loss": "1.0"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["val_mcc", "val_recall", "val_loss"])
            writer.writeheader()
            writer.writerows(rows)
            path = f.name
        try:
            m = rbm.parse_metrics_csv(path)
            self.assertAlmostEqual(m["best_val_recall"], 0.1)
            self.assertAlmostEqual(m["best_val_loss"], 2.0)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_parse_metrics_csv_allows_non_finite_val_loss(self) -> None:
        rows = [
            {"val_mcc": "0.1", "val_recall": "0.2", "val_loss": "1.0"},
            {"val_mcc": "0.2", "val_recall": "0.3", "val_loss": "nan"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["val_mcc", "val_recall", "val_loss"])
            writer.writeheader()
            writer.writerows(rows)
            path = f.name
        try:
            m = rbm.parse_metrics_csv(path)
            self.assertAlmostEqual(m["best_val_mcc"], 0.2)
            self.assertAlmostEqual(m["best_val_recall"], 0.3)
            self.assertTrue(math.isnan(m["best_val_loss"]))
            self.assertTrue(math.isnan(m["final_val_loss"]))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_sample_std_over_seeds(self) -> None:
        self.assertEqual(rbm.sample_std_over_seeds([1.0]), 0.0)
        self.assertAlmostEqual(rbm.sample_std_over_seeds([0.0, 2.0]), 2**0.5)

    def test_write_backend_stats_sample_std(self) -> None:
        header = rbm.PROGRESS_HEADER
        rows = [
            {
                "backend": "mahalanobis",
                "seed": "1",
                "epochs": "50",
                "run_dir": "/tmp/a",
                "best_val_mcc": "0.0",
                "best_val_recall": "0",
                "best_val_loss": "0",
                "final_val_mcc": "0.0",
                "final_val_recall": "0",
                "final_val_loss": "0",
            },
            {
                "backend": "mahalanobis",
                "seed": "2",
                "epochs": "50",
                "run_dir": "/tmp/b",
                "best_val_mcc": "2.0",
                "best_val_recall": "0",
                "best_val_loss": "0",
                "final_val_mcc": "2.0",
                "final_val_recall": "0",
                "final_val_loss": "0",
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            progress = Path(td) / "progress_summary.csv"
            stats = Path(td) / "backend_stats.csv"
            with progress.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                w.writerows(rows)
            rbm.write_backend_stats(str(progress), str(stats))
            with stats.open(newline="") as f:
                out = list(csv.reader(f))
            self.assertEqual(out[0][0], "backend")
            # mahalanobis, n=2, mean best=1, std_best=sqrt(2), mean final=1, std_final=sqrt(2)
            self.assertEqual(out[1][0], "mahalanobis")
            self.assertEqual(out[1][1], "2")
            self.assertAlmostEqual(float(out[1][2]), 1.0)
            self.assertAlmostEqual(float(out[1][3]), 2**0.5)
            self.assertAlmostEqual(float(out[1][4]), 1.0)
            self.assertAlmostEqual(float(out[1][5]), 2**0.5)
