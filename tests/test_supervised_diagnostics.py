"""Tests for supervised training early-abort diagnostics."""

import unittest

import torch

from tda_ml.models import AnisotropicOutlierClassifier
from tda_ml.supervised_diagnostics import (
    collect_classification_health,
    collect_ellipse_stats,
    infer_failure_hypotheses,
    should_early_abort,
)


class TestSupervisedDiagnostics(unittest.TestCase):
    def test_should_early_abort_after_probe(self):
        cfg = {
            "enabled": True,
            "probe_epochs": 3,
            "min_val_mcc": 0.05,
            "min_train_mcc": 0.02,
        }
        ok, _ = should_early_abort(
            epoch=2,
            best_val_mcc=0.0,
            val_recall=1.0,
            val_mcc=0.0,
            train_mcc=0.0,
            early_abort_cfg=cfg,
        )
        self.assertFalse(ok)

        abort, reason = should_early_abort(
            epoch=3,
            best_val_mcc=0.0,
            val_recall=1.0,
            val_mcc=0.0,
            train_mcc=0.0,
            early_abort_cfg=cfg,
        )
        self.assertTrue(abort)
        self.assertIn("best_val_mcc", reason)

    def test_should_not_abort_when_mcc_rises(self):
        cfg = {"enabled": True, "probe_epochs": 3, "min_val_mcc": 0.05}
        abort, _ = should_early_abort(
            epoch=3,
            best_val_mcc=0.12,
            val_recall=0.5,
            val_mcc=0.12,
            train_mcc=0.15,
            early_abort_cfg=cfg,
        )
        self.assertFalse(abort)

    def test_all_outlier_hypothesis(self):
        report = {
            "classification": {"pred_outlier_fraction": 0.99, "logits": {"min": 0.0, "max": 0.01}},
            "ellipse": {"frac_below_1e-4": 0.0, "a": {"max": 1.0}},
            "encoder": {"base_axes_minor": {"median": 0.5}},
            "distance": {"distance_matrix": {"nan_count": 0}},
            "metrics_history": [{"val_mcc": 0.0}],
        }
        hyps = infer_failure_hypotheses(report)
        ids = {h["id"] for h in hyps}
        self.assertIn("all_outlier_predictions", ids)
        self.assertIn("mcc_never_rose", ids)

    def test_collect_ellipse_stats_on_model_output(self):
        model = AnisotropicOutlierClassifier()
        x = torch.rand(2, 16, 2)
        with torch.no_grad():
            _, params = model(x)
        stats = collect_ellipse_stats(params)
        self.assertGreater(stats["a"]["median"], 0.0)
        self.assertIn("aspect_ratio", stats)

    def test_collect_classification_health(self):
        logits = torch.tensor([[[5.0], [-5.0], [0.0], [2.0]]])
        labels = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        health = collect_classification_health(logits, labels, threshold=0.5)
        self.assertEqual(health["pred_outlier_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
