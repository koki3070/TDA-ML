"""Unit tests for label-free one-class baselines (no mixed-label fit)."""

from __future__ import annotations

import unittest

import numpy as np

from tda_ml.oneclass_baselines import (
    apply_known_ratio,
    contamination_ratio,
    confusion_counts,
    evaluate_predictions,
    fit_gmm,
    fit_ocsvm_rbf,
    fit_svdd_primal,
    predict_clouds,
    resolve_threshold,
    score_clouds,
    select_threshold_val_mcc,
    threshold_for_inlier_recall,
)
from tda_ml.ring_dataset import ThinRingsDataset

RING_KW = dict(
    max_points=40,
    num_outliers=10,
    noise_std=0.005,
    outlier_mode="ring_radial",
    ring_count_min=1,
    ring_count_max=1,
    ring_radius_min=0.40,
    ring_radius_max=0.40,
    ring_center_box=0.0,
    ring_outlier_offset_min=0.03,
    ring_outlier_offset_max=0.06,
    ring_outlier_clearance=0.03,
)


def _ring_clean(n: int = 80, radius: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, size=n)
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)


class TestThresholdRules(unittest.TestCase):
    def test_known_ratio_flags_k_fraction(self):
        scores = np.arange(100, dtype=np.float64)
        pred = apply_known_ratio(scores, k=0.2)
        self.assertEqual(int(pred.sum()), 20)
        self.assertTrue(np.all(pred[-20:] == 1))
        self.assertTrue(np.all(pred[:-20] == 0))

    def test_known_ratio_rejects_labels_api(self):
        # The helper has no labels argument; passing k outside (0,1) hard-fails.
        with self.assertRaises(ValueError):
            apply_known_ratio(np.arange(10, dtype=np.float64), k=0.0)
        with self.assertRaises(ValueError):
            apply_known_ratio(np.arange(10, dtype=np.float64), k=1.0)

    def test_inlier_recall_keeps_tau_on_clean(self):
        rng = np.random.default_rng(1)
        clean_scores = rng.normal(0.0, 1.0, size=1000)
        thr = threshold_for_inlier_recall(clean_scores, tau=0.95)
        kept = float(np.mean(clean_scores < thr))
        self.assertGreaterEqual(kept, 0.94)
        self.assertLessEqual(kept, 0.96)

    def test_val_mcc_does_not_read_test(self):
        # Separable val: outliers have strictly higher scores.
        val_scores = [np.array([0.0, 0.1, 0.2, 5.0, 6.0])]
        val_y = [np.array([0, 0, 0, 1, 1])]
        test_scores = [np.array([0.0, 9.0])]
        test_y = [np.array([0, 1])]
        thr, val_mcc = select_threshold_val_mcc(val_scores, val_y)
        self.assertGreater(val_mcc, 0.9)
        # Any cutoff in (0.2, 5] separates val; must not depend on test=9.
        self.assertGreater(thr, 0.2)
        self.assertLessEqual(thr, 5.0)
        del test_scores, test_y  # unused on purpose: leak guard

    def test_resolve_known_ratio_omits_point_labels(self):
        chosen = resolve_threshold("known_ratio", k=1.0 / 6.0)
        self.assertIsNone(chosen["threshold"])
        self.assertAlmostEqual(chosen["k"], 1.0 / 6.0)

    def test_contamination_ratio_paper_rings(self):
        self.assertAlmostEqual(contamination_ratio(100, 20), 20 / 120)


class TestFitUsesCleanOnly(unittest.TestCase):
    def test_ocsvm_and_primal_svdd_score_far_points_higher(self):
        clean = _ring_clean()
        far = np.array([[0.9, 0.9], [0.85, -0.85]])
        on_ring = _ring_clean(n=16, seed=1)
        ocsvm = fit_ocsvm_rbf(clean)
        svdd = fit_svdd_primal(clean)
        for scorer in (ocsvm, svdd):
            s_far = scorer.score(far).mean()
            s_on = scorer.score(on_ring).mean()
            self.assertGreater(s_far, s_on)

    def test_gmm_selects_on_clean_val(self):
        clean_tr = _ring_clean(seed=2)
        clean_va = _ring_clean(n=40, seed=3)
        scorer, n_comp = fit_gmm(
            clean_tr, clean_va, n_components_grid=(2, 4), seed=0
        )
        self.assertIn(n_comp, (2, 4))
        far = np.array([[0.95, 0.95]])
        on = clean_va[:5]
        self.assertGreater(
            float(scorer.score(far).mean()), float(scorer.score(on).mean())
        )


class TestEvaluatePredictions(unittest.TestCase):
    def test_confusion_and_mcc(self):
        y = np.array([0, 0, 1, 1])
        pred = np.array([0, 1, 1, 0])
        c = confusion_counts(y, pred)
        self.assertEqual((c.tp, c.tn, c.fp, c.fn), (1, 1, 1, 1))
        pts = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.2, 0.0]])
        clean = np.array([[1.0, 0.0], [0.0, 1.0]])
        metrics = evaluate_predictions(
            [pred], [y], [pts], [clean], compute_wdist=False
        )
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["mcc"], 0.0)

    def test_predict_clouds_known_ratio(self):
        scores = [np.array([0.0, 1.0, 2.0, 3.0])]
        pred = predict_clouds(scores, rule="known_ratio", k=0.25)
        self.assertEqual(int(pred[0].sum()), 1)
        self.assertEqual(int(pred[0][-1]), 1)


class TestRingsIntegration(unittest.TestCase):
    def test_thin_rings_ocsvm_smoke(self):
        ds = ThinRingsDataset(4, noise_seed=42, **RING_KW)
        cleans, mixed, labels = [], [], []
        for i in range(4):
            pts, lab, clean = ds[i]
            mixed.append(pts.numpy().astype(np.float64))
            labels.append(lab.numpy().astype(int))
            mask = np.abs(clean.numpy()).sum(axis=1) > 1e-6
            cleans.append(clean.numpy().astype(np.float64)[mask])
        train_clean = np.concatenate(cleans[:2], axis=0)
        scorer = fit_ocsvm_rbf(train_clean)
        va_scores = score_clouds(scorer, mixed[2:])
        chosen = resolve_threshold(
            "val_mcc",
            val_mixed_scores=va_scores,
            val_mixed_labels=labels[2:],
            k=contamination_ratio(RING_KW["max_points"], RING_KW["num_outliers"]),
        )
        self.assertEqual(chosen["rule"], "val_mcc")
        self.assertIsNotNone(chosen["threshold"])
        preds = predict_clouds(
            va_scores, rule="val_mcc", threshold=chosen["threshold"]
        )
        metrics = evaluate_predictions(
            preds, labels[2:], mixed[2:], cleans[2:], compute_wdist=False
        )
        self.assertTrue(np.isfinite(metrics["mcc"]))
        self.assertEqual(
            metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"],
            2 * (RING_KW["max_points"] + RING_KW["num_outliers"]),
        )


class TestExperimentImport(unittest.TestCase):
    def test_eval_driver_imports(self):
        import experiments.eval_fair_oneclass_rings as driver

        self.assertTrue(callable(driver.main))
        self.assertIn("val_mcc", driver.THRESHOLD_RULES)


if __name__ == "__main__":
    unittest.main()
