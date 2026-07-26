"""Smoke tests for paper evaluation entrypoints (imports + contracts)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from tda_ml.checkpoint_io import resolve_val_topo_checkpoint
from tda_ml.config import load_config
from tda_ml.preflight import assert_paper_no_cls_contract, preflight_paper_eval_run_dir
from tda_ml.reproducibility import build_reproducibility_manifest_fields


class TestPaperEvalImports(unittest.TestCase):
    def test_baselines_and_protocol_import(self):
        import experiments.eval_baselines as baselines
        import experiments.eval_paper as protocol

        self.assertTrue(callable(baselines.evaluate_adbscan))
        self.assertTrue(callable(protocol.load_model_from_run))
        self.assertTrue(callable(protocol.build_split_loader))

    def test_train_and_eval_share_data_root(self):
        """Training and paper eval must not disagree on MNIST cache path (cwd-independent)."""
        from tda_ml.config import default_data_root
        from tda_ml.run_setup import default_data_root as train_data_root

        import experiments.eval_paper as protocol

        self.assertIs(protocol.default_data_root, default_data_root)
        self.assertIs(train_data_root, default_data_root)
        self.assertEqual(default_data_root().name, "data")

    def test_public_paper_configs_pass_contract(self):
        assert_paper_no_cls_contract(
            load_config("paper_n100_o20_nocls_h1_ellphi_lpca_power")
        )
        assert_paper_no_cls_contract(
            load_config("tune_n100_o20_nocls_h1_ellphi_lpca_power")
        )

    def test_paper_eval_requires_best_model_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "logs").mkdir()
            with self.assertRaises(FileNotFoundError):
                preflight_paper_eval_run_dir(run_dir)
            # Wrong name must not satisfy paper eval.
            torch.save({"model_state_dict": {}, "selection_metric_value": 1.0}, run_dir / "final_model.pth")
            with self.assertRaises(FileNotFoundError):
                resolve_val_topo_checkpoint(run_dir)

    def test_manifest_records_zero_pad_constant(self):
        cfg = load_config("paper_n100_o20_nocls_h1_ellphi_lpca_power")
        fields = build_reproducibility_manifest_fields(cfg)
        self.assertIn("ZERO_PAD_ABS_SUM", fields["numerical_constants"])


if __name__ == "__main__":
    unittest.main()
