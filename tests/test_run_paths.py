"""Tests for tda_ml.run_paths."""

from __future__ import annotations

import datetime
import unittest

from tda_ml.run_paths import (
    OUTPUT_CATEGORIES,
    build_run_dir,
    experiment_base,
    resolve_run_slug,
    shorten_config_id,
    tune_base,
    visualization_filename,
)


class TestRunPaths(unittest.TestCase):
    def test_shorten_config_id(self) -> None:
        self.assertEqual(shorten_config_id("paper_seed42"), "paper_s42")
        self.assertEqual(shorten_config_id("backend_ellphi_seed42"), "eph_s42")
        self.assertEqual(shorten_config_id("tune_mcc_t010"), "t010")

    def test_resolve_run_slug_prefers_explicit(self) -> None:
        cfg = {
            "meta": {"config_id": "backend_ellphi_seed42", "run_slug": "custom"},
            "outputs": {},
        }
        self.assertEqual(resolve_run_slug(cfg), "custom")

    def test_build_run_dir(self) -> None:
        when = datetime.datetime(2026, 7, 9, 13, 5, 0)
        cfg = {
            "meta": {"config_id": "tune_mcc_t010", "run_slug": "t010"},
            "outputs": {"base_dir": "outputs/tune/0709_mcc"},
        }
        run_dir, slug, stamp = build_run_dir(cfg, when=when)
        self.assertEqual(slug, "t010")
        self.assertEqual(stamp, "0709_130500")
        self.assertEqual(run_dir, "outputs/tune/0709_mcc/t010_0709_130500")

    def test_build_run_dir_refuses_existing(self) -> None:
        import tempfile
        from pathlib import Path

        when = datetime.datetime(2026, 7, 9, 13, 5, 1)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "out"
            existing = base / "t010_0709_130501"
            existing.mkdir(parents=True)
            cfg = {
                "meta": {"run_slug": "t010"},
                "outputs": {"base_dir": str(base)},
            }
            with self.assertRaises(FileExistsError):
                build_run_dir(cfg, when=when)

    def test_experiment_base(self) -> None:
        when = datetime.datetime(2026, 7, 9, 13, 5)
        self.assertEqual(experiment_base("supervised", "paper30", when=when), "outputs/supervised/0709_paper30")
        self.assertEqual(tune_base("mcc", when=when), "outputs/tune/0709_mcc")
        self.assertEqual(OUTPUT_CATEGORIES, {"supervised", "supervised_no_cls", "tune"})

    def test_visualization_filename(self) -> None:
        self.assertEqual(visualization_filename(30), "e30.png")


if __name__ == "__main__":
    unittest.main()
