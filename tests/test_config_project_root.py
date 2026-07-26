"""Configuration loading resolves paths from package/repo root by default."""

from __future__ import annotations

import unittest

from tda_ml.config import (
    deep_update,
    default_data_root,
    default_project_root,
    load_config,
    model_kwargs_from_config,
)


class TestConfigProjectRoot(unittest.TestCase):
    def test_default_root_contains_configs_base(self) -> None:
        root = default_project_root()
        self.assertTrue((root / "configs" / "base.yaml").is_file())

    def test_default_data_root_is_under_project_root(self) -> None:
        self.assertEqual(default_data_root(), default_project_root() / "data")

    def test_load_dev_from_default_root(self) -> None:
        cfg = load_config("dev")
        self.assertIn("data", cfg)
        self.assertIn("training", cfg)
        self.assertEqual(cfg["data"]["train_size"], 80)

    def test_deep_update_nested(self) -> None:
        base = {"model": {"topology_loss": {"distance_backend": "mahalanobis"}}}
        deep_update(base, {"model": {"topology_loss": {"distance_backend": "ellphi"}}})
        self.assertEqual(base["model"]["topology_loss"]["distance_backend"], "ellphi")

    def test_model_kwargs_from_config(self) -> None:
        cfg = load_config("reproduce")
        kw = model_kwargs_from_config(cfg)
        self.assertEqual(kw["point_dim"], 2)
        self.assertNotIn("ellipse_param_dim", kw)


if __name__ == "__main__":
    unittest.main()
