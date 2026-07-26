"""Tests for safe checkpoint loading helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from tda_ml.checkpoint_io import extract_model_state_dict, load_torch_checkpoint


def _minimal_state_dict() -> dict[str, torch.Tensor]:
    """Tiny state dict for round-trip tests (no dependency on ``tda_ml.models``)."""
    return {
        "layer.weight": torch.randn(4, 8),
        "layer.bias": torch.zeros(4),
    }


class TestCheckpointIO(unittest.TestCase):
    def test_roundtrip_state_dict_wrapper(self) -> None:
        original = _minimal_state_dict()
        payload = {"epoch": 1, "model_state_dict": original}
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = Path(f.name)
        try:
            torch.save(payload, path)
            loaded = load_torch_checkpoint(path, map_location="cpu")
            sd = extract_model_state_dict(loaded)
            self.assertEqual(set(sd.keys()), set(original.keys()))
            for key in original:
                self.assertTrue(torch.equal(sd[key], original[key]))
        finally:
            path.unlink(missing_ok=True)

    def test_roundtrip_bare_state_dict(self) -> None:
        original = _minimal_state_dict()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = Path(f.name)
        try:
            torch.save(original, path)
            loaded = load_torch_checkpoint(path, map_location="cpu")
            sd = extract_model_state_dict(loaded)
            self.assertTrue(torch.equal(sd["layer.weight"], original["layer.weight"]))
        finally:
            path.unlink(missing_ok=True)

    def test_extract_model_state_dict_rejects_meta_only_payload(self) -> None:
        bad = {"epoch": 1, "val_mcc": 0.5}
        with self.assertRaises(ValueError):
            extract_model_state_dict(bad)

    def test_extract_model_state_dict_rejects_bad_nested_model_state_dict(self) -> None:
        with self.assertRaises(ValueError):
            extract_model_state_dict({"model_state_dict": {"epoch": 1}})


if __name__ == "__main__":
    unittest.main()
