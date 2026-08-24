"""Unit tests for the thin synthetic rings dataset."""

from __future__ import annotations

import unittest

import torch

from tda_ml.local_pca import local_pca_ellipse_params
from tda_ml.numerical_eps import MIN_ELLPHI_CENTER_SEPARATION
from tda_ml.ring_dataset import ThinRingsDataset, ring_kwargs_from_config

RING_KW = dict(
    max_points=100,
    num_outliers=20,
    noise_std=0.005,
    outlier_mode="ring_radial",
    ring_count_min=1,
    ring_count_max=2,
    ring_radius_min=0.25,
    ring_radius_max=0.60,
    ring_center_box=0.30,
    ring_outlier_offset_min=0.03,
    ring_outlier_offset_max=0.06,
    ring_outlier_clearance=0.03,
)


class TestThinRingsDataset(unittest.TestCase):
    def test_shapes_and_labels(self):
        ds = ThinRingsDataset(5, noise_seed=42, **RING_KW)
        data, labels, clean = ds[0]
        self.assertEqual(tuple(data.shape), (120, 2))
        self.assertEqual(tuple(labels.shape), (120,))
        self.assertEqual(tuple(clean.shape), (100, 2))
        self.assertEqual(int((labels == 0).sum()), 100)
        self.assertEqual(int((labels == 1).sum()), 20)
        self.assertTrue(torch.all(data >= -1.0).item())
        self.assertTrue(torch.all(data <= 1.0).item())

    def test_deterministic_and_split_disjoint(self):
        a = ThinRingsDataset(3, noise_seed=42, **RING_KW)[1]
        b = ThinRingsDataset(3, noise_seed=42, **RING_KW)[1]
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        # A different index_offset must generate a different cloud stream.
        c = ThinRingsDataset(3, noise_seed=42, index_offset=4500, **RING_KW)[1]
        self.assertFalse(torch.equal(a[0], c[0]))

    def test_min_separation_all_points(self):
        ds = ThinRingsDataset(4, noise_seed=7, **RING_KW)
        for i in range(4):
            data, labels, clean = ds[i]
            d = torch.cdist(data, data)
            d.fill_diagonal_(float("inf"))
            self.assertGreaterEqual(float(d.min()), MIN_ELLPHI_CENTER_SEPARATION)

    def test_teacher_ellipses_strongly_anisotropic(self):
        # The design goal: local-PCA teacher aspect must be far above MNIST's ~1.8.
        ds = ThinRingsDataset(6, noise_seed=42, **RING_KW)
        aspects = []
        for i in range(6):
            _, _, clean = ds[i]
            p = local_pca_ellipse_params(clean, k=10, normalize_axes=True)
            aspects.extend((p[:, 0] / p[:, 1]).tolist())
        med = torch.tensor(aspects).median()
        self.assertGreater(float(med), 4.0)

    def test_outliers_off_ring_band(self):
        # Outliers must be at least ~clearance away from every CLEAN ring point
        # (band clearance implies point clearance up to sampling density).
        ds = ThinRingsDataset(4, noise_seed=3, **RING_KW)
        for i in range(4):
            data, labels, clean = ds[i]
            out = data[labels == 1]
            d = torch.cdist(out, clean).min(dim=1).values
            self.assertGreaterEqual(float(d.min()), RING_KW["ring_outlier_offset_min"] * 0.9)

    def test_invalid_geometry_rejected(self):
        bad = dict(RING_KW)
        bad.update(ring_radius_max=0.70, ring_center_box=0.30)  # extent 1.06 > 1
        with self.assertRaises(ValueError):
            ThinRingsDataset(2, **bad)
        bad = dict(RING_KW)
        bad.update(ring_outlier_clearance=0.05)  # > offset_min
        with self.assertRaises(ValueError):
            ThinRingsDataset(2, **bad)
        bad = dict(RING_KW)
        bad.update(ring_count_min=0)
        with self.assertRaises(ValueError):
            ThinRingsDataset(2, **bad)

    def test_ring_kwargs_from_config_requires_all_keys(self):
        cfg = {k: v for k, v in RING_KW.items() if k.startswith("ring_") or k == "outlier_mode"}
        self.assertEqual(ring_kwargs_from_config(cfg)["ring_count_max"], 2)
        del cfg["ring_outlier_clearance"]
        with self.assertRaises(ValueError):
            ring_kwargs_from_config(cfg)

    def test_bridge_outliers_interior(self):
        kw = dict(RING_KW)
        kw.update(
            outlier_mode="ring_bridge",
            ring_count_min=1,
            ring_count_max=1,
            ring_bridge_gap_min=1.2,
            ring_bridge_gap_max=2.0,
            ring_outlier_clearance=0.04,
            ring_outlier_offset_min=0.04,
            ring_outlier_offset_max=0.08,
        )
        ds = ThinRingsDataset(4, noise_seed=11, **kw)
        for i in range(4):
            data, labels, clean = ds[i]
            out = data[labels == 1]
            # Bridges must stay off the clean rim band.
            d = torch.cdist(out, clean).min(dim=1).values
            self.assertGreaterEqual(float(d.min()), 0.035)
            # And lie inside the convex hull scale of the clean ring (interior).
            c = clean.mean(dim=0)
            r_clean = torch.linalg.norm(clean - c, dim=-1).median()
            r_out = torch.linalg.norm(out - c, dim=-1)
            self.assertTrue(bool((r_out < r_clean - 0.02).all().item()))


if __name__ == "__main__":
    unittest.main()
