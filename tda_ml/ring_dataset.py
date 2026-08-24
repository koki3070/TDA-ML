"""Thin synthetic rings with radial or chord-bridge outliers.

Motivation (2026-07-19, docs/experiments/20260719_neartangent_barrier.md):
on MNIST 100-point clouds the local-PCA teacher ellipses are only mildly
anisotropic (aspect median ~1.8) because strokes have width, so no outlier
*direction* can create an advantage for anisotropic distances. Thin rings are
the minimal geometry where the anisotropy hypothesis is structurally testable:

- the curve is thin (transverse sigma << along-curve spacing), so local-PCA
  ellipses reach aspect ~8-9;
- radial outliers sit within ~1 nearest-neighbour spacing of the ring
  (Euclidean-ambiguous) but cross the ellipse minor axis (Mahalanobis-clear);
- rings are H1 loops; paper persistence includes H0 and H1.

``outlier_mode``:

- ``ring_radial`` — offsets along the ring normal (paper Methods).
- ``ring_bridge`` — interior chord / shortcut points that can kill H1 early
  in a filtration while remaining locally assignable to rim LPCA via NN.

Interface mirrors :class:`tda_ml.data_loader.NoisyMNISTDataset`: items are
``(data, labels, clean_pc)`` with ``data`` of shape
``(max_points + num_outliers, 2)`` shuffled, ``labels`` 0=inlier / 1=outlier,
and ``clean_pc`` the noise-free inlier points.

Split scheme (no images to index): each cloud is generated from
``noise_seed + index_offset + idx``. Train/val use offsets ``0`` /
``train_size``; test uses ``TEST_INDEX_OFFSET`` so its stream never overlaps
train/val for the configured sizes.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset

from tda_ml.cloud_separation import apply_noise_with_min_separation
from tda_ml.numerical_eps import MIN_ELLPHI_CENTER_SEPARATION

TEST_INDEX_OFFSET = 1_000_000
RING_OUTLIER_MODES = frozenset({"ring_radial", "ring_bridge"})


class ThinRingsDataset(Dataset):
    """1-2 thin rings per cloud + radial or chord-bridge outliers.

    All geometry parameters are required explicitly (no silent defaults) in
    line with the repo reproducibility policy; callers wire them from
    ``config['data']``.
    """

    def __init__(
        self,
        num_samples: int,
        *,
        max_points: int,
        num_outliers: int,
        noise_std: float,
        ring_count_min: int,
        ring_count_max: int,
        ring_radius_min: float,
        ring_radius_max: float,
        ring_center_box: float,
        ring_outlier_offset_min: float,
        ring_outlier_offset_max: float,
        ring_outlier_clearance: float,
        outlier_mode: str = "ring_radial",
        ring_bridge_gap_min: float | None = None,
        ring_bridge_gap_max: float | None = None,
        noise_seed: int = 0,
        index_offset: int = 0,
        min_separation: float = MIN_ELLPHI_CENTER_SEPARATION,
        max_attempts: int = 400,
    ) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0; got {num_samples}")
        if max_points < 8:
            raise ValueError(f"max_points must be >= 8; got {max_points}")
        if num_outliers < 0:
            raise ValueError(f"num_outliers must be >= 0; got {num_outliers}")
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0; got {noise_std}")
        if not (1 <= ring_count_min <= ring_count_max):
            raise ValueError(
                f"Require 1 <= ring_count_min <= ring_count_max; "
                f"got {ring_count_min}, {ring_count_max}"
            )
        if not (0 < ring_radius_min <= ring_radius_max):
            raise ValueError(
                f"Require 0 < ring_radius_min <= ring_radius_max; "
                f"got {ring_radius_min}, {ring_radius_max}"
            )
        if ring_center_box < 0:
            raise ValueError(f"ring_center_box must be >= 0; got {ring_center_box}")
        if not (0 < ring_outlier_offset_min <= ring_outlier_offset_max):
            raise ValueError(
                f"Require 0 < ring_outlier_offset_min <= ring_outlier_offset_max; "
                f"got {ring_outlier_offset_min}, {ring_outlier_offset_max}"
            )
        if ring_outlier_clearance < 0:
            raise ValueError(
                f"ring_outlier_clearance must be >= 0; got {ring_outlier_clearance}"
            )
        mode = str(outlier_mode).strip().lower()
        if mode not in RING_OUTLIER_MODES:
            raise ValueError(
                f"outlier_mode must be one of {sorted(RING_OUTLIER_MODES)}, got {outlier_mode!r}"
            )
        self.outlier_mode = mode
        if mode == "ring_radial" and ring_outlier_clearance > ring_outlier_offset_min:
            raise ValueError(
                "ring_outlier_clearance must be <= ring_outlier_offset_min, otherwise "
                "every candidate is rejected against its own source ring; got "
                f"{ring_outlier_clearance} > {ring_outlier_offset_min}"
            )
        if mode == "ring_bridge":
            if ring_bridge_gap_min is None or ring_bridge_gap_max is None:
                raise ValueError(
                    "ring_bridge_gap_min and ring_bridge_gap_max must be set "
                    "explicitly for outlier_mode='ring_bridge'"
                )
            gap_min = float(ring_bridge_gap_min)
            gap_max = float(ring_bridge_gap_max)
            if not (0.0 < gap_min <= gap_max < 2.0 * math.pi):
                raise ValueError(
                    "Require 0 < ring_bridge_gap_min <= ring_bridge_gap_max < 2π; "
                    f"got {gap_min}, {gap_max}"
                )
            self.ring_bridge_gap_min = gap_min
            self.ring_bridge_gap_max = gap_max
        else:
            self.ring_bridge_gap_min = None
            self.ring_bridge_gap_max = None
        extent = ring_radius_max + ring_center_box + (
            ring_outlier_offset_max if mode == "ring_radial" else 0.0
        )
        if extent > 1.0:
            raise ValueError(
                "ring geometry can leave [-1, 1]^2: "
                f"extent={extent:.3f} > 1.0 (mode={mode})"
            )
        if min_separation < 0:
            raise ValueError(f"min_separation must be >= 0; got {min_separation}")

        self.num_samples = int(num_samples)
        self.max_points = int(max_points)
        self.num_outliers = int(num_outliers)
        self.noise_std = float(noise_std)
        self.ring_count_min = int(ring_count_min)
        self.ring_count_max = int(ring_count_max)
        self.ring_radius_min = float(ring_radius_min)
        self.ring_radius_max = float(ring_radius_max)
        self.ring_center_box = float(ring_center_box)
        self.ring_outlier_offset_min = float(ring_outlier_offset_min)
        self.ring_outlier_offset_max = float(ring_outlier_offset_max)
        self.ring_outlier_clearance = float(ring_outlier_clearance)
        self.noise_seed = int(noise_seed)
        self.index_offset = int(index_offset)
        self.min_separation = float(min_separation)
        self.max_attempts = int(max_attempts)

    def __len__(self) -> int:
        return self.num_samples

    def _sample_clean_rings(self, rng: torch.Generator):
        """Sample ring parameters and min-separated clean points on the circles."""
        n_rings = int(
            torch.randint(self.ring_count_min, self.ring_count_max + 1, (1,), generator=rng)
        )
        counts = []
        remaining = self.max_points
        for r in range(n_rings):
            if r == n_rings - 1:
                counts.append(remaining)
            else:
                # 35-65% share keeps every ring dense enough for local PCA (k=10).
                share = 0.35 + 0.3 * float(torch.rand(1, generator=rng))
                cnt = max(20, int(round(remaining * share)))
                cnt = min(cnt, remaining - 20)
                counts.append(cnt)
                remaining -= cnt
        centers, radii = [], []
        for _ in range(n_rings):
            centers.append((torch.rand(2, generator=rng) * 2.0 - 1.0) * self.ring_center_box)
            radii.append(
                self.ring_radius_min
                + (self.ring_radius_max - self.ring_radius_min)
                * float(torch.rand(1, generator=rng))
            )

        clean = torch.empty(self.max_points, 2)
        ring_of = torch.empty(self.max_points, dtype=torch.long)
        pos = 0
        for r, (c, radius, cnt) in enumerate(zip(centers, radii, counts)):
            for j in range(cnt):
                for _ in range(self.max_attempts):
                    ang = float(torch.rand(1, generator=rng)) * 2.0 * math.pi
                    cand = c + radius * torch.tensor([math.cos(ang), math.sin(ang)])
                    if pos > 0 and self.min_separation > 0:
                        d = torch.linalg.norm(clean[:pos] - cand, dim=-1).min()
                        if bool((d < self.min_separation).item()):
                            continue
                    clean[pos] = cand
                    ring_of[pos] = r
                    pos += 1
                    break
                else:
                    raise RuntimeError(
                        "Failed to place a min-separated clean ring point after "
                        f"{self.max_attempts} attempts (ring={r}, point={j}, "
                        f"radius={radius:.3f}, min_separation={self.min_separation})."
                    )
        return clean, ring_of, centers, radii

    def _sample_radial_outliers(
        self,
        rng: torch.Generator,
        clean: torch.Tensor,
        ring_of: torch.Tensor,
        centers: list[torch.Tensor],
        radii: list[float],
        inliers: torch.Tensor,
    ) -> torch.Tensor:
        outliers = torch.empty(self.num_outliers, 2)
        n = clean.shape[0]
        for j in range(self.num_outliers):
            for _ in range(self.max_attempts):
                i = int(torch.randint(0, n, (1,), generator=rng))
                c = centers[int(ring_of[i])]
                v = clean[i] - c
                nv = float(torch.linalg.norm(v))
                if nv < 1e-9:
                    continue
                u = v / nv
                sign = 1.0 if float(torch.rand(1, generator=rng)) < 0.5 else -1.0
                mag = self.ring_outlier_offset_min + (
                    self.ring_outlier_offset_max - self.ring_outlier_offset_min
                ) * float(torch.rand(1, generator=rng))
                cand = clean[i] + (sign * mag) * u
                if not bool(torch.all((cand >= -1.0) & (cand <= 1.0)).item()):
                    continue
                # Semantic guard: candidate must stay off EVERY ring band, not
                # just its source ring (a second ring may pass nearby).
                ok = True
                for c_r, r_r in zip(centers, radii):
                    band = abs(float(torch.linalg.norm(cand - c_r)) - r_r)
                    if band < self.ring_outlier_clearance:
                        ok = False
                        break
                if not ok:
                    continue
                if self.min_separation > 0:
                    d = torch.linalg.norm(inliers - cand, dim=-1).min()
                    if bool((d < self.min_separation).item()):
                        continue
                    if j > 0:
                        d_out = torch.linalg.norm(outliers[:j] - cand, dim=-1).min()
                        if bool((d_out < self.min_separation).item()):
                            continue
                outliers[j] = cand
                break
            else:
                raise RuntimeError(
                    "Failed to sample an in-bounds, ring-cleared radial outlier after "
                    f"{self.max_attempts} attempts (outlier_index={j}, "
                    f"offset=[{self.ring_outlier_offset_min}, {self.ring_outlier_offset_max}], "
                    f"clearance={self.ring_outlier_clearance}, "
                    f"min_separation={self.min_separation})."
                )
        return outliers

    def _candidate_cleared(
        self,
        cand: torch.Tensor,
        *,
        centers: list[torch.Tensor],
        radii: list[float],
        inliers: torch.Tensor,
        outliers: torch.Tensor,
        n_out_placed: int,
    ) -> bool:
        if not bool(torch.all((cand >= -1.0) & (cand <= 1.0)).item()):
            return False
        for c_r, r_r in zip(centers, radii):
            band = abs(float(torch.linalg.norm(cand - c_r)) - r_r)
            if band < self.ring_outlier_clearance:
                return False
        if self.min_separation > 0:
            d = torch.linalg.norm(inliers - cand, dim=-1).min()
            if bool((d < self.min_separation).item()):
                return False
            if n_out_placed > 0:
                d_out = torch.linalg.norm(outliers[:n_out_placed] - cand, dim=-1).min()
                if bool((d_out < self.min_separation).item()):
                    return False
        return True

    def _sample_bridge_outliers(
        self,
        rng: torch.Generator,
        centers: list[torch.Tensor],
        radii: list[float],
        inliers: torch.Tensor,
    ) -> torch.Tensor:
        """Interior chord points spanning a random angular gap (H1 shortcuts)."""
        assert self.ring_bridge_gap_min is not None
        assert self.ring_bridge_gap_max is not None
        outliers = torch.empty(self.num_outliers, 2)
        n_rings = len(centers)
        for j in range(self.num_outliers):
            for _ in range(self.max_attempts):
                r_idx = int(torch.randint(0, n_rings, (1,), generator=rng))
                c = centers[r_idx]
                radius = float(radii[r_idx])
                theta0 = float(torch.rand(1, generator=rng)) * 2.0 * math.pi
                gap = self.ring_bridge_gap_min + (
                    self.ring_bridge_gap_max - self.ring_bridge_gap_min
                ) * float(torch.rand(1, generator=rng))
                theta1 = theta0 + gap
                p0 = c + radius * torch.tensor(
                    [math.cos(theta0), math.sin(theta0)], dtype=c.dtype
                )
                p1 = c + radius * torch.tensor(
                    [math.cos(theta1), math.sin(theta1)], dtype=c.dtype
                )
                # Stay away from the rim endpoints (t near 0/1 looks radial-local).
                t = 0.25 + 0.5 * float(torch.rand(1, generator=rng))
                cand = (1.0 - t) * p0 + t * p1
                if self._candidate_cleared(
                    cand,
                    centers=centers,
                    radii=radii,
                    inliers=inliers,
                    outliers=outliers,
                    n_out_placed=j,
                ):
                    outliers[j] = cand
                    break
            else:
                raise RuntimeError(
                    "Failed to sample a ring-cleared bridge outlier after "
                    f"{self.max_attempts} attempts (outlier_index={j}, "
                    f"gap=[{self.ring_bridge_gap_min}, {self.ring_bridge_gap_max}], "
                    f"clearance={self.ring_outlier_clearance}, "
                    f"min_separation={self.min_separation})."
                )
        return outliers

    def __getitem__(self, idx: int):
        if not (0 <= idx < self.num_samples):
            raise IndexError(idx)
        rng = torch.Generator()
        rng.manual_seed(self.noise_seed + self.index_offset + idx)

        clean, ring_of, centers, radii = self._sample_clean_rings(rng)
        inliers = apply_noise_with_min_separation(clean, self.noise_std, generator=rng)

        if self.num_outliers > 0:
            if self.outlier_mode == "ring_radial":
                outliers = self._sample_radial_outliers(
                    rng, clean, ring_of, centers, radii, inliers
                )
            else:
                outliers = self._sample_bridge_outliers(rng, centers, radii, inliers)
        else:
            outliers = torch.empty(0, 2)

        total = self.max_points + self.num_outliers
        data = torch.zeros(total, 2)
        labels = torch.ones(total, dtype=torch.long)
        data[: self.max_points] = inliers
        labels[: self.max_points] = 0
        if self.num_outliers > 0:
            data[self.max_points :] = outliers

        perm = torch.randperm(total, generator=rng)
        data = data[perm]
        labels = labels[perm]

        clean_pc = torch.zeros(self.max_points, 2)
        clean_pc[: clean.shape[0]] = clean
        return data, labels, clean_pc


REQUIRED_RING_KEYS = (
    "ring_count_min",
    "ring_count_max",
    "ring_radius_min",
    "ring_radius_max",
    "ring_center_box",
    "ring_outlier_offset_min",
    "ring_outlier_offset_max",
    "ring_outlier_clearance",
)


def ring_kwargs_from_config(data_cfg: dict) -> dict:
    """Extract required ring geometry keys, hard-failing on any omission."""
    for key in REQUIRED_RING_KEYS:
        if key not in data_cfg:
            raise ValueError(
                f"data.{key} must be set explicitly for dataset_type=thin_rings"
            )
    if "outlier_mode" not in data_cfg:
        raise ValueError(
            "data.outlier_mode must be set explicitly for dataset_type=thin_rings "
            f"(one of {sorted(RING_OUTLIER_MODES)})"
        )
    mode = str(data_cfg["outlier_mode"]).strip().lower()
    if mode not in RING_OUTLIER_MODES:
        raise ValueError(
            f"data.outlier_mode must be one of {sorted(RING_OUTLIER_MODES)}, got {mode!r}"
        )
    out = dict(
        outlier_mode=mode,
        ring_count_min=int(data_cfg["ring_count_min"]),
        ring_count_max=int(data_cfg["ring_count_max"]),
        ring_radius_min=float(data_cfg["ring_radius_min"]),
        ring_radius_max=float(data_cfg["ring_radius_max"]),
        ring_center_box=float(data_cfg["ring_center_box"]),
        ring_outlier_offset_min=float(data_cfg["ring_outlier_offset_min"]),
        ring_outlier_offset_max=float(data_cfg["ring_outlier_offset_max"]),
        ring_outlier_clearance=float(data_cfg["ring_outlier_clearance"]),
    )
    if mode == "ring_bridge":
        for key in ("ring_bridge_gap_min", "ring_bridge_gap_max"):
            if key not in data_cfg:
                raise ValueError(
                    f"data.{key} must be set explicitly for outlier_mode='ring_bridge'"
                )
        out["ring_bridge_gap_min"] = float(data_cfg["ring_bridge_gap_min"])
        out["ring_bridge_gap_max"] = float(data_cfg["ring_bridge_gap_max"])
    return out
