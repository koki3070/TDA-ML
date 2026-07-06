#!/usr/bin/env python3
"""Side-by-side heatmaps of Mahalanobis vs ellphi anisotropic distance matrices."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tda_ml.config import load_config
from tda_ml.data_loader import NoisyMNISTDataset
from tda_ml.dbscan import calculate_anisotropic_distance_matrix
from tda_ml.model_inference import load_model

REPO_ROOT = Path(__file__).resolve().parents[1]

INLIER_COLOR = "#2166ac"
OUTLIER_COLOR = "#b2182b"


def _build_val_dataset(config: dict, sample_idx: int) -> tuple[np.ndarray, np.ndarray, int]:
    data_cfg = config["data"]
    seed = int(data_cfg.get("seed", 42))
    train_size = int(data_cfg.get("train_size", 4500))
    val_size = int(data_cfg.get("val_size", 500))
    generator = torch.Generator().manual_seed(seed)
    full_train_indices = torch.randperm(60000, generator=generator)[: train_size + val_size]
    val_indices = full_train_indices[train_size:]

    dataset = NoisyMNISTDataset(
        root=str(REPO_ROOT / "data"),
        train=True,
        max_points=data_cfg["max_points"],
        num_outliers=data_cfg["num_outliers"],
        indices=val_indices,
        deterministic=True,
        noise_seed=seed,
        preload=True,
    )
    idx = sample_idx % len(dataset)
    data, labels, _clean = dataset[idx]
    return data.numpy(), labels.numpy().astype(int).flatten(), idx


def _sort_by_label(points: np.ndarray, labels: np.ndarray, params: np.ndarray):
    order = np.argsort(labels, kind="stable")
    return points[order], labels[order], params[order]


def _off_diagonal(dm: np.ndarray) -> np.ndarray:
    return dm[np.triu_indices(dm.shape[0], k=1)]


def _matrix_stats(dm: np.ndarray) -> dict[str, float]:
    off = _off_diagonal(dm)
    return {
        "min": float(off.min()),
        "median": float(np.median(off)),
        "mean": float(off.mean()),
        "max": float(off.max()),
    }


def _normalize_matrix(dm: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    """Return (normalized matrix, scale factor applied to off-diagonal pairs)."""
    if mode == "none":
        return dm, 1.0
    off = _off_diagonal(dm)
    if mode == "median":
        scale = float(np.median(off))
    elif mode == "max":
        scale = float(off.max())
    elif mode == "mean":
        scale = float(off.mean())
    else:
        raise ValueError(f"Unknown normalize mode: {mode!r}")
    scale = max(scale, 1e-12)
    out = dm.copy()
    n = dm.shape[0]
    mask = ~np.eye(n, dtype=bool)
    out[mask] = dm[mask] / scale
    return out, scale


def plot_distance_matrices(
    points: np.ndarray,
    params: np.ndarray,
    labels: np.ndarray,
    *,
    output_path: Path,
    title_suffix: str = "",
    normalize: str = "none",
) -> None:
    dm_maha = calculate_anisotropic_distance_matrix(
        points, params, metric="max", probs=None, backend="mahalanobis"
    )
    dm_ell = calculate_anisotropic_distance_matrix(
        points, params, metric="max", probs=None, backend="ellphi"
    )

    dm_maha, scale_m = _normalize_matrix(dm_maha, normalize)
    dm_ell, scale_e = _normalize_matrix(dm_ell, normalize)
    ratio = dm_ell / np.clip(dm_maha, 1e-12, None)

    stats_m = _matrix_stats(dm_maha)
    stats_e = _matrix_stats(dm_ell)
    stats_r = _matrix_stats(ratio)

    n = points.shape[0]
    n_in = int((labels == 0).sum())
    n_out = int((labels == 1).sum())
    boundary = n_in - 0.5

    norm_note = ""
    if normalize != "none":
        norm_note = f" (normalized by off-diagonal {normalize}; scales maha={scale_m:.3f}, ellphi={scale_e:.3f})"

    vmax = max(stats_m["max"], stats_e["max"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    maha_title = "Mahalanobis" + (f" / {normalize}" if normalize != "none" else "")
    ell_title = "ellphi (tangency)" + (f" / {normalize}" if normalize != "none" else "")

    for ax, mat, title, stats, cmap in (
        (axes[0], dm_maha, maha_title, stats_m, "viridis"),
        (axes[1], dm_ell, ell_title, stats_e, "viridis"),
        (axes[2], ratio, "ellphi / Mahalanobis", stats_r, "coolwarm"),
    ):
        if cmap == "viridis":
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax)
        else:
            rlim = max(abs(stats_r["min"] - 1.0), abs(stats_r["max"] - 1.0), 0.05)
            im = ax.imshow(mat, cmap=cmap, vmin=1.0 - rlim, vmax=1.0 + rlim)
        ax.axhline(boundary, color="white", lw=0.8, alpha=0.9)
        ax.axvline(boundary, color="white", lw=0.8, alpha=0.9)
        ax.set_title(
            f"{title}\n"
            f"min={stats['min']:.3f}  med={stats['median']:.3f}  "
            f"mean={stats['mean']:.3f}  max={stats['max']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("point j")
        ax.set_ylabel("point i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    suptitle = (
        f"Anisotropic distance matrices (N={n}, inlier={n_in}, outlier={n_out})"
        f"{': ' + title_suffix if title_suffix else ''}{norm_note}\n"
        "Rows/cols sorted: inliers (blue region) then outliers (red region); "
        "white lines mark the inlier|outlier boundary."
    )
    fig.suptitle(suptitle, fontsize=11)

    # Label strip on the left panel
    strip = np.where(labels == 0, 0, 1)[None, :]
    inset = axes[0].inset_axes([-0.12, 0.0, 0.04, 1.0])
    inset.imshow(strip, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap([INLIER_COLOR, OUTLIER_COLOR]))
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_ylabel("GT", rotation=0, labelpad=12, va="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(
        f"  Mahalanobis  median={stats_m['median']:.4f}  mean={stats_m['mean']:.4f}\n"
        f"  ellphi       median={stats_e['median']:.4f}  mean={stats_e['mean']:.4f}\n"
        f"  ratio        median={stats_r['median']:.4f}  mean={stats_r['mean']:.4f}"
    )


def plot_point_cloud(
    points: np.ndarray,
    params: np.ndarray,
    labels: np.ndarray,
    *,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Two panels: GT-labeled scatter, and the same points with learned ellipses."""
    in_m = labels == 0
    out_m = labels == 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)

    ax0 = axes[0]
    ax0.scatter(points[in_m, 0], points[in_m, 1], c=INLIER_COLOR, s=14, label=f"Inlier ({in_m.sum()})")
    ax0.scatter(points[out_m, 0], points[out_m, 1], c=OUTLIER_COLOR, s=14, label=f"Outlier ({out_m.sum()})")
    ax0.set_title("GT labels")
    ax0.legend(loc="upper right", fontsize=8)

    ax1 = axes[1]
    ax1.scatter(points[in_m, 0], points[in_m, 1], c=INLIER_COLOR, s=10)
    ax1.scatter(points[out_m, 0], points[out_m, 1], c=OUTLIER_COLOR, s=10)
    t = np.linspace(0, 2 * np.pi, 60)
    for k in range(len(points)):
        a, b, theta = params[k]
        x_e = a * np.cos(t)
        y_e = b * np.sin(t)
        x_r = x_e * np.cos(theta) - y_e * np.sin(theta) + points[k, 0]
        y_r = x_e * np.sin(theta) + y_e * np.cos(theta) + points[k, 1]
        color = OUTLIER_COLOR if labels[k] == 1 else INLIER_COLOR
        ax1.plot(x_r, y_r, color=color, alpha=0.35, linewidth=0.8)
    ax1.set_title("Learned ellipses (GT-colored)")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

    fig.suptitle(f"Point cloud used for the distance matrices{': ' + title_suffix if title_suffix else ''}", fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="elongate_n100_no_cls_full120_ellphi_tuned")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "images" / "distance_matrix_maha_vs_ellphi.png",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--normalize",
        choices=["none", "median", "mean", "max"],
        default="none",
        help="Scale off-diagonal entries by median/mean/max of upper triangle (diagonal stays 0).",
    )
    p.add_argument(
        "--cloud-output",
        type=Path,
        default=None,
        help="Also save a scatter plot (GT labels + learned ellipses) of the point cloud.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)

    points, labels, dataset_idx = _build_val_dataset(config, args.sample_idx)
    model = load_model(device, weights_path=str(args.weights))
    with torch.no_grad():
        data_t = torch.as_tensor(points, dtype=torch.float32, device=device).unsqueeze(0)
        _logits, params_t = model(data_t)
        params = params_t.squeeze(0).cpu().numpy()

    points, labels, params = _sort_by_label(points, labels, params)
    title_suffix = f"val idx={dataset_idx}, weights={args.weights.parent.name}"
    plot_distance_matrices(
        points,
        params,
        labels,
        output_path=args.output,
        title_suffix=title_suffix,
        normalize=args.normalize,
    )
    if args.cloud_output is not None:
        plot_point_cloud(
            points,
            params,
            labels,
            output_path=args.cloud_output,
            title_suffix=title_suffix,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
