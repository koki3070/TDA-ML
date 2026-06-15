"""
Non-official clustering demo (sparse sine wave + DBSCAN / ADBSCAN).

Requires a trained checkpoint (not shipped in this repository). Example:

  uv run python -m tda_ml.experiments.clustering_benchmark \\
    --checkpoint outputs/backend_compare/backend_mahalanobis_seed42_*/best_model.pth \\
    --output outputs/clustering_benchmark.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics import matthews_corrcoef

from tda_ml.checkpoint_io import extract_model_state_dict, load_torch_checkpoint
from tda_ml.config import model_kwargs_from_config
from tda_ml.dbscan import apply_anisotropic_dbscan
from tda_ml.experiments.run_noise_sensitivity import generate_sparse_sine_wave
from tda_ml.metrics import compute_recall_specificity_gmean_mcc
from tda_ml.models import AnisotropicOutlierClassifier


def calculate_metrics(labels_gt, labels_pred):
    return compute_recall_specificity_gmean_mcc(labels_gt, labels_pred)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_model.pth or final_model.pth from a training run.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="outputs/clustering_benchmark.png",
        help="Path for the comparison figure.",
    )
    p.add_argument("--min-b", type=float, default=0.3, help="Minimum minor axis clamp.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {checkpoint_path}...")
    model = AnisotropicOutlierClassifier(**model_kwargs_from_config({"model": {}}))
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(extract_model_state_dict(checkpoint), strict=False)
    model.to(device)
    model.eval()

    print("Generating sparse test data...")
    n_signal = 30
    n_noise = 60
    data_norm, gt_labels, _data_raw = generate_sparse_sine_wave(
        n_signal=n_signal, n_noise=n_noise, noise_seed=42
    )

    data_tensor = torch.tensor(data_norm, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, params = model(data_tensor)

    ellipse_params_np = params.cpu().squeeze(0).numpy()
    ellipse_params_np[:, 3] = np.maximum(ellipse_params_np[:, 3], args.min_b)

    best_euc_mcc = -1
    best_euc_preds = None
    best_euc_eps = 0

    for eps in [0.15, 0.2, 0.25, 0.3]:
        db = DBSCAN(eps=eps, min_samples=4)
        preds = db.fit_predict(data_norm)
        mcc = matthews_corrcoef(gt_labels, (preds == -1).astype(int))
        if mcc > best_euc_mcc:
            best_euc_mcc = mcc
            best_euc_preds = preds
            best_euc_eps = eps

    print(f"Running ADBSCAN sweep (min_b={args.min_b}, min_samples=4)...")
    best_ell_mcc = -1
    best_ell_preds = None
    best_ell_eps = 0

    for eps in [0.4, 0.6, 0.8, 1.0]:
        labels = apply_anisotropic_dbscan(
            data_norm,
            ellipse_params_np,
            eps=eps,
            min_samples=4,
            metric="max",
        )
        mcc = matthews_corrcoef(gt_labels, (labels == -1).astype(int))
        if mcc > best_ell_mcc:
            best_ell_mcc = mcc
            best_ell_preds = labels
            best_ell_eps = eps

    r_euc, s_euc, g_euc, m_euc = calculate_metrics(
        gt_labels, (best_euc_preds == -1).astype(int)
    )
    r_ell, s_ell, g_ell, m_ell = calculate_metrics(
        gt_labels, (best_ell_preds == -1).astype(int)
    )

    print("\n--- Benchmark Results ---")
    print(f"Euclidean DBSCAN (eps={best_euc_eps}): MCC={m_euc:.4f}, G-Mean={g_euc:.4f}")
    print(f"ADBSCAN (eps={best_ell_eps}): MCC={m_ell:.4f}, G-Mean={g_ell:.4f}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(data_norm[gt_labels == 0, 0], data_norm[gt_labels == 0, 1], c="blue", label="Signal")
    plt.scatter(
        data_norm[gt_labels == 1, 0],
        data_norm[gt_labels == 1, 1],
        c="red",
        alpha=0.3,
        label="Noise",
    )
    plt.title("Ground Truth")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter(data_norm[:, 0], data_norm[:, 1], c=best_euc_preds, cmap="tab20")
    plt.title(f"Euclidean DBSCAN (eps={best_euc_eps})\nMCC={m_euc:.3f}")

    plt.subplot(1, 3, 3)
    plt.scatter(data_norm[:, 0], data_norm[:, 1], c=best_ell_preds, cmap="tab20")
    plt.title(f"ADBSCAN (eps={best_ell_eps})\nMCC={m_ell:.3f}")

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    print(f"\nComparison image saved to {out}")


if __name__ == "__main__":
    main()
