"""Training-time visualization helpers."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from tda_ml.run_paths import visualization_filename

INLIER_COLOR = "tab:blue"
OUTLIER_COLOR = "tab:red"


def _draw_ellipses(ax, points_np, params_np, gt_labels):
    """Overlay each point's learned ellipse, colored by its GT label."""
    t = np.linspace(0, 2 * np.pi, 50)
    for k in range(len(points_np)):
        a, b, theta = params_np[k]
        cx, cy = points_np[k, 0], points_np[k, 1]
        x_e = a * np.cos(t)
        y_e = b * np.sin(t)
        x_r = x_e * np.cos(theta) - y_e * np.sin(theta) + cx
        y_r = x_e * np.sin(theta) + y_e * np.cos(theta) + cy
        color = OUTLIER_COLOR if int(gt_labels[k]) == 1 else INLIER_COLOR
        ax.plot(x_r, y_r, color=color, alpha=0.35, linewidth=0.8)


def visualize(
    model,
    device,
    dataset,
    epoch,
    output_dir=".",
    title_prefix="",
    sample_indices=None,
    threshold=0.5,
    backend="ellphi",
    eps=None,
    min_samples=5,
):
    """Per sample cloud, draw a row of 2 panels.

    col 0  GT Labels             : points colored by ground-truth (inlier/outlier).
    col 1  GT + learned ellipses : same GT points, each point's learned ellipse drawn
                                   at true scale and colored by the point's GT label
                                   (the anisotropy-acquisition check).

    ``threshold``, ``backend``, ``eps``, and ``min_samples`` are kept for call-site
    compatibility (unused).
    """
    del threshold, backend, eps, min_samples

    model.eval()
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    fig.suptitle(f"Epoch {epoch} - {title_prefix} Results", fontsize=16)

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(3):
            if sample_indices is not None and i < len(sample_indices):
                idx = sample_indices[i]
            else:
                idx = torch.randint(0, len(dataset), (1,)).item()
            data, labels, _clean_pc = dataset[idx]

            data_np = data.numpy()
            labels_np = labels.numpy().astype(int).flatten()

            data_batch = data.to(device).unsqueeze(0)
            _logits, params = model(data_batch)
            params_np = params.squeeze(0).cpu().numpy()

            in_m = labels_np == 0
            out_m = labels_np == 1

            ax0 = axes[i, 0]
            ax0.scatter(data_np[in_m, 0], data_np[in_m, 1], c=INLIER_COLOR, s=10, label="Inlier (GT)")
            ax0.scatter(data_np[out_m, 0], data_np[out_m, 1], c=OUTLIER_COLOR, s=10, label="Outlier (GT)")
            ax0.set_title(f"Sample {i + 1}: GT Labels")
            ax0.legend(loc="upper right", fontsize=8)

            ax1 = axes[i, 1]
            ax1.scatter(data_np[in_m, 0], data_np[in_m, 1], c=INLIER_COLOR, s=8)
            ax1.scatter(data_np[out_m, 0], data_np[out_m, 1], c=OUTLIER_COLOR, s=8)
            _draw_ellipses(ax1, data_np, params_np, labels_np)
            ax1.set_title(f"Sample {i + 1}: Learned Ellipses (GT-colored)")

            for ax in axes[i]:
                ax.set_aspect("equal")
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = os.path.join(output_dir, visualization_filename(epoch))
    plt.savefig(filename)
    plt.close()
