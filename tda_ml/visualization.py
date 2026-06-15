"""Training-time visualization helpers."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize(
    model,
    device,
    dataset,
    epoch,
    output_dir=".",
    title_prefix="",
    sample_indices=None,
    threshold=0.5,
):
    """
    Visualizes model predictions on 3 random samples from the dataset.
    """
    model.eval()
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Epoch {epoch} - {title_prefix} Results", fontsize=16)

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(3):
            if sample_indices is not None and i < len(sample_indices):
                idx = sample_indices[i]
            else:
                idx = torch.randint(0, len(dataset), (1,)).item()
            data, labels, clean_pc = dataset[idx]

            data_np = data.numpy()
            labels_np = labels.numpy()

            data_batch = data.to(device).unsqueeze(0)
            logits, params = model(data_batch)

            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            pred_labels = (probs > threshold).astype(int).flatten()

            params_np = params.squeeze(0).cpu().numpy()

            axes[i, 0].scatter(
                data_np[labels_np == 1, 0],
                data_np[labels_np == 1, 1],
                c="red",
                s=10,
                label="Outlier (GT)",
            )
            axes[i, 0].scatter(
                data_np[labels_np == 0, 0],
                data_np[labels_np == 0, 1],
                c="blue",
                s=10,
                label="Inlier (GT)",
            )
            axes[i, 0].set_title(f"Sample {i + 1}: GT Labels")
            axes[i, 0].legend()

            axes[i, 1].scatter(
                data_np[pred_labels == 1, 0],
                data_np[pred_labels == 1, 1],
                c="red",
                s=10,
                marker="x",
                label="Pred Outlier",
            )
            axes[i, 1].scatter(
                data_np[pred_labels == 0, 0],
                data_np[pred_labels == 0, 1],
                c="blue",
                s=10,
                label="Pred Inlier",
            )
            axes[i, 1].set_title(f"Sample {i + 1}: Prediction")
            axes[i, 1].legend()

            axes[i, 2].scatter(
                data_np[pred_labels == 1, 0],
                data_np[pred_labels == 1, 1],
                c="red",
                s=10,
                marker="x",
                label="Pred Outlier",
            )
            axes[i, 2].scatter(
                data_np[pred_labels == 0, 0],
                data_np[pred_labels == 0, 1],
                c="blue",
                s=10,
                label="Pred Inlier",
            )

            t = np.linspace(0, 2 * np.pi, 50)
            if len(data_np) > 0:
                for k in range(len(data_np)):
                    dx, dy, a, b, theta = params_np[k]

                    cx = data_np[k, 0] + dx
                    cy = data_np[k, 1] + dy

                    x_e = a * np.cos(t)
                    y_e = b * np.sin(t)
                    x_r = x_e * np.cos(theta) - y_e * np.sin(theta) + cx
                    y_r = x_e * np.sin(theta) + y_e * np.cos(theta) + cy

                    line_color = "blue" if pred_labels[k] == 0 else "red"
                    axes[i, 2].plot(x_r, y_r, color=line_color, alpha=0.3, linewidth=1)

            axes[i, 2].set_title(f"Sample {i + 1}: Predicted Inliers & Ellipses")
            for ax in axes[i]:
                ax.set_aspect("equal")
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = os.path.join(output_dir, f"{title_prefix}_result_epoch_{epoch}.png")
    plt.savefig(filename)
    plt.close()
