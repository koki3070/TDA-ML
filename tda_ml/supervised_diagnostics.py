"""Diagnostics when supervised training MCC stays near zero (issue #59 verification)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tda_ml.distance_backend import compute_distance_matrix_batch
from tda_ml.metrics import compute_recall_specificity_gmean_mcc
from tda_ml.numerical_eps import NUMERICAL_EPS
from tda_ml.topology import compute_anisotropic_metric


def git_revision(repo_root: Path | None = None) -> str:
    """Return HEAD SHA, with ``-dirty`` when the worktree has local changes."""
    root = repo_root or Path(__file__).resolve().parents[1]
    try:
        head = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if porcelain.strip():
            return f"{head}-dirty"
        return head
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _finite_stats(t: torch.Tensor) -> dict[str, float | int]:
    flat = t.detach().float().reshape(-1)
    finite = torch.isfinite(flat)
    n = int(flat.numel())
    n_finite = int(finite.sum().item())
    if n_finite == 0:
        return {
            "count": n,
            "finite_count": 0,
            "min": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "nan_count": n - n_finite,
        }
    vals = flat[finite]
    return {
        "count": n,
        "finite_count": n_finite,
        "min": float(vals.min().item()),
        "median": float(vals.median().item()),
        "max": float(vals.max().item()),
        "nan_count": n - n_finite,
    }


def collect_ellipse_stats(params: torch.Tensor) -> dict[str, Any]:
    if params.shape[-1] not in (3, 5):
        raise ValueError(f"params last dim must be 3 or 5; got {params.shape}")
    axes = params[..., 2:4] if params.shape[-1] == 5 else params[..., 0:2]
    a = axes[..., 0]
    b = axes[..., 1]
    major = torch.maximum(a, b)
    minor = torch.minimum(a, b)
    ratio = major / (minor + NUMERICAL_EPS)
    return {
        "a": _finite_stats(a),
        "b": _finite_stats(b),
        "major": _finite_stats(major),
        "minor": _finite_stats(minor),
        "aspect_ratio": _finite_stats(ratio),
        "frac_below_1e-4": float((axes < 1e-4).float().mean().item()),
        "frac_below_1e-3": float((axes < 1e-3).float().mean().item()),
    }


def collect_classification_health(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, Any]:
    probs = torch.sigmoid(logits.squeeze(-1)).detach().float()
    labels_flat = labels.detach().float().reshape(-1)
    preds = (probs > threshold).long()
    pred_np = preds.reshape(-1).cpu().numpy()
    label_np = labels_flat.cpu().numpy()
    recall, specificity, gmean, mcc = compute_recall_specificity_gmean_mcc(
        label_np, pred_np
    )
    return {
        "logits": _finite_stats(logits),
        "probs": _finite_stats(probs),
        "pred_outlier_fraction": float(pred_np.mean()),
        "label_outlier_fraction": float(label_np.mean()),
        "recall": float(recall),
        "specificity": float(specificity),
        "gmean": float(gmean),
        "mcc": float(mcc),
        "threshold": threshold,
    }


def collect_encoder_health(model, data: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        _, _, base_angle, base_axes = model.encoder(data)
    ratio = base_axes[..., 0] / (base_axes[..., 1] + NUMERICAL_EPS)
    return {
        "base_axes_major": _finite_stats(base_axes[..., 0]),
        "base_axes_minor": _finite_stats(base_axes[..., 1]),
        "base_pca_aspect_ratio": _finite_stats(ratio),
        "base_angle": _finite_stats(base_angle),
        "frac_base_minor_below_0.2": float(
            (base_axes[..., 1] < 0.2).float().mean().item()
        ),
    }


def collect_metric_tensor_health(params: torch.Tensor) -> dict[str, Any]:
    metric_params = params[..., 2:5] if params.shape[-1] == 5 else params
    m00, m11, m01 = compute_anisotropic_metric(
        metric_params[..., 0:2], metric_params[..., 2:3]
    )
    return {
        "m00": _finite_stats(m00),
        "m11": _finite_stats(m11),
        "m01": _finite_stats(m01),
    }


def collect_distance_health(
    points: torch.Tensor,
    params: torch.Tensor,
    logits: torch.Tensor,
    *,
    distance_backend: str,
    ellphi_differentiable: bool,
    threshold: float,
) -> dict[str, Any]:
    probs = torch.sigmoid(logits.squeeze(-1))
    d = compute_distance_matrix_batch(
        points,
        params,
        probs=probs,
        symmetrize="max",
        backend=distance_backend,
        ellphi_differentiable=ellphi_differentiable,
    )
    d0 = d[0]
    sym_err = (d0 - d0.T).abs().max().item()
    off_diag = d0[~torch.eye(d0.shape[0], dtype=torch.bool, device=d0.device)]
    return {
        "distance_matrix": _finite_stats(d0),
        "off_diagonal": _finite_stats(off_diag),
        "max_symmetry_error": float(sym_err),
        "backend": distance_backend,
    }


def _first_val_batch(val_loader, device: torch.device):
    for data, labels, clean_pc in val_loader:
        return (
            data.to(device),
            labels.to(device),
            clean_pc.to(device),
        )
    raise RuntimeError("validation loader is empty")


def infer_failure_hypotheses(report: dict[str, Any]) -> list[dict[str, str]]:
    hypotheses: list[dict[str, str]] = []
    cls = report.get("classification", {})
    ell = report.get("ellipse", {})
    enc = report.get("encoder", {})
    dist = report.get("distance", {})
    hist = report.get("metrics_history", [])

    if cls.get("pred_outlier_fraction", 0.0) > 0.95:
        hypotheses.append(
            {
                "id": "all_outlier_predictions",
                "detail": (
                    "Almost all points classified as outlier "
                    f"(fraction={cls['pred_outlier_fraction']:.3f}). "
                    "Check BCE signal, threshold, logit scale."
                ),
            }
        )
    if cls.get("pred_outlier_fraction", 0.0) < 0.05:
        hypotheses.append(
            {
                "id": "all_inlier_predictions",
                "detail": (
                    "Almost no outlier predictions; specificity may be high but "
                    "recall near zero."
                ),
            }
        )

    frac_tiny = ell.get("frac_below_1e-4", 0.0)
    if frac_tiny > 0.1:
        hypotheses.append(
            {
                "id": "axis_collapse",
                "detail": (
                    f"{frac_tiny:.1%} of axis values below 1e-4 "
                    "(fixpullback-style collapse). Check w_size / removed encoder floors."
                ),
            }
        )

    a_max = ell.get("a", {}).get("max", 0.0)
    if a_max > 100.0:
        hypotheses.append(
            {
                "id": "axis_explosion",
                "detail": f"Large axis lengths detected (a_max={a_max:.3g}). Check w_aniso / w_size.",
            }
        )

    if dist.get("distance_matrix", {}).get("nan_count", 0) > 0:
        hypotheses.append(
            {
                "id": "non_finite_distances",
                "detail": "Distance matrix contains NaN/Inf; inspect metric tensor and axes.",
            }
        )

    logit_std = cls.get("logits", {}).get("max", 0.0) - cls.get("logits", {}).get("min", 0.0)
    if logit_std < 0.05:
        hypotheses.append(
            {
                "id": "logit_collapse",
                "detail": (
                    f"Logit dynamic range ~{logit_std:.4f}; classifier head may not be learning."
                ),
            }
        )

    if hist:
        val_mccs = [float(row.get("val_mcc", 0.0)) for row in hist]
        if max(val_mccs) < 0.05:
            hypotheses.append(
                {
                    "id": "mcc_never_rose",
                    "detail": (
                        f"val_mcc stayed below 0.05 for all {len(val_mccs)} logged epochs "
                        f"(max={max(val_mccs):.4f})."
                    ),
                }
            )

    minor_med = enc.get("base_axes_minor", {}).get("median", float("nan"))
    if minor_med == minor_med and minor_med < 0.05:
        hypotheses.append(
            {
                "id": "degenerate_pca_minor_axis",
                "detail": (
                    f"Encoder PCA minor axis median={minor_med:.4g}; "
                    "local neighborhoods may be nearly collinear."
                ),
            }
        )

    if not hypotheses:
        hypotheses.append(
            {
                "id": "unclear",
                "detail": (
                    "MCC remained low but no single dominant failure mode was flagged. "
                    "Inspect metrics_history and saved checkpoint."
                ),
            }
        )
    return hypotheses


def run_abort_diagnostics(
    *,
    trainer,
    model,
    val_loader,
    device: torch.device,
    epoch: int,
    metrics_history: list[dict[str, Any]],
    abort_reason: str,
) -> dict[str, Any]:
    data, labels, _clean_pc = _first_val_batch(val_loader, device)
    with torch.no_grad():
        logits, params = model(data)

    report: dict[str, Any] = {
        "status": "early-aborted",
        "abort_reason": abort_reason,
        "abort_epoch": epoch,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_id": trainer.config["meta"].get("config_id"),
        "distance_backend": trainer.distance_backend,
        "metrics_history": metrics_history,
        "classification": collect_classification_health(
            logits, labels, threshold=trainer.threshold
        ),
        "ellipse": collect_ellipse_stats(params),
        "encoder": collect_encoder_health(model, data),
        "metric_tensor": collect_metric_tensor_health(params),
        "distance": collect_distance_health(
            data,
            params,
            logits,
            distance_backend=trainer.distance_backend,
            ellphi_differentiable=trainer.ellphi_differentiable,
            threshold=trainer.threshold,
        ),
    }
    report["hypotheses"] = infer_failure_hypotheses(report)
    return report


def write_abort_report(log_dir: str | Path, report: dict[str, Any]) -> Path:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / "issue59_abort_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    md_path = log_dir / "issue59_abort_report.md"
    lines = [
        "# Supervised training early abort report",
        "",
        f"- **status**: {report.get('status')}",
        f"- **reason**: {report.get('abort_reason')}",
        f"- **epoch**: {report.get('abort_epoch')}",
        f"- **backend**: {report.get('distance_backend')}",
        "",
        "## Hypotheses",
        "",
    ]
    for h in report.get("hypotheses", []):
        lines.append(f"- **{h['id']}**: {h['detail']}")
    lines.extend(["", "## Classification", ""])
    cls = report.get("classification", {})
    lines.append(
        f"- pred_outlier_fraction={cls.get('pred_outlier_fraction')}, "
        f"mcc={cls.get('mcc')}, recall={cls.get('recall')}, "
        f"specificity={cls.get('specificity')}"
    )
    lines.extend(["", "## Ellipse axes", ""])
    ell = report.get("ellipse", {})
    lines.append(
        f"- a_median={ell.get('a', {}).get('median')}, "
        f"b_median={ell.get('b', {}).get('median')}, "
        f"frac_below_1e-4={ell.get('frac_below_1e-4')}"
    )
    lines.extend(["", "## Metrics history", ""])
    for row in report.get("metrics_history", []):
        lines.append(
            f"- epoch {row.get('epoch')}: val_mcc={row.get('val_mcc')}, "
            f"train_mcc={row.get('train_mcc')}, val_recall={row.get('val_recall')}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path


def should_early_abort(
    *,
    epoch: int,
    best_val_mcc: float,
    val_recall: float,
    val_mcc: float,
    train_mcc: float,
    early_abort_cfg: dict[str, Any],
) -> tuple[bool, str]:
    if not early_abort_cfg.get("enabled", False):
        return False, ""
    probe_epochs = int(early_abort_cfg.get("probe_epochs", 3))
    if epoch < probe_epochs:
        return False, ""

    min_val_mcc = float(early_abort_cfg.get("min_val_mcc", 0.05))
    min_train_mcc = float(early_abort_cfg.get("min_train_mcc", 0.02))
    max_val_recall = float(early_abort_cfg.get("max_val_recall_for_abort", 0.99))

    if best_val_mcc < min_val_mcc and train_mcc < min_train_mcc:
        return True, (
            f"after epoch {epoch}: best_val_mcc={best_val_mcc:.4f} < {min_val_mcc} "
            f"and train_mcc={train_mcc:.4f} < {min_train_mcc}"
        )
    if val_mcc < min_val_mcc and val_recall >= max_val_recall:
        return True, (
            f"after epoch {epoch}: val_mcc={val_mcc:.4f} with val_recall={val_recall:.4f} "
            f"(likely all-outlier predictions)"
        )
    return False, ""
