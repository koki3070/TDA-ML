#!/usr/bin/env python3
"""
Paper-aligned evaluation: mahalanobis DBSCAN inference + MCC / G-Mean / topo W-Dist.

W-Dist matches ``TopologicalLoss``: learned ellipses on the full noisy cloud vs
clean teacher PD (local PCA + ellphi by default). DBSCAN affects MCC only.

Usage::

    uv run python experiments/evaluate_paper_protocol.py \\
        --run-dir outputs/supervised/.../pwr_s42_<stamp> \\
        --base-config elongate_n100_no_cls_full120_teacher_local_pca \\
        --split val

    uv run python experiments/evaluate_paper_protocol.py \\
        --run-dir ... --split test \\
        --dbscan-hparams outputs/.../logs/dbscan_hparams.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tda_ml.checkpoint_io import extract_model_state_dict, load_torch_checkpoint
from tda_ml.config import deep_update, load_config, model_kwargs_from_config
from tda_ml.data_loader import NoisyMNISTDataset, create_data_loader
from tda_ml.dbscan_eval import evaluate_model_grid, iter_cloud_predictions
from tda_ml.metrics import compute_recall_specificity_gmean_mcc_wdist
from tda_ml.models import AnisotropicOutlierClassifier
from tda_ml.preflight import preflight_paper_eval_run_dir
from tda_ml.reproducibility import reproducibility_settings
from tda_ml.seed_utils import set_global_seed
from tda_ml.supervised_diagnostics import git_revision
from tda_ml.topo_wdist import TopoWdistOptions, topo_wdist_options_from_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CloudMetrics:
    recall: float
    specificity: float
    gmean: float
    mcc: float
    wdist: float


@dataclass
class SplitMetrics:
    split: str
    n_clouds: int
    recall: float
    specificity: float
    gmean: float
    mcc: float
    wdist: float
    dbscan_eps: float | None = None
    dbscan_min_samples: int | None = None
    backend: str = "ellphi"


def _valid_clean_inliers(clean_pc: np.ndarray) -> np.ndarray:
    mask = np.abs(clean_pc).sum(axis=1) > 1e-6
    return clean_pc[mask]


def dbscan_labels_to_outlier_pred(labels: np.ndarray) -> np.ndarray:
    """DBSCAN noise (-1) -> outlier (1); clustered points -> inlier (0)."""
    return (labels == -1).astype(np.int64)


def evaluate_cloud_dbscan(
    points: np.ndarray,
    params: np.ndarray,
    labels_gt: np.ndarray,
    clean_pc: np.ndarray,
    *,
    eps: float,
    min_samples: int,
    backend: str = "ellphi",
    metric: str = "max",
    topo_options: TopoWdistOptions | None = None,
) -> CloudMetrics:
    from tda_ml.dbscan import apply_anisotropic_dbscan

    db_labels = apply_anisotropic_dbscan(
        points,
        params,
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        backend=backend,
    )
    pred = dbscan_labels_to_outlier_pred(db_labels)
    recall, specificity, gmean, mcc, wdist = compute_recall_specificity_gmean_mcc_wdist(
        labels_gt,
        pred,
        points=points,
        params=params,
        clean_pc=clean_pc,
        topo_options=topo_options,
    )
    return CloudMetrics(recall, specificity, gmean, mcc, wdist)


def _aggregate_cloud_metrics(rows: list[CloudMetrics]) -> tuple[float, float, float, float, float]:
    if not rows:
        raise ValueError("No clouds to aggregate")
    return (
        float(np.mean([r.recall for r in rows])),
        float(np.mean([r.specificity for r in rows])),
        float(np.mean([r.gmean for r in rows])),
        float(np.mean([r.mcc for r in rows])),
        float(np.mean([r.wdist for r in rows])),
    )


def build_split_loader(config: dict[str, Any], split: str, device: torch.device):
    data_cfg = config["data"]
    seed = int(data_cfg.get("seed", 42))
    set_global_seed(seed, deterministic_algorithms=False)

    train_size = int(data_cfg.get("train_size", 4500))
    val_size = int(data_cfg.get("val_size", 500))
    test_size = int(data_cfg.get("test_size", 1000))
    generator = torch.Generator().manual_seed(seed)
    full_train_indices = torch.randperm(60000, generator=generator)[: train_size + val_size]
    val_indices = full_train_indices[train_size:]
    test_indices = torch.randperm(10000, generator=generator)[:test_size]

    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", device.type == "cuda"))
    batch_size = int(data_cfg.get("batch_size", 64))

    if split == "val":
        indices = val_indices
        train_flag = True
    elif split == "test":
        indices = test_indices
        train_flag = False
    else:
        raise ValueError(f"split must be 'val' or 'test'; got {split!r}")

    if "outlier_mode" not in data_cfg:
        raise ValueError(
            "data.outlier_mode must be set explicitly (uniform|local_pca_tangent); "
            "refusing silent uniform default"
        )
    outlier_mode = str(data_cfg["outlier_mode"]).strip().lower()
    if outlier_mode not in ("uniform", "local_pca_tangent"):
        raise ValueError(
            f"data.outlier_mode must be 'uniform' or 'local_pca_tangent', got {outlier_mode!r}"
        )
    if "noise_std" not in data_cfg:
        raise ValueError("data.noise_std must be set explicitly; refusing silent default")

    dataset_kwargs: dict[str, Any] = dict(
        root=str(REPO_ROOT / "data"),
        train=train_flag,
        max_points=data_cfg["max_points"],
        num_outliers=data_cfg["num_outliers"],
        noise_std=float(data_cfg["noise_std"]),
        indices=indices,
        deterministic=True,
        noise_seed=seed,
        preload=True,
        outlier_mode=outlier_mode,
    )
    if outlier_mode == "local_pca_tangent":
        for key in (
            "tangent_pca_k",
            "tangent_offset_min",
            "tangent_offset_max",
            "tangent_angle_jitter_deg",
            "tangent_stroke_clearance",
        ):
            if key not in data_cfg:
                raise ValueError(
                    f"data.{key} must be set explicitly for outlier_mode=local_pca_tangent"
                )
        dataset_kwargs.update(
            tangent_pca_k=int(data_cfg["tangent_pca_k"]),
            tangent_offset_min=float(data_cfg["tangent_offset_min"]),
            tangent_offset_max=float(data_cfg["tangent_offset_max"]),
            tangent_angle_jitter_deg=float(data_cfg["tangent_angle_jitter_deg"]),
            tangent_stroke_clearance=float(data_cfg["tangent_stroke_clearance"]),
        )
    dataset = NoisyMNISTDataset(**dataset_kwargs)
    loader = create_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader


def load_model_from_run(
    run_dir: Path,
    config: dict[str, Any],
    device: torch.device,
    checkpoint_name: str = "best_model.pth",
) -> AnisotropicOutlierClassifier:
    ckpt_path = run_dir / checkpoint_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    model = AnisotropicOutlierClassifier(**model_kwargs_from_config(config))
    ckpt = load_torch_checkpoint(str(ckpt_path), map_location="cpu")
    model.load_state_dict(extract_model_state_dict(ckpt), strict=False)
    model.to(device)
    model.eval()
    return model



def evaluate_split(
    run_dir: Path,
    config: dict[str, Any],
    split: str,
    device: torch.device,
    *,
    eps: float | None = None,
    min_samples: int | None = None,
    eps_values: list[float] | None = None,
    min_samples_values: list[int] | None = None,
    backend: str = "ellphi",
    checkpoint_name: str = "best_model.pth",
    tag: str | None = None,
) -> SplitMetrics:
    loader = build_split_loader(config, split, device)
    model = load_model_from_run(run_dir, config, device, checkpoint_name=checkpoint_name)
    topo_options = topo_wdist_options_from_config(config)
    rep = reproducibility_settings(config)
    log_dir = run_dir / "logs"
    tag_suffix = f"_{tag}" if tag else ""

    if split == "val":
        grid = evaluate_model_grid(
            model,
            loader,
            device,
            config=config,
            backend=backend,
            eps_values=eps_values,
            min_samples_values=min_samples_values,
            objective="mcc",
            topo_options=topo_options,
            allow_skip_degenerate_grid_cells=rep["allow_skip_degenerate_grid_cells"],
            grid_log_path=log_dir / f"dbscan_grid_log_val{tag_suffix}.json",
        )
        eps = grid.eps
        min_samples = grid.min_samples
        return SplitMetrics(
            split=split,
            n_clouds=grid.n_clouds,
            recall=grid.recall,
            specificity=grid.specificity,
            gmean=grid.gmean,
            mcc=grid.mcc,
            wdist=grid.wdist,
            dbscan_eps=float(eps),
            dbscan_min_samples=int(min_samples),
            backend=backend,
        )

    if eps is None or min_samples is None:
        raise ValueError("test split requires eps and min_samples (from val tuning)")

    clouds = list(iter_cloud_predictions(model, loader, device))
    per_cloud: list[CloudMetrics] = []
    for points, params, labels_gt, clean_pc in clouds:
        per_cloud.append(
            evaluate_cloud_dbscan(
                points,
                params,
                labels_gt,
                clean_pc,
                eps=float(eps),
                min_samples=int(min_samples),
                backend=backend,
                topo_options=topo_options,
            )
        )
    recall, specificity, gmean, mcc, wdist = _aggregate_cloud_metrics(per_cloud)
    return SplitMetrics(
        split=split,
        n_clouds=len(per_cloud),
        recall=recall,
        specificity=specificity,
        gmean=gmean,
        mcc=mcc,
        wdist=wdist,
        dbscan_eps=float(eps),
        dbscan_min_samples=int(min_samples),
        backend=backend,
    )


def load_run_config(run_dir: Path, base_config: str, seed: int | None) -> dict[str, Any]:
    manifest_path = run_dir / "logs" / "run_manifest.json"
    cfg = load_config(base_config, project_root=REPO_ROOT)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if seed is None:
            seed = manifest.get("seed")
        loss_overrides = manifest.get("loss_overrides") or {}
        if loss_overrides:
            topo_patch: dict[str, Any] = {}
            if "homology_dimensions" in loss_overrides:
                topo_patch["homology_dimensions"] = loss_overrides["homology_dimensions"]
            loss_patch = {
                key: loss_overrides[key]
                for key in (
                    "aniso_mode",
                    "size_mode",
                    "size_ref",
                    "size_power",
                    "w_topo",
                    "w_aniso",
                    "w_size",
                )
                if key in loss_overrides
            }
            patch: dict[str, Any] = {}
            if loss_patch:
                patch["loss"] = loss_patch
            if topo_patch:
                patch["model"] = {"topology_loss": topo_patch}
            if patch:
                cfg = deep_update(cfg, patch)
        contract = manifest.get("paper_no_cls_contract")
        if isinstance(contract, dict):
            # Prefer method fields recorded at train time when present.
            loss_c = {
                key: contract[key]
                for key in (
                    "teacher_mode",
                    "aniso_mode",
                    "size_mode",
                    "w_class",
                    "teacher_local_pca_k",
                    "teacher_local_pca_normalize_axes",
                )
                if key in contract
            }
            topo_c = {
                key: contract[key]
                for key in ("homology_dimensions", "prob_weighting", "distance_backend")
                if key in contract
            }
            patch = {}
            if loss_c:
                patch["loss"] = loss_c
            if topo_c:
                patch["model"] = {"topology_loss": topo_c}
            if patch:
                cfg = deep_update(cfg, patch)
    if seed is not None:
        cfg = deep_update(cfg, {"data": {"seed": int(seed)}})
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--base-config",
        type=str,
        default="elongate_n100_no_cls_full120_teacher_local_pca",
        help="YAML used when run_manifest lacks method overrides (paper no_cls default).",
    )
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--seed", type=int, default=None, help="Override data.seed (else run_manifest)")
    p.add_argument(
        "--backend",
        type=str,
        default="mahalanobis",
        choices=["ellphi", "mahalanobis"],
        help="DBSCAN backend for MCC (paper protocol default: mahalanobis).",
    )
    p.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_model.pth",
        help="Checkpoint filename inside run-dir (e.g. final_model.pth).",
    )
    p.add_argument(
        "--eps-values",
        type=float,
        nargs="+",
        default=None,
        help="Override DBSCAN eps grid (default: evaluation.dbscan.eps_values from config).",
    )
    p.add_argument(
        "--min-samples-values",
        type=int,
        nargs="+",
        default=None,
        help="Override DBSCAN min_samples grid (default: evaluation.dbscan from config).",
    )
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help=(
            "Optional suffix for output files (dbscan_hparams_<tag>.json, "
            "paper_metrics_<split>_<tag>.json) to avoid clobbering originals."
        ),
    )
    p.add_argument(
        "--dbscan-hparams",
        type=Path,
        default=None,
        help="JSON with eps/min_samples for test split",
    )
    p.add_argument("--out-json", type=Path, default=None, help="Write metrics JSON (default: run_dir/logs/)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    preflight_paper_eval_run_dir(run_dir, checkpoint_name=args.checkpoint_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_run_config(run_dir, args.base_config, args.seed)

    tag_suffix = f"_{args.tag}" if args.tag else ""
    hparams_name = f"dbscan_hparams{tag_suffix}.json"

    eps = min_samples = None
    if args.split == "test":
        if args.dbscan_hparams is None:
            args.dbscan_hparams = run_dir / "logs" / hparams_name
        if not args.dbscan_hparams.is_file():
            raise FileNotFoundError(f"DBSCAN hparams JSON not found: {args.dbscan_hparams}")
        hparams = json.loads(args.dbscan_hparams.read_text())
        eps = float(hparams["eps"])
        min_samples = int(hparams["min_samples"])

    metrics = evaluate_split(
        run_dir,
        config,
        args.split,
        device,
        eps=eps,
        min_samples=min_samples,
        eps_values=args.eps_values,
        min_samples_values=args.min_samples_values,
        backend=args.backend,
        checkpoint_name=args.checkpoint_name,
        tag=args.tag,
    )

    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "val":
        hparams_path = log_dir / hparams_name
        hparams_path.write_text(
            json.dumps(
                {
                    "eps": metrics.dbscan_eps,
                    "min_samples": metrics.dbscan_min_samples,
                    "backend": metrics.backend,
                    "checkpoint_name": args.checkpoint_name,
                    "mean_val_mcc_dbscan": metrics.mcc,
                    "selection": "max mean cloud MCC on validation split",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Saved DBSCAN hparams: {hparams_path}")

    out_path = args.out_json or log_dir / f"paper_metrics_{args.split}{tag_suffix}.json"
    payload = {
        "source_revision": git_revision(REPO_ROOT),
        "run_dir": str(run_dir),
        "split": args.split,
        "checkpoint_name": args.checkpoint_name,
        **asdict(metrics),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
