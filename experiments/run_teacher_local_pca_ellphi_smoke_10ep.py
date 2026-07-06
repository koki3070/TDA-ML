#!/usr/bin/env python3
"""10-epoch smoke: ellphi + local_pca teacher. Logs init Wasserstein before training."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch_topological.nn import VietorisRipsComplex, WassersteinDistance

from tda_ml.config import deep_update, load_config, model_kwargs_from_config
from tda_ml.data_loader import NoisyMNISTDataset, create_data_loader
from tda_ml.distance_backend import compute_distance_matrix_batch
from tda_ml.main import main as train_main
from tda_ml.run_setup import configure_torch_runtime, resolve_dataloader_settings
from tda_ml.models import AnisotropicOutlierClassifier
from tda_ml.seed_utils import set_global_seed
from tda_ml.trainer import Trainer

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def _init_wasserstein(
    trainer: Trainer,
    loader,
    device: torch.device,
) -> dict[str, float]:
    """Mean init Wasserstein^2 between pred PD and teacher PD (one train batch)."""
    model = trainer.model
    model.eval()
    vr = VietorisRipsComplex(dim=1)
    wdist_fn = WassersteinDistance(q=2)
    topo_fn = trainer.topo_loss_fn

    data, _labels, clean_pc = next(iter(loader))
    data = data.to(device)
    clean_pc = clean_pc.to(device)

    with torch.no_grad():
        logits, params = model(data)
        clean_pd, clean_scales = trainer._compute_clean_pd_info(clean_pc)

    per_sample: list[float] = []
    with torch.no_grad():
        for i in range(data.shape[0]):
            pts_i, par_i, logits_i = topo_fn._subsample_points(
                data[i], params[i], logits[i]
            )
            probs_i = (
                torch.sigmoid(logits_i).squeeze(-1)
                if topo_fn.prob_weighting
                else None
            )
            d_batch = compute_distance_matrix_batch(
                pts_i.unsqueeze(0),
                par_i.unsqueeze(0),
                probs=probs_i.unsqueeze(0) if probs_i is not None else None,
                symmetrize="max",
                backend=topo_fn.distance_backend,
                ellphi_differentiable=topo_fn.ellphi_differentiable,
            )
            clean_scale_i = clean_scales[i] if clean_scales is not None else None
            d_mat = topo_fn._rescale_distance_matrix(d_batch[0], clean_scale=clean_scale_i)
            pd_pred = vr(d_mat, treat_as_distances=True)
            w2 = float(wdist_fn(pd_pred, clean_pd[i]) ** 2)
            per_sample.append(w2)

    return {
        "n_samples": len(per_sample),
        "wasserstein2_mean": float(sum(per_sample) / len(per_sample)),
        "wasserstein2_min": float(min(per_sample)),
        "wasserstein2_max": float(max(per_sample)),
        "teacher_mode": trainer.teacher_mode,
        "distance_backend": trainer.distance_backend,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="configs/elongate_n100_no_cls_full120_teacher_local_pca.yaml",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-base",
        type=Path,
        default=REPO_ROOT / "outputs/supervised/20260705_teacher_local_pca_ellphi_10ep",
    )
    p.add_argument("--backend", default="ellphi", choices=["ellphi", "mahalanobis"])
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    set_global_seed(args.seed)

    cfg = load_config(args.config, project_root=REPO_ROOT)
    cfg = deep_update(
        cfg,
        {
            "training": {"epochs": args.epochs},
            "data": {"seed": args.seed},
            "model": {
                "topology_loss": {
                    "distance_backend": args.backend,
                    "prob_weighting": False,
                }
            },
            "loss": {
                "teacher_mode": "local_pca",
                "w_class": 0.0,
            },
            "outputs": {"base_dir": str(args.out_base)},
            "meta": {"config_id": f"teacher_local_pca_{args.backend}_seed{args.seed}"},
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_torch_runtime(cfg, device)
    num_workers, pin_memory, persistent_workers, prefetch_factor = resolve_dataloader_settings(
        cfg, device
    )
    data_cfg = cfg["data"]
    train_ds = NoisyMNISTDataset(
        root="./data",
        train=True,
        num_samples=data_cfg["train_size"],
        max_points=data_cfg["max_points"],
        num_outliers=data_cfg["num_outliers"],
        noise_std=data_cfg["noise_std"],
        deterministic=True,
        noise_seed=data_cfg["seed"],
    )
    loader = create_data_loader(
        train_ds,
        batch_size=min(8, data_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    model = AnisotropicOutlierClassifier(**model_kwargs_from_config(cfg)).to(device)
    trainer = Trainer(model, cfg, device=device)
    init_stats = _init_wasserstein(trainer, loader, device)
    logger.info("Init teacher Wasserstein^2: %s", json.dumps(init_stats, indent=2))

    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    init_path = out_base / f"init_wasserstein_{args.backend}_seed{args.seed}.json"
    init_path.write_text(json.dumps(init_stats, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", init_path)

    train_main(config=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
