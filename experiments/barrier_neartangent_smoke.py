#!/usr/bin/env python3
"""Degeneracy-guard smoke: rerun a failed near-tangent tune condition with elongate_barrier.

Motivation: on near-tangent data, 15/16 stage-2 (20ep) conditions of the plain
``elongate`` stack hard-failed in ellphi (minor axes collapsed to ~1e-5, aspect
~350 by epoch 15). This smoke reruns one of those exact failed weight sets on
the ``elongate_barrier`` base config for the full 20 epochs and reports the
final ellipse geometry, so the guard is validated on the worst case before
spending hours on a fresh Optuna study.

Usage::

    uv run python experiments/barrier_neartangent_smoke.py \
        --w-topo 0.201 --w-aniso 0.107 --w-size 0.179 --lr 0.000205 \
        --out-base outputs/supervised_no_cls/0719_barrier_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from tda_ml.checkpoint_io import resolve_val_topo_checkpoint  # noqa: E402
from tda_ml.main import main as train_main  # noqa: E402
from tda_ml.supervised_diagnostics import git_revision  # noqa: E402
from tda_ml.topo_wdist import topo_wdist_options_from_config  # noqa: E402

from evaluate_paper_protocol import (  # noqa: E402
    build_split_loader,
    iter_cloud_predictions,
    load_model_from_run,
)
from tune_elongate_wdist import build_trial_config, mean_val_topo_wdist  # noqa: E402

BASE_CONFIG = "elongate_n100_no_cls_tune_local_pca_ellphi_power_h1_neartangent_barrier"


def geometry_stats(par: np.ndarray) -> dict[str, float]:
    axes = par[:, 0:2]
    major = axes.max(axis=1)
    minor = axes.min(axis=1)
    aspect = major / np.maximum(minor, 1e-12)
    return {
        "minor_min": float(minor.min()),
        "aspect_max": float(aspect.max()),
        "aspect_p99": float(np.percentile(aspect, 99)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--w-topo", type=float, required=True)
    p.add_argument("--w-aniso", type=float, required=True)
    p.add_argument("--w-size", type=float, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--base-config", type=str, default=BASE_CONFIG)
    p.add_argument("--trial-number", type=int, default=900)
    p.add_argument("--out-base", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_base.mkdir(parents=True, exist_ok=True)

    cfg = build_trial_config(
        args.base_config,
        w_aniso=args.w_aniso,
        w_size=args.w_size,
        w_topo=args.w_topo,
        lr=args.lr,
        backend="ellphi",
        tune_epochs=args.epochs,
        out_base=str(args.out_base),
        trial_number=args.trial_number,
        size_mode="power",
    )

    t0 = time.perf_counter()
    result = train_main(config=cfg)
    train_s = time.perf_counter() - t0
    run_dir = Path(result["run_dir"])

    device = torch.device("cpu")
    ckpt_name, ckpt_epoch, val_topo_sel = resolve_val_topo_checkpoint(run_dir)
    topo_options = topo_wdist_options_from_config(cfg)
    model = load_model_from_run(run_dir, cfg, device, checkpoint_name=ckpt_name)
    loader = build_split_loader(cfg, "val", device)
    clouds = list(iter_cloud_predictions(model, loader, device))
    wdist = mean_val_topo_wdist(clouds, topo_options=topo_options)

    geom = [geometry_stats(par) for _, par, _, _ in clouds]
    summary = {
        "purpose": (
            "elongate_barrier 20ep smoke on a weight set that hard-failed under "
            "plain elongate (near-tangent stage-2 tune)"
        ),
        "base_config": args.base_config,
        "source_revision": git_revision(REPO_ROOT),
        "weights": {
            "w_topo": args.w_topo,
            "w_aniso": args.w_aniso,
            "w_size": args.w_size,
            "lr": args.lr,
        },
        "epochs": args.epochs,
        "train_elapsed_s": round(train_s, 1),
        "run_dir": str(run_dir),
        "checkpoint_name": ckpt_name,
        "checkpoint_epoch": ckpt_epoch,
        "val_topo_loss_at_ckpt": val_topo_sel,
        "val_topo_wdist_mean": wdist,
        "geometry_val": {
            "minor_min": min(g["minor_min"] for g in geom),
            "aspect_max": max(g["aspect_max"] for g in geom),
            "aspect_p99_max": max(g["aspect_p99"] for g in geom),
        },
    }
    out_path = args.out_base / f"barrier_smoke_t{args.trial_number:03d}.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
