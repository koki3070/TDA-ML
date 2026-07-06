#!/usr/bin/env python3
"""
Optuna tuning of elongate hyperparameters (val topo W-Dist objective).

Search space (log-uniform): ``w_aniso``, ``w_size``, ``w_topo``, ``lr``.

Each trial:
1. Trains ``--tune-epochs`` with ``save_every=1`` (``best_model.pth`` = val_topo min).
2. Loads ``best_model.pth`` (fallback: train_topo best saved checkpoint).
3. Objective = mean val **topo W-Dist** (learned ellipses vs local_pca teacher PD).

Base config ``elongate_n100_no_cls_tune_local_pca`` sets ``teacher_mode: local_pca``,
``w_class: 0``, ``selection.metric: val_topo``.

Usage (parallel, recommended)::

    bash experiments/run_tune_local_pca_parallel.sh 8 50 20 ellphi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from tda_ml.checkpoint_io import load_torch_checkpoint
from tda_ml.config import deep_update, load_config
from tda_ml.main import main as train_main
from tda_ml.topo_wdist import TopoWdistOptions, compute_topo_wdist, topo_wdist_options_from_config

from checkpoint_selection import best_topo_checkpoint  # noqa: E402
from evaluate_paper_protocol import (  # noqa: E402
    build_split_loader,
    iter_cloud_predictions,
    load_model_from_run,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_POLICY = "val_topo_best"
SAVE_EVERY = 1


def mean_val_topo_wdist(
    clouds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    topo_options: TopoWdistOptions,
) -> float:
    """Mean per-cloud topo W-Dist on val (independent of DBSCAN)."""
    vals: list[float] = []
    for points, params, _labels_gt, clean_pc in clouds:
        vals.append(compute_topo_wdist(points, params, clean_pc, topo_options))
    if not vals:
        return float("inf")
    return float(np.mean(vals))


def resolve_tune_checkpoint(run_dir: Path) -> tuple[str, int, float]:
    """Prefer ``best_model.pth`` (val_topo); else train_topo best saved ckpt."""
    best_path = run_dir / "best_model.pth"
    if best_path.is_file():
        ckpt = load_torch_checkpoint(str(best_path), map_location="cpu")
        epoch = int(ckpt.get("epoch", -1))
        sel = ckpt.get("selection_value", ckpt.get("val_topo_loss"))
        val_topo = float(sel) if sel is not None else float("nan")
        return "best_model.pth", epoch, val_topo
    ckpt_name, epoch, train_topo, _g_ep, _g_topo = best_topo_checkpoint(run_dir)
    return ckpt_name, epoch, train_topo


def build_trial_config(
    base_config: str, *, w_aniso: float, w_size: float, w_topo: float, lr: float,
    backend: str, tune_epochs: int, out_base: str, trial_number: int,
) -> dict[str, Any]:
    cfg = load_config(base_config, project_root=REPO_ROOT)
    overrides = {
        "meta": {"config_id": f"tune_elongate_t{trial_number:03d}"},
        "loss": {
            "w_aniso": float(w_aniso),
            "w_size": float(w_size),
            "w_topo": float(w_topo),
            "aniso_mode": "elongate",
        },
        "training": {
            "epochs": int(tune_epochs),
            "lr": float(lr),
            "visualize_every": 10_000,
            "selection": {"metric": "val_topo", "eval_every": 1},
        },
        "model": {
            "topology_loss": {
                "distance_backend": backend,
                "prob_weighting": False,
            }
        },
        "outputs": {"base_dir": out_base, "save_every": SAVE_EVERY},
    }
    return deep_update(cfg, overrides)


def make_objective(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.Trial) -> float:
        w_aniso = trial.suggest_float("w_aniso", 0.05, 5.0, log=True)
        w_size = trial.suggest_float("w_size", 0.005, 0.5, log=True)
        w_topo = trial.suggest_float("w_topo", 0.02, 0.5, log=True)
        lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)

        cfg = build_trial_config(
            args.base_config, w_aniso=w_aniso, w_size=w_size, w_topo=w_topo, lr=lr,
            backend=args.backend, tune_epochs=args.tune_epochs, out_base=args.out_base,
            trial_number=trial.number,
        )
        result = train_main(config=cfg)
        run_dir = Path(result["run_dir"])

        ckpt_name, ckpt_epoch, val_topo_sel = resolve_tune_checkpoint(run_dir)
        topo_options = topo_wdist_options_from_config(cfg)
        model = load_model_from_run(run_dir, cfg, device, checkpoint_name=ckpt_name)
        loader = build_split_loader(cfg, "val", device)
        clouds = list(iter_cloud_predictions(model, loader, device))

        mwd = mean_val_topo_wdist(clouds, topo_options=topo_options)
        trial.set_user_attr("checkpoint_policy", CHECKPOINT_POLICY)
        trial.set_user_attr("checkpoint_name", ckpt_name)
        trial.set_user_attr("checkpoint_epoch", ckpt_epoch)
        trial.set_user_attr("val_topo_loss_at_ckpt", val_topo_sel)
        trial.set_user_attr("teacher_mode", topo_options.teacher_mode)
        trial.set_user_attr("run_dir", str(run_dir))
        print(
            f"[trial {trial.number}] w_aniso={w_aniso:.4f} w_size={w_size:.4f} "
            f"w_topo={w_topo:.4f} lr={lr:.2e} "
            f"ckpt={ckpt_name} ep={ckpt_epoch} val_topo_sel={val_topo_sel:.5f} "
            f"-> val topo W-Dist={mwd:.5f}"
        )
        return mwd

    return objective


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", type=str, default="elongate_n100_no_cls_tune_local_pca")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument(
        "--n-startup-trials",
        type=int,
        default=12,
        help="Random startup trials before TPE modeling kicks in.",
    )
    p.add_argument("--tune-epochs", type=int, default=20)
    p.add_argument("--backend", type=str, default="mahalanobis", choices=["mahalanobis", "ellphi"])
    p.add_argument("--out-base", type=str, default="outputs/tune_elongate")
    p.add_argument("--seed", type=int, default=42, help="Optuna sampler seed")
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (required for parallel workers).",
    )
    p.add_argument(
        "--study-name",
        type=str,
        default="elongate_wdist_4d",
        help="Study name (shared across parallel workers when using --storage).",
    )
    p.add_argument(
        "--write-best",
        action="store_true",
        help="Write best_elongate_wdist.json at the end (run once after all workers).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    Path(args.out_base).mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.n_startup_trials),
    )

    if not args.write_best:
        callbacks = []
        if args.storage:
            callbacks.append(
                MaxTrialsCallback(args.n_trials, states=(TrialState.COMPLETE,))
            )
        study.optimize(
            make_objective(args),
            n_trials=args.n_trials,
            callbacks=callbacks,
            catch=(Exception,),
        )
        n_done = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        print(f"[worker done] completed trials in study so far: {n_done}")
        return 0

    best = study.best_trial
    payload = {
        "objective": "val_topo_wdist_min_at_val_topo_best_ckpt",
        "checkpoint_policy": CHECKPOINT_POLICY,
        "save_every": SAVE_EVERY,
        "base_config": args.base_config,
        "backend": args.backend,
        "teacher_mode": "local_pca",
        "tune_epochs": args.tune_epochs,
        "n_trials": args.n_trials,
        "n_startup_trials": args.n_startup_trials,
        "search_space": {
            "w_aniso": [0.05, 5.0],
            "w_size": [0.005, 0.5],
            "w_topo": [0.02, 0.5],
            "lr": [1e-4, 2e-3],
        },
        "best_value_wdist": best.value,
        "best_params": best.params,
        "best_checkpoint_epoch": best.user_attrs.get("checkpoint_epoch"),
        "best_val_topo_loss_at_ckpt": best.user_attrs.get("val_topo_loss_at_ckpt"),
        "best_run_dir": best.user_attrs.get("run_dir"),
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "checkpoint_epoch": t.user_attrs.get("checkpoint_epoch"),
                "val_topo_loss_at_ckpt": t.user_attrs.get("val_topo_loss_at_ckpt"),
            }
            for t in study.trials
        ],
    }
    out_path = Path(args.out_base) / f"best_elongate_wdist_{args.backend}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in (
        "best_value_wdist", "best_params", "best_checkpoint_epoch",
        "best_val_topo_loss_at_ckpt",
    )}, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
