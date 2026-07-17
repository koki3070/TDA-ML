#!/usr/bin/env python3
"""
Optuna tuning for ellphi + local_pca teacher with **val DBSCAN MCC** objective.

Protocol (2026-07-09):
- Loss stack matches production: ``size_mode=power`` (or CLI override).
- Train checkpoint: ``val_topo`` min (fast; no per-epoch DBSCAN grid).
- Trial score: one val DBSCAN grid on ``best_model.pth`` → MCC max.

Distance backends are separated by role:
- ``--backend`` (default ``ellphi``): training topo-loss = ellipse tangency filtration.
- ``--dbscan-backend`` (default ``mahalanobis``): clustering distance for the MCC
  objective. Mahalanobis is the geometrically meaningful point-to-point distance;
  ellphi tangency time is a filtration parameter, not a clustering distance.

Search: ``w_topo``, ``w_aniso``, ``w_size``, ``lr`` (narrow bands around recover).
Optional: ``size_ref``, ``size_power`` when ``--tune-size-hyperparams``.

Usage::

    bash experiments/run_tune_local_pca_power_mcc_parallel.sh 4 24 20 ellphi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import optuna
import torch
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from tda_ml.checkpoint_io import resolve_val_topo_checkpoint
from tda_ml.config import deep_update, load_config
from tda_ml.dbscan_eval import evaluate_model_grid
from tda_ml.main import main as train_main
from tda_ml.preflight import preflight_mcc_tune_study
from tda_ml.reproducibility import reproducibility_settings, write_json
from tda_ml.supervised_diagnostics import git_revision
from tda_ml.topo_wdist import topo_wdist_options_from_config

from evaluate_paper_protocol import (  # noqa: E402
    build_split_loader,
    iter_cloud_predictions,
    load_model_from_run,
)
from tune_elongate_wdist import mean_val_topo_wdist  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_POLICY = "val_topo_best"
SAVE_EVERY = 1

# Recover-informed narrow search (log-uniform).
W_TOPO_RANGE = (0.05, 0.25)
W_SIZE_RANGE = (0.1, 0.6)
W_ANISO_RANGE = (0.03, 0.15)
LR_RANGE = (1e-4, 5e-4)
SIZE_REF_RANGE = (0.8, 1.6)
SIZE_POWER_RANGE = (1.0, 2.0)


def build_trial_config(
    base_config: str,
    *,
    w_aniso: float,
    w_size: float,
    w_topo: float,
    lr: float,
    backend: str,
    tune_epochs: int,
    out_base: str,
    trial_number: int,
    size_mode: str,
    size_ref: float,
    size_power: float,
) -> dict[str, Any]:
    cfg = load_config(base_config, project_root=REPO_ROOT)
    overrides = {
        "meta": {"config_id": f"tune_mcc_t{trial_number:03d}", "run_slug": f"t{trial_number:03d}"},
        "loss": {
            "w_aniso": float(w_aniso),
            "w_size": float(w_size),
            "w_topo": float(w_topo),
            "aniso_mode": "elongate",
            "size_mode": size_mode,
            "size_ref": float(size_ref),
            "size_power": float(size_power),
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
                # Mirror paper contract / STUDY_PREFLIGHT overrides onto every trial.
                "homology_dimensions": [1],
            }
        },
        "outputs": {"base_dir": out_base, "save_every": SAVE_EVERY},
    }
    return deep_update(cfg, overrides)


def make_objective(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.Trial) -> float:
        w_aniso = trial.suggest_float("w_aniso", *W_ANISO_RANGE, log=True)
        w_size = trial.suggest_float("w_size", *W_SIZE_RANGE, log=True)
        w_topo = trial.suggest_float("w_topo", *W_TOPO_RANGE, log=True)
        lr = trial.suggest_float("lr", *LR_RANGE, log=True)
        size_ref = (
            trial.suggest_float("size_ref", *SIZE_REF_RANGE)
            if args.tune_size_hyperparams
            else float(args.size_ref)
        )
        size_power = (
            trial.suggest_float("size_power", *SIZE_POWER_RANGE)
            if args.tune_size_hyperparams
            else float(args.size_power)
        )

        cfg = build_trial_config(
            args.base_config,
            w_aniso=w_aniso,
            w_size=w_size,
            w_topo=w_topo,
            lr=lr,
            backend=args.backend,
            tune_epochs=args.tune_epochs,
            out_base=args.out_base,
            trial_number=trial.number,
            size_mode=args.size_mode,
            size_ref=size_ref,
            size_power=size_power,
        )
        rep = reproducibility_settings(cfg)
        trial_manifest_path = Path(args.out_base) / f"trial_{trial.number:03d}_manifest.json"
        write_json(
            trial_manifest_path,
            {
                "trial_number": trial.number,
                "params": trial.params,
                "checkpoint_policy": CHECKPOINT_POLICY,
                "dbscan_backend": args.dbscan_backend,
            },
        )
        result = train_main(config=cfg)
        run_dir = Path(result["run_dir"])

        ckpt_name, ckpt_epoch, val_topo = resolve_val_topo_checkpoint(run_dir)
        topo_options = topo_wdist_options_from_config(cfg)
        model = load_model_from_run(run_dir, cfg, device, checkpoint_name=ckpt_name)
        loader = build_split_loader(cfg, "val", device)

        grid = evaluate_model_grid(
            model,
            loader,
            device,
            config=cfg,
            backend=args.dbscan_backend,
            objective="mcc",
            topo_options=topo_options,
            allow_skip_degenerate_grid_cells=rep["allow_skip_degenerate_grid_cells"],
            grid_log_path=run_dir / "logs" / "dbscan_grid_log_tune_trial.json",
            manifest_ref=cfg.get("_manifest"),
        )
        clouds = list(iter_cloud_predictions(model, loader, device))
        mwd = mean_val_topo_wdist(clouds, topo_options=topo_options)

        trial.set_user_attr("checkpoint_policy", CHECKPOINT_POLICY)
        trial.set_user_attr("checkpoint_name", ckpt_name)
        trial.set_user_attr("checkpoint_epoch", ckpt_epoch)
        trial.set_user_attr("val_topo_loss", val_topo)
        trial.set_user_attr("val_topo_wdist", mwd)
        trial.set_user_attr("size_mode", args.size_mode)
        trial.set_user_attr("dbscan_backend", args.dbscan_backend)
        trial.set_user_attr("size_ref", size_ref)
        trial.set_user_attr("size_power", size_power)
        trial.set_user_attr("dbscan_eps", grid.eps)
        trial.set_user_attr("dbscan_min_samples", grid.min_samples)
        trial.set_user_attr("run_dir", str(run_dir))
        print(
            f"[trial {trial.number}] mcc={grid.mcc:.4f} ({args.dbscan_backend} DBSCAN) "
            f"topo_WDist={mwd:.4f} eps={grid.eps:.3f} ms={grid.min_samples} "
            f"w_topo={w_topo:.3f} w_size={w_size:.3f} ep={ckpt_epoch}"
        )
        return grid.mcc

    return objective


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-config",
        default="elongate_n100_no_cls_tune_local_pca_ellphi_power_mcc",
    )
    p.add_argument("--n-trials", type=int, default=24)
    p.add_argument(
        "--max-complete-trials",
        type=int,
        default=None,
        help="Shared-study completion cap; defaults to --n-trials.",
    )
    p.add_argument("--n-startup-trials", type=int, default=8)
    p.add_argument("--tune-epochs", type=int, default=20)
    p.add_argument(
        "--backend",
        default="ellphi",
        choices=["ellphi", "mahalanobis"],
        help="Training topo-loss distance backend (ellphi = ellipse tangency filtration).",
    )
    p.add_argument(
        "--dbscan-backend",
        default="mahalanobis",
        choices=["ellphi", "mahalanobis"],
        help=(
            "DBSCAN distance backend for the trial objective. Mahalanobis is the "
            "geometrically meaningful choice for clustering (ellphi tangency time is "
            "not a point-to-point distance)."
        ),
    )
    p.add_argument("--out-base", default="outputs/tune/pwr_mcc")
    p.add_argument("--size-mode", default="power", choices=["quadratic", "power", "softplus", "barrier"])
    p.add_argument("--size-ref", type=float, default=1.34)
    p.add_argument("--size-power", type=float, default=1.5)
    p.add_argument(
        "--tune-size-hyperparams",
        action="store_true",
        help="Also search size_ref and size_power (6D search).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--storage", default=None)
    p.add_argument("--study-name", default="elongate_local_pca_power_mcc_ellphi")
    p.add_argument("--write-best", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    Path(args.out_base).mkdir(parents=True, exist_ok=True)
    preflight = preflight_mcc_tune_study(
        base_config=args.base_config,
        project_root=REPO_ROOT,
        out_base=args.out_base,
        config_overrides={
            "model": {
                "topology_loss": {
                    "distance_backend": args.backend,
                    "prob_weighting": False,
                    "homology_dimensions": [1],
                }
            },
            "loss": {
                "aniso_mode": "elongate",
                "size_mode": args.size_mode,
                "size_ref": args.size_ref,
                "size_power": args.size_power,
            },
            "training": {"epochs": args.tune_epochs},
            "outputs": {"base_dir": args.out_base},
        },
    )
    preflight.update(
        {
            "run_status": "pending",
            "command_entry": "experiments/tune_elongate_mcc.py",
            "source_revision": git_revision(REPO_ROOT),
            "study_name": args.study_name,
            "storage": args.storage,
            "n_trials": args.n_trials,
            "max_complete_trials": args.max_complete_trials or args.n_trials,
            "n_startup_trials": args.n_startup_trials,
            "sampler_seed": args.seed,
            "fallbacks": [],
        }
    )
    write_json(
        Path(args.out_base) / f"WORKER_PREFLIGHT_seed{args.seed}.json",
        preflight,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.n_startup_trials),
    )

    if not args.write_best:
        callbacks = []
        if args.storage:
            max_complete = args.max_complete_trials or args.n_trials
            callbacks.append(
                MaxTrialsCallback(max_complete, states=(TrialState.COMPLETE,))
            )
        study.optimize(
            make_objective(args),
            n_trials=args.n_trials,
            callbacks=callbacks,
        )
        n_done = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        print(f"[worker done] completed trials in study so far: {n_done}")
        return 0

    best = study.best_trial
    payload = {
        "objective": "val_dbscan_mcc_max_at_val_topo_best_ckpt",
        "objective_kind": "mcc",
        "checkpoint_policy": CHECKPOINT_POLICY,
        "checkpoint_selection": "val_topo",
        "save_every": SAVE_EVERY,
        "base_config": args.base_config,
        "backend": args.backend,
        "dbscan_backend": args.dbscan_backend,
        "size_mode": args.size_mode,
        "size_ref_default": args.size_ref,
        "size_power_default": args.size_power,
        "tune_size_hyperparams": bool(args.tune_size_hyperparams),
        "tune_epochs": args.tune_epochs,
        "n_trials": args.n_trials,
        "max_complete_trials": args.max_complete_trials or args.n_trials,
        "source_revision": git_revision(REPO_ROOT),
        "study_name": args.study_name,
        "storage": args.storage,
        "sampler_seed": args.seed,
        **preflight["paper_no_cls_contract"],
        "search_space": {
            "w_aniso": list(W_ANISO_RANGE),
            "w_size": list(W_SIZE_RANGE),
            "w_topo": list(W_TOPO_RANGE),
            "lr": list(LR_RANGE),
            **(
                {"size_ref": list(SIZE_REF_RANGE), "size_power": list(SIZE_POWER_RANGE)}
                if args.tune_size_hyperparams
                else {}
            ),
        },
        "best_value_mcc": best.value,
        "best_params": best.params,
        "best_val_topo_wdist": best.user_attrs.get("val_topo_wdist"),
        "best_checkpoint_epoch": best.user_attrs.get("checkpoint_epoch"),
        "best_run_dir": best.user_attrs.get("run_dir"),
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "val_topo_wdist": t.user_attrs.get("val_topo_wdist"),
                "checkpoint_epoch": t.user_attrs.get("checkpoint_epoch"),
            }
            for t in study.trials
        ],
    }
    out_path = (
        Path(args.out_base)
        / f"best_elongate_mcc_{args.backend}_dbscan_{args.dbscan_backend}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("best_value_mcc", "best_params", "best_val_topo_wdist")}, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
