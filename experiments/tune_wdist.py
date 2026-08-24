#!/usr/bin/env python3
"""
Optuna tuning of paper loss weights (val topo W-Dist objective).

Search space (log-uniform): ``w_aniso``, ``w_size``, ``w_topo``, ``lr``.

Each trial:
1. Trains ``--tune-epochs`` with ``save_every=1`` (``best_model.pth`` = val_topo min).
2. Loads ``best_model.pth`` (val_topo selection checkpoint; hard-fail if missing).
3. Objective = mean val **topo W-Dist** (learned ellipses vs local_pca teacher PD).

Base config ``tune_rings`` sets
``teacher_mode: local_pca``, H0+H1 persistence, thin rings + radial outliers,
and ``size_mode: power``. ``w_class: 0``, ``selection.metric: val_topo``.

Usage (parallel, recommended)::

    bash experiments/tune_wdist_parallel.sh 8 50 20 ellphi
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

from tda_ml.checkpoint_io import resolve_val_topo_checkpoint
from tda_ml.val_topo_cliff import ValTopoCliffError, cliff_max_from_config
from tda_ml.config import deep_update, load_config
from tda_ml.main import main as train_main
from tda_ml.preflight import paper_aniso_fields, preflight_wdist_tune_study, resolve_experiment_contract
from tda_ml.supervised_diagnostics import git_revision
from tda_ml.topo_wdist import TopoWdistOptions, compute_topo_wdist, topo_wdist_options_from_config

from eval_paper import (  # noqa: E402
    build_split_loader,
    iter_cloud_predictions,
    load_model_from_run,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_POLICY = "val_topo_best"
SAVE_EVERY = 1

# Narrow bands for H0+H1 rings + teacher_local_pca_major_scale=0.083.
# Objective remains val topo W-Dist; the band is the declared search constraint.
NARROW_W_TOPO_RANGE = (0.008, 0.045)
NARROW_W_SIZE_RANGE = (0.10, 0.40)
NARROW_W_ANISO_RANGE = (0.05, 0.15)
NARROW_LR_RANGE = (1.2e-4, 3.0e-4)

# Legacy wide search (also shifted down historically for H0+H1).
WIDE_W_ANISO_RANGE = (0.05, 5.0)
WIDE_W_SIZE_RANGE = (0.005, 0.5)
WIDE_W_TOPO_RANGE = (0.0005, 0.5)
WIDE_LR_RANGE = (1e-4, 2e-3)


def mean_val_topo_wdist(
    clouds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    topo_options: TopoWdistOptions,
) -> float:
    """Mean per-cloud topo W-Dist on val (independent of DBSCAN) without ad-hoc fallbacks."""
    vals: list[float] = []
    for points, params, _labels_gt, clean_pc in clouds:
        val = compute_topo_wdist(points, params, clean_pc, topo_options)
        vals.append(val)
    return float(np.mean(vals))


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
    size_mode: str = "power",
    size_ref: float = 1.34,
    size_power: float = 1.5,
) -> dict[str, Any]:
    cfg = load_config(base_config, project_root=REPO_ROOT)
    homology_dimensions = list(
        resolve_experiment_contract(cfg)["homology_dimensions"]
    )
    overrides = {
        "meta": {"config_id": f"tune_t{trial_number:03d}", "run_slug": f"t{trial_number:03d}"},
        "loss": {
            "w_aniso": float(w_aniso),
            "w_size": float(w_size),
            "w_topo": float(w_topo),
            # Mirror paper contract / STUDY_PREFLIGHT overrides onto every trial.
            # aniso variant (elongate vs elongate_barrier) comes from the base
            # config declaration; hard-fails if the base config omits it.
            **paper_aniso_fields(cfg),
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
                "homology_dimensions": homology_dimensions,
            }
        },
        "outputs": {"base_dir": out_base, "save_every": SAVE_EVERY},
    }
    return deep_update(cfg, overrides)


def search_ranges(*, narrow: bool) -> tuple[tuple[float, float], ...]:
    if narrow:
        return (
            NARROW_W_ANISO_RANGE,
            NARROW_W_SIZE_RANGE,
            NARROW_W_TOPO_RANGE,
            NARROW_LR_RANGE,
        )
    return (
        WIDE_W_ANISO_RANGE,
        WIDE_W_SIZE_RANGE,
        WIDE_W_TOPO_RANGE,
        WIDE_LR_RANGE,
    )


def make_objective(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w_aniso_range, w_size_range, w_topo_range, lr_range = search_ranges(
        narrow=bool(args.narrow_search)
    )

    def objective(trial: optuna.Trial) -> float:
        w_aniso = trial.suggest_float("w_aniso", *w_aniso_range, log=True)
        w_size = trial.suggest_float("w_size", *w_size_range, log=True)
        w_topo = trial.suggest_float("w_topo", *w_topo_range, log=True)
        lr = trial.suggest_float("lr", *lr_range, log=True)

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
            size_ref=args.size_ref,
            size_power=args.size_power,
        )
        try:
            result = train_main(config=cfg)
        except ValTopoCliffError as exc:
            trial.set_user_attr("val_topo_cliff_passed", False)
            trial.set_user_attr("val_topo_cliff_error", str(exc))
            raise optuna.TrialPruned(str(exc)) from exc
        run_dir = Path(result["run_dir"])

        ckpt_name, ckpt_epoch, val_topo_sel = resolve_val_topo_checkpoint(run_dir)
        cliff_max = cliff_max_from_config(cfg)
        if cliff_max is not None:
            trial.set_user_attr("val_topo_cliff_max", cliff_max)
            trial.set_user_attr("val_topo_cliff_passed", True)
        topo_options = topo_wdist_options_from_config(cfg)
        model = load_model_from_run(run_dir, cfg, device)
        loader = build_split_loader(cfg, "val", device)
        clouds = list(iter_cloud_predictions(model, loader, device))

        mwd = mean_val_topo_wdist(clouds, topo_options=topo_options)
        trial.set_user_attr("checkpoint_policy", CHECKPOINT_POLICY)
        trial.set_user_attr("checkpoint_name", ckpt_name)
        trial.set_user_attr("checkpoint_epoch", ckpt_epoch)
        trial.set_user_attr("val_topo_loss_at_ckpt", val_topo_sel)
        trial.set_user_attr("teacher_mode", topo_options.teacher_mode)
        trial.set_user_attr("size_mode", args.size_mode)
        trial.set_user_attr("run_dir", str(run_dir))
        print(
            f"[trial {trial.number}] wdist={mwd:.5f} "
            f"w_topo={w_topo:.3f} w_size={w_size:.3f} ep={ckpt_epoch}"
        )
        return mwd

    return objective


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # No implicit default: every study must state its config surface explicitly
    # (power stack uses tune_rings).
    p.add_argument("--base-config", type=str, required=True)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument(
        "--max-complete-trials",
        type=int,
        default=None,
        help="Shared-study completion cap; defaults to --n-trials.",
    )
    p.add_argument(
        "--n-startup-trials",
        type=int,
        default=12,
        help="Random startup trials before TPE modeling kicks in.",
    )
    p.add_argument("--tune-epochs", type=int, default=20)
    p.add_argument(
        "--backend",
        type=str,
        default="ellphi",
        choices=["ellphi"],
        help="Training topo-loss backend (ellphi = ellipse tangency filtration).",
    )
    p.add_argument("--out-base", type=str, default="outputs/tune/wdist")
    p.add_argument(
        "--size-mode",
        default="power",
        choices=["quadratic", "power", "softplus", "barrier"],
    )
    p.add_argument("--size-ref", type=float, default=1.34)
    p.add_argument("--size-power", type=float, default=1.5)
    p.add_argument(
        "--narrow-search",
        action="store_true",
        help="Use narrow search bands (same as tune_mcc power protocol).",
    )
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
        default="tune_wdist",
        help="Study name (shared across parallel workers when using --storage).",
    )
    p.add_argument(
        "--write-best",
        action="store_true",
        help="Write best_wdist.json at the end (run once after all workers).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    Path(args.out_base).mkdir(parents=True, exist_ok=True)
    base_cfg_for_aniso = load_config(args.base_config, project_root=REPO_ROOT)
    homology_dimensions = list(
        resolve_experiment_contract(base_cfg_for_aniso)["homology_dimensions"]
    )
    preflight = preflight_wdist_tune_study(
        base_config=args.base_config,
        project_root=REPO_ROOT,
        out_base=args.out_base,
        config_overrides={
            "model": {
                "topology_loss": {
                    "distance_backend": args.backend,
                    "prob_weighting": False,
                    "homology_dimensions": homology_dimensions,
                }
            },
            "loss": {
                **paper_aniso_fields(base_cfg_for_aniso),
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
            "command_entry": "experiments/tune_wdist.py",
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
    (Path(args.out_base) / f"WORKER_PREFLIGHT_seed{args.seed}.json").write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
    )

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

    complete = [
        t
        for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None
    ]
    if not complete:
        raise RuntimeError(
            f"No COMPLETE Optuna trials in study {args.study_name!r}; cannot write best. "
            "If require_val_topo_cliff is enabled, all trials may have been pruned "
            "for missing the val_topo phase transition — lengthen --tune-epochs / expand search."
        )
    best = min(complete, key=lambda t: float(t.value))
    w_aniso_range, w_size_range, w_topo_range, lr_range = search_ranges(
        narrow=bool(args.narrow_search)
    )
    payload = {
        "objective": "val_topo_wdist_min_at_val_topo_best_ckpt",
        "objective_kind": "wdist",
        "checkpoint_policy": CHECKPOINT_POLICY,
        "save_every": SAVE_EVERY,
        "base_config": args.base_config,
        "backend": args.backend,
        "size_mode": args.size_mode,
        "size_ref_default": args.size_ref,
        "size_power_default": args.size_power,
        "narrow_search": bool(args.narrow_search),
        "tune_epochs": args.tune_epochs,
        "n_trials": args.n_trials,
        "max_complete_trials": args.max_complete_trials or args.n_trials,
        "n_startup_trials": args.n_startup_trials,
        "source_revision": git_revision(REPO_ROOT),
        "study_name": args.study_name,
        "storage": args.storage,
        "sampler_seed": args.seed,
        **preflight["paper_no_cls_contract"],
        "search_space": {
            "w_aniso": list(w_aniso_range),
            "w_size": list(w_size_range),
            "w_topo": list(w_topo_range),
            "lr": list(lr_range),
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
    out_path = Path(args.out_base) / f"best_wdist_{args.backend}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in (
        "best_value_wdist", "best_params", "best_checkpoint_epoch",
        "best_val_topo_loss_at_ckpt",
    )}, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
