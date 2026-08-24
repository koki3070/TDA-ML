#!/usr/bin/env python3
"""30ep full run: recover loss weights + size_mode=power + val_topo ckpt + paper eval."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tda_ml.config import deep_update, load_config  # noqa: E402
from tda_ml.main import main as train_main  # noqa: E402
from tda_ml.preflight import (  # noqa: E402
    classify_tune_objective,
    paper_aniso_fields,
    preflight_tune_production_run,
    resolve_experiment_contract,
)
from tda_ml.seed_utils import cliff_init_model_seeds  # noqa: E402
from tda_ml.val_topo_cliff import ValTopoCliffError, cliff_max_from_config  # noqa: E402

BASE_CONFIG = "paper_rings"
TAG_MCC = "mcc"
TAG_WDIST = "wdist"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-config", type=str, default=BASE_CONFIG)
    p.add_argument("--size-ref", type=float, default=1.34)
    p.add_argument("--size-power", type=float, default=1.5)
    p.add_argument("--out-base", type=Path, default=None)
    p.add_argument(
        "--tune-json",
        type=Path,
        required=True,
        help=(
            "Rings H0+H1 Optuna best JSON (required). YAML-embedded w_*/lr are "
            "prior centres only and must not be used as paper production weights."
        ),
    )
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Paper eval output tag (default: inferred from tune JSON objective).",
    )
    p.add_argument(
        "--dbscan-backend",
        type=str,
        default="mahalanobis",
        choices=["ellphi", "mahalanobis"],
        help="DBSCAN distance for paper eval (default: mahalanobis).",
    )
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Override training.model_seed (default: data seed).",
    )
    p.add_argument(
        "--cliff-init-restarts",
        type=int,
        default=None,
        help=(
            "When val_topo cliff is required, try this many model_seed values "
            "before hard-failing. Default: 8 for rings cliff configs, else 1."
        ),
    )
    return p.parse_args()


def load_tune_weights(
    path: Path,
    *,
    size_ref_default: float,
    size_power_default: float,
) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Tune JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload["best_params"]
    weights = {
        "w_topo": float(params["w_topo"]),
        "w_aniso": float(params["w_aniso"]),
        "w_size": float(params["w_size"]),
        "lr": float(params["lr"]),
    }
    has_ref = "size_ref" in params
    has_power = "size_power" in params
    if has_ref != has_power:
        raise ValueError(
            "Tune JSON best_params must define both size_ref and size_power "
            f"together (or neither): {path}"
        )
    if has_ref:
        weights["size_ref"] = float(params["size_ref"])
        weights["size_power"] = float(params["size_power"])
        weights["size_from_tune"] = 1.0
    else:
        weights["size_ref"] = float(size_ref_default)
        weights["size_power"] = float(size_power_default)
        weights["size_from_tune"] = 0.0
    return weights


def infer_tag(tune_json: Path, explicit: str | None) -> str:
    """Map tune objective to paper-eval tag; hard-fail on unknown / mismatched tag."""
    payload = json.loads(tune_json.read_text(encoding="utf-8"))
    kind = classify_tune_objective(
        str(payload.get("objective", "")),
        objective_kind=payload.get("objective_kind"),
    )
    expected = TAG_WDIST if kind == "wdist" else TAG_MCC
    if explicit is not None and explicit != expected:
        raise ValueError(
            f"--tag {explicit!r} does not match tune objective kind {kind!r} "
            f"(expected tag {expected!r})"
        )
    return expected


def resolve_cliff_init_restarts(cfg: dict, explicit: int | None) -> int:
    if explicit is not None:
        if explicit < 1:
            raise ValueError(f"--cliff-init-restarts must be >= 1; got {explicit}")
        return int(explicit)
    if cliff_max_from_config(cfg) is None:
        return 1
    return 8


def main() -> int:
    args = parse_args()
    cfg = load_config(args.base_config, project_root=REPO_ROOT)
    paper_contract = resolve_experiment_contract(cfg)
    homology_dimensions = list(paper_contract["homology_dimensions"])
    tune_json = args.tune_json.resolve()
    tag = infer_tag(tune_json, args.tag)
    out_base = args.out_base
    if out_base is None:
        if tag == TAG_WDIST:
            out_base = REPO_ROOT / "outputs/supervised/paper30_wdist"
        else:
            out_base = REPO_ROOT / "outputs/supervised/paper30_mcc"
    out_base = out_base.resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    tune_weights = load_tune_weights(
        tune_json,
        size_ref_default=args.size_ref,
        size_power_default=args.size_power,
    )
    size_ref = tune_weights["size_ref"]
    size_power = tune_weights["size_power"]
    n_restarts = resolve_cliff_init_restarts(cfg, args.cliff_init_restarts)
    if args.model_seed is not None:
        model_seeds = [int(args.model_seed)]
    else:
        model_seeds = cliff_init_model_seeds(args.seed, n_restarts=n_restarts)

    aniso_fields = paper_aniso_fields(cfg)
    preflight_tune_production_run(
        base_config=args.base_config,
        tune_json=tune_json,
        project_root=REPO_ROOT,
        out_base=out_base,
        preflight_filename=f"RUN_PREFLIGHT_s{args.seed}.json",
        config_overrides={
            "loss": {
                **aniso_fields,
                "size_mode": "power",
                "size_ref": size_ref,
                "size_power": size_power,
            },
            "training": {"epochs": args.epochs},
            "data": {"seed": args.seed},
            "model": {
                "topology_loss": {
                    "distance_backend": "ellphi",
                    "prob_weighting": False,
                    "homology_dimensions": homology_dimensions,
                }
            },
            "outputs": {"base_dir": str(out_base)},
        },
    )

    loss_overrides: dict = {
        **aniso_fields,
        "size_mode": "power",
        "size_ref": size_ref,
        "size_power": size_power,
        "w_topo": tune_weights["w_topo"],
        "w_aniso": tune_weights["w_aniso"],
        "w_size": tune_weights["w_size"],
    }
    training_overrides: dict = {"epochs": args.epochs, "lr": tune_weights["lr"]}
    tune_source = str(tune_json)

    purpose_base = {
        "tag": tag,
        "seed": args.seed,
        "size_mode": "power",
        "size_ref": size_ref,
        "size_power": size_power,
        "size_from_tune": bool(tune_weights["size_from_tune"]),
        "weights": {
            "w_topo": tune_weights["w_topo"],
            "w_aniso": tune_weights["w_aniso"],
            "w_size": tune_weights["w_size"],
            "lr": tune_weights["lr"],
        },
        "selection": cfg["training"]["selection"]["metric"],
        "tune_json": tune_source,
        "dbscan_backend": args.dbscan_backend,
        "aniso_mode": cfg["loss"]["aniso_mode"],
        "aniso_barrier_threshold": aniso_fields.get("aniso_barrier_threshold"),
        "homology_dimensions": homology_dimensions,
        "paper_no_cls_contract": paper_contract,
        "cliff_init_restarts": n_restarts,
        "cliff_init_model_seeds": model_seeds,
        "protocol_note": (
            "paper 30ep H0+H1: raw ellipse params to ellphi; degenerate geometry "
            "hard-fails; tune weights fixed from seed-42 Optuna; rings cliff "
            "failures retry distinct training.model_seed values"
        ),
        "reference": tune_source,
    }
    if float(tune_weights["w_topo"]) == 0.0:
        purpose_base["ablation"] = "w_topo_zero"
        purpose_base["protocol_note"] = (
            "rings PH-off ablation: same aniso/size/lr as tune best with w_topo=0; "
            "selection=dbscan_mcc (val_topo is undefined when topo path is skipped); "
            "no val_topo cliff gate"
        )

    last_cliff_error: ValTopoCliffError | None = None
    result = None
    purpose = None
    for attempt, model_seed in enumerate(model_seeds):
        purpose = {
            **purpose_base,
            "model_seed": model_seed,
            "cliff_init_attempt": attempt,
        }
        cfg_try = deep_update(
            load_config(args.base_config, project_root=REPO_ROOT),
            {
                "meta": {
                    "config_id": f"paper_seed{args.seed}",
                    "run_slug": f"paper_s{args.seed}",
                },
                "loss": loss_overrides,
                "training": {**training_overrides, "model_seed": model_seed},
                "data": {"seed": args.seed},
                "model": {
                    "topology_loss": {
                        "distance_backend": "ellphi",
                        "prob_weighting": False,
                        "homology_dimensions": homology_dimensions,
                    }
                },
                "outputs": {"base_dir": str(out_base)},
            },
        )
        cfg_try["_manifest_extras"] = {
            "tune_json": tune_source,
            "tune_objective": json.loads(tune_json.read_text(encoding="utf-8")).get(
                "objective"
            ),
            "checkpoint_selection": cfg_try["training"]["selection"]["metric"],
            "paper_eval_dbscan_backend": args.dbscan_backend,
            "paper_no_cls_contract": paper_contract,
            "model_seed": model_seed,
            "cliff_init_attempt": attempt,
            "loss_overrides": {
                "size_mode": purpose["size_mode"],
                "size_ref": purpose["size_ref"],
                "size_power": purpose["size_power"],
                "w_topo": purpose["weights"]["w_topo"],
                "w_aniso": purpose["weights"]["w_aniso"],
                "w_size": purpose["weights"]["w_size"],
                "aniso_mode": purpose.get("aniso_mode"),
                "aniso_barrier_threshold": purpose.get("aniso_barrier_threshold"),
                "homology_dimensions": purpose.get("homology_dimensions"),
            },
        }

        plan_name = f"RUN_PLAN_s{args.seed}.json"
        (out_base / plan_name).write_text(
            json.dumps(purpose, indent=2) + "\n", encoding="utf-8"
        )
        (out_base / f"PURPOSE_s{args.seed}.json").write_text(
            json.dumps(purpose, indent=2) + "\n", encoding="utf-8"
        )

        print(
            f"[cliff-init] seed={args.seed} attempt={attempt}/{len(model_seeds) - 1} "
            f"model_seed={model_seed}",
            flush=True,
        )
        try:
            result = train_main(config=cfg_try)
            break
        except ValTopoCliffError as exc:
            last_cliff_error = exc
            print(
                f"[cliff-init] seed={args.seed} model_seed={model_seed} FAILED: {exc}",
                flush=True,
            )
            continue

    if result is None:
        raise RuntimeError(
            f"val_topo cliff failed for data seed={args.seed} after "
            f"{len(model_seeds)} model_seed attempts {model_seeds}"
        ) from last_cliff_error

    assert purpose is not None
    run_dir = Path(result["run_dir"])
    print(f"run_dir={run_dir}")
    (run_dir / "RUN_PLAN.json").write_text(
        json.dumps(purpose, indent=2) + "\n", encoding="utf-8"
    )

    if args.skip_eval:
        return 0

    eval_script = REPO_ROOT / "experiments" / "eval_paper.py"
    for split in ("val", "test"):
        cmd = [
            "uv",
            "run",
            "python",
            str(eval_script),
            "--run-dir",
            str(run_dir),
            "--base-config",
            args.base_config,
            "--split",
            split,
            "--backend",
            args.dbscan_backend,
            "--tag",
            tag,
        ]
        if split == "test":
            cmd.extend(
                [
                    "--dbscan-hparams",
                    str(run_dir / "logs" / f"dbscan_hparams_{tag}.json"),
                ]
            )
        print(f"[eval {split}] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
