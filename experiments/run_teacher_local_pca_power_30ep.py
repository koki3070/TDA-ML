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
    assert_paper_no_cls_contract,
    classify_tune_objective,
    preflight_tune_production_run,
)

BASE_CONFIG = "elongate_n100_no_cls_full120_teacher_local_pca"
TAG_MCC = "power_mcc_valtopo_paper_eval"
TAG_WDIST = "power_wdist_valtopo_paper_eval"


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
            "H1-only Optuna best JSON (required). YAML-embedded w_*/lr are prior "
            "centres only and must not be used as paper production weights."
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
    return p.parse_args()


def load_tune_weights(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Tune JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload["best_params"]
    return {
        "w_topo": float(params["w_topo"]),
        "w_aniso": float(params["w_aniso"]),
        "w_size": float(params["w_size"]),
        "lr": float(params["lr"]),
    }


def infer_tag(tune_json: Path, explicit: str | None) -> str:
    """Map tune objective to paper-eval tag; hard-fail on unknown objective."""
    if explicit:
        return explicit
    payload = json.loads(tune_json.read_text(encoding="utf-8"))
    kind = classify_tune_objective(
        str(payload.get("objective", "")),
        objective_kind=payload.get("objective_kind"),
    )
    return TAG_WDIST if kind == "wdist" else TAG_MCC


def main() -> int:
    args = parse_args()
    cfg = load_config(args.base_config, project_root=REPO_ROOT)
    paper_contract = assert_paper_no_cls_contract(cfg)
    tune_json = args.tune_json
    tag = infer_tag(tune_json, args.tag)
    out_base = args.out_base
    if out_base is None:
        if tag == TAG_WDIST:
            out_base = REPO_ROOT / "outputs/supervised/pwr30_wdist"
        else:
            out_base = REPO_ROOT / "outputs/supervised/pwr30_mcc"
    out_base.mkdir(parents=True, exist_ok=True)

    preflight_tune_production_run(
        base_config=args.base_config,
        tune_json=tune_json,
        project_root=REPO_ROOT,
        out_base=out_base,
        config_overrides={
            "loss": {
                "aniso_mode": "elongate",
                "size_mode": "power",
                "size_ref": args.size_ref,
                "size_power": args.size_power,
            },
            "training": {"epochs": args.epochs},
            "data": {"seed": args.seed},
            "model": {
                "topology_loss": {
                    "distance_backend": "ellphi",
                    "prob_weighting": False,
                    "homology_dimensions": [1],
                }
            },
            "outputs": {"base_dir": str(out_base)},
        },
    )

    loss_overrides: dict = {
        "aniso_mode": "elongate",
        "size_mode": "power",
        "size_ref": args.size_ref,
        "size_power": args.size_power,
    }
    training_overrides: dict = {"epochs": args.epochs}
    tune_weights = load_tune_weights(tune_json)
    loss_overrides.update(
        {
            "w_topo": tune_weights["w_topo"],
            "w_aniso": tune_weights["w_aniso"],
            "w_size": tune_weights["w_size"],
        }
    )
    training_overrides["lr"] = tune_weights["lr"]
    tune_source = str(tune_json)

    cfg = deep_update(
        cfg,
        {
            "meta": {
                "config_id": f"teacher_local_pca_power_seed{args.seed}",
                "run_slug": f"pwr_s{args.seed}",
            },
            "loss": loss_overrides,
            "training": training_overrides,
            "data": {"seed": args.seed},
            "model": {
                "topology_loss": {
                    "distance_backend": "ellphi",
                    "prob_weighting": False,
                    "homology_dimensions": [1],
                }
            },
            "outputs": {"base_dir": str(out_base)},
        },
    )

    purpose = {
        "tag": tag,
        "size_mode": cfg["loss"]["size_mode"],
        "size_ref": cfg["loss"]["size_ref"],
        "size_power": cfg["loss"]["size_power"],
        "weights": {
            "w_topo": cfg["loss"]["w_topo"],
            "w_aniso": cfg["loss"]["w_aniso"],
            "w_size": cfg["loss"]["w_size"],
            "lr": cfg["training"]["lr"],
        },
        "selection": cfg["training"]["selection"]["metric"],
        "tune_json": tune_source,
        "dbscan_backend": args.dbscan_backend,
        "aniso_mode": cfg["loss"]["aniso_mode"],
        "homology_dimensions": cfg["model"]["topology_loss"]["homology_dimensions"],
        "paper_no_cls_contract": paper_contract,
        "protocol_note": (
            "power 30ep H1-only: raw ellipse params to ellphi; degenerate geometry "
            "hard-fails; tune weights fixed from seed-42 Optuna"
        ),
        "reference": tune_source,
    }
    cfg["_manifest_extras"] = {
        "tune_json": tune_source,
        "tune_objective": purpose.get("tune_json")
        and json.loads(Path(tune_source).read_text(encoding="utf-8")).get("objective"),
        "checkpoint_selection": cfg["training"]["selection"]["metric"],
        "paper_eval_dbscan_backend": args.dbscan_backend,
        "loss_overrides": {
            "size_mode": purpose["size_mode"],
            "size_ref": purpose["size_ref"],
            "size_power": purpose["size_power"],
            "w_topo": purpose["weights"]["w_topo"],
            "w_aniso": purpose["weights"]["w_aniso"],
            "w_size": purpose["weights"]["w_size"],
            "aniso_mode": purpose.get("aniso_mode"),
            "homology_dimensions": purpose.get("homology_dimensions"),
        },
    }

    (out_base / "RUN_PLAN.json").write_text(
        json.dumps(purpose, indent=2) + "\n", encoding="utf-8"
    )
    # Legacy alias
    (out_base / "PURPOSE.json").write_text(
        json.dumps(purpose, indent=2) + "\n", encoding="utf-8"
    )

    result = train_main(config=cfg)
    run_dir = Path(result["run_dir"])
    print(f"run_dir={run_dir}")

    if args.skip_eval:
        return 0

    eval_script = REPO_ROOT / "experiments" / "evaluate_paper_protocol.py"
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
            "--checkpoint-name",
            "best_model.pth",
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
