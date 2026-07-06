#!/usr/bin/env python3
"""Evaluate a run using the epoch with minimum train_topo_loss (saved checkpoints).

Selects checkpoint_epoch_{k}.pth where k minimizes train_topo_loss in metrics.csv,
runs val DBSCAN grid search, then test evaluation (paper protocol).
Optionally compares against a baseline run JSON output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from checkpoint_selection import best_topo_checkpoint, best_topo_epoch, list_checkpoint_epochs

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_eval(
    run_dir: Path,
    base_config: str,
    split: str,
    backend: str,
    checkpoint_name: str,
    *,
    tag: str | None = None,
    dbscan_hparams: Path | None = None,
) -> Path:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "evaluate_paper_protocol.py"),
        "--run-dir",
        str(run_dir),
        "--base-config",
        base_config,
        "--split",
        split,
        "--backend",
        backend,
        "--checkpoint-name",
        checkpoint_name,
    ]
    if tag:
        cmd.extend(["--tag", tag])
    if dbscan_hparams is not None:
        cmd.extend(["--dbscan-hparams", str(dbscan_hparams)])
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    suffix = f"_{tag}" if tag else ""
    name = f"paper_metrics_{split}{suffix}.json"
    return run_dir / "logs" / name


def dbscan_hparams_path(run_dir: Path, tag: str | None) -> Path:
    suffix = f"_{tag}" if tag else ""
    return run_dir / "logs" / f"dbscan_hparams{suffix}.json"


def load_metrics_json(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--base-config", type=str, required=True)
    p.add_argument("--backend", type=str, default="ellphi", choices=["ellphi", "mahalanobis"])
    p.add_argument("--tag", type=str, default="topo_best")
    p.add_argument("--baseline-run-dir", type=Path, default=None)
    p.add_argument("--baseline-config", type=str, default=None)
    p.add_argument("--baseline-tag", type=str, default="topo_best")
    p.add_argument(
        "--compare-json",
        type=Path,
        default=None,
        help="Write side-by-side test metrics JSON (default: run-dir/logs/ellphi_tuned_vs_baseline_test.json)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metrics_path = run_dir / "logs" / "metrics.csv"
    if not metrics_path.is_file():
        raise SystemExit(f"missing {metrics_path}")

    saved_epochs = list_checkpoint_epochs(run_dir)
    if not saved_epochs:
        raise SystemExit(f"no checkpoint_epoch_*.pth under {run_dir}")

    ckpt, epoch, topo_loss, global_epoch, global_topo = best_topo_checkpoint(run_dir)
    ckpt_path = run_dir / ckpt
    if not ckpt_path.is_file():
        raise SystemExit(f"missing checkpoint {ckpt_path} (best saved-topo ep={epoch})")

    print(
        f"[topo] best saved epoch={epoch} train_topo_loss={topo_loss:.6f} checkpoint={ckpt}"
    )
    if global_epoch is not None and global_topo is not None:
        print(
            f"[topo] note: global train_topo min is ep={global_epoch} "
            f"(loss={global_topo:.6f}) but no checkpoint on disk"
        )

    run_eval(
        run_dir,
        args.base_config,
        "val",
        args.backend,
        ckpt,
        tag=args.tag,
    )
    hparams_path = dbscan_hparams_path(run_dir, args.tag)
    if not hparams_path.is_file():
        raise SystemExit(f"missing {hparams_path} (expected after val DBSCAN grid search)")
    test_json = run_eval(
        run_dir,
        args.base_config,
        "test",
        args.backend,
        ckpt,
        tag=args.tag,
        dbscan_hparams=hparams_path,
    )
    tuned = load_metrics_json(test_json)
    print(f"[test] MCC={tuned['mcc']:.4f} W-Dist={tuned['wdist']:.4f} recall={tuned['recall']:.4f}")

    out: dict = {
        "run_dir": str(run_dir),
        "base_config": args.base_config,
        "backend": args.backend,
        "best_topo_epoch": epoch,
        "train_topo_loss": topo_loss,
        "checkpoint_name": ckpt,
        "test_metrics": tuned,
        "val_dbscan_hparams": json.loads(hparams_path.read_text()),
    }

    if args.baseline_run_dir is not None:
        base_dir = args.baseline_run_dir.resolve()
        base_config = args.baseline_config or args.base_config
        base_metrics = None
        for candidate in (
            base_dir / "logs" / "paper_metrics_test_topo_ep30.json",
            base_dir / "logs" / f"paper_metrics_test_{args.baseline_tag}.json",
        ):
            if candidate.is_file():
                base_metrics = load_metrics_json(candidate)
                print(f"[baseline] loaded existing {candidate}")
                break
        if base_metrics is None:
            base_saved = list_checkpoint_epochs(base_dir)
            if not base_saved:
                raise SystemExit(f"no checkpoint_epoch_*.pth under {base_dir}")
            base_epoch, base_topo = best_topo_epoch(
                base_dir / "logs" / "metrics.csv",
                saved_epochs=base_saved,
            )
            base_ckpt = f"checkpoint_epoch_{base_epoch}.pth"
            print(
                f"[baseline] evaluating ep={base_epoch} train_topo={base_topo:.6f} ckpt={base_ckpt}"
            )
            run_eval(base_dir, base_config, "val", args.backend, base_ckpt, tag=args.baseline_tag)
            hp = dbscan_hparams_path(base_dir, args.baseline_tag)
            base_test_path = run_eval(
                base_dir,
                base_config,
                "test",
                args.backend,
                base_ckpt,
                tag=args.baseline_tag,
                dbscan_hparams=hp,
            )
            base_metrics = load_metrics_json(base_test_path)
        baseline = base_metrics
        out["baseline"] = {
            "run_dir": str(base_dir),
            "base_config": base_config,
            "test_metrics": baseline,
        }
        print(
            f"[baseline test] MCC={baseline['mcc']:.4f} W-Dist={baseline['wdist']:.4f} "
            f"recall={baseline['recall']:.4f}"
        )

    compare_path = args.compare_json or (run_dir / "logs" / "ellphi_tuned_vs_baseline_test.json")
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    compare_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
