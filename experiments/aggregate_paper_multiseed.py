#!/usr/bin/env python3
"""Aggregate multiseed 30ep proposed runs (W-Dist / MCC tune weights) for paper comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tda_ml.preflight import (
    PAPER_NO_CLS_BARRIER_CONTRACT,
    PAPER_NO_CLS_CONTRACT,
    preflight_tune_json,
)
from tda_ml.supervised_diagnostics import git_revision

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_SEEDS = [42, 123, 456, 789, 1024]

MCC_KEYS = ("mcc", "test_mcc")
GMEAN_KEYS = ("gmean", "g_mean", "test_gmean")
WDIST_KEYS = ("wdist", "w_dist", "test_wdist")


def sample_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        raise ValueError(
            f"sample_std requires at least 2 values; got {arr.size} "
            "(refusing silent zero std for incomplete seed sets)"
        )
    return float(np.std(arr, ddof=1))


def _first_key(payload: dict[str, Any], keys: Sequence[str], *, label: str) -> float:
    for key in keys:
        if key in payload and payload[key] is not None:
            return float(payload[key])
    raise KeyError(f"Missing {label}; keys={sorted(payload)}")


def discover_seed_metrics(out_base: Path, test_glob: str) -> dict[int, dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for metrics_path in sorted(out_base.glob(test_glob)):
        run_dir = metrics_path.parent.parent
        manifest_path = run_dir / "logs" / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                f"Missing run_manifest.json for {metrics_path}; "
                "refusing directory-name seed inference"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("seed") is None:
            raise ValueError(
                f"run_manifest.json missing seed for {metrics_path}; "
                "refusing directory-name seed inference"
            )
        seed = int(manifest["seed"])
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "seed": seed,
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path),
            "tune_json": manifest.get("tune_json"),
            "source_revision": manifest.get("source_revision"),
            "mcc": _first_key(payload, MCC_KEYS, label="mcc"),
            "gmean": _first_key(payload, GMEAN_KEYS, label="gmean"),
            "wdist": _first_key(payload, WDIST_KEYS, label="wdist"),
            "recall": float(payload["recall"]) if payload.get("recall") is not None else None,
        }
        if seed in by_seed:
            raise ValueError(
                f"Duplicate metrics for seed={seed}: "
                f"{by_seed[seed]['metrics_path']} and {metrics_path}; "
                "refusing silent last-wins selection"
            )
        by_seed[seed] = row
    return by_seed


def aggregate_row(
    *,
    method: str,
    by_seed: dict[int, dict[str, Any]],
    expected_seeds: Sequence[int],
    tune_json: str,
) -> dict[str, Any]:
    missing = [s for s in expected_seeds if s not in by_seed]
    if missing:
        raise ValueError(
            f"{method}: missing seeds {missing}; refusing partial aggregate"
        )
    expected_tune = str(Path(tune_json).resolve())
    mismatched = []
    for seed in expected_seeds:
        actual = by_seed[seed].get("tune_json")
        if actual is None:
            mismatched.append((seed, None))
            continue
        actual_resolved = str(Path(actual).resolve())
        if actual_resolved != expected_tune:
            mismatched.append((seed, actual_resolved))
    if mismatched:
        raise ValueError(
            f"{method}: per-seed tune_json mismatch vs aggregate {expected_tune}: "
            f"{mismatched}"
        )
    seeds_present = [by_seed[s] for s in expected_seeds]

    mccs = [r["mcc"] for r in seeds_present]
    gmeans = [r["gmean"] for r in seeds_present]
    wdist = [r["wdist"] for r in seeds_present]
    n = len(seeds_present)
    return {
        "method": method,
        "mcc_mean": float(np.mean(mccs)),
        "mcc_std": sample_std(mccs),
        "gmean_mean": float(np.mean(gmeans)),
        "gmean_std": sample_std(gmeans),
        "wdist_mean": float(np.mean(wdist)),
        "wdist_std": sample_std(wdist),
        "notes": (
            f"{n}/{len(expected_seeds)} seeds; fixed tune weights from {expected_tune}; "
            "30ep val_topo ckpt; maha DBSCAN eval; "
            "wdist_* are diagnostics only (not main-table columns)"
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "method",
        "mcc_mean",
        "mcc_std",
        "gmean_mean",
        "gmean_std",
        "wdist_mean",
        "wdist_std",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wdist-out", type=Path, default=REPO_ROOT / "outputs/supervised/paper30_wdist")
    p.add_argument("--mcc-out", type=Path, default=REPO_ROOT / "outputs/supervised/paper30_mcc")
    p.add_argument(
        "--wdist-tune-json",
        default="outputs/tune/wdist/best_wdist_ellphi.json",
    )
    p.add_argument(
        "--mcc-tune-json",
        default="outputs/tune/mcc_dbscan_mahalanobis/best_mcc_ellphi_dbscan_mahalanobis.json",
    )
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/supervised/paper30_multiseed")
    p.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["wdist", "mcc"],
        default=["wdist", "mcc"],
        help="Which tune objectives to aggregate (single-mode drivers pass one).",
    )
    p.add_argument(
        "--aniso-variant",
        choices=["elongate", "elongate_barrier"],
        default="elongate",
        help=(
            "Declared paper contract variant the tune JSONs must match "
            "(elongate_barrier = degeneracy-guard stack)."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    expected_contract = (
        PAPER_NO_CLS_BARRIER_CONTRACT
        if args.aniso_variant == "elongate_barrier"
        else PAPER_NO_CLS_CONTRACT
    )
    tune_paths = {
        "wdist": Path(args.wdist_tune_json),
        "mcc": Path(args.mcc_tune_json),
    }
    for key in args.methods:
        path = tune_paths[key]
        if not path.is_absolute():
            path = REPO_ROOT / path
        preflight_tune_json(path, expected_contract=expected_contract)
        tune_paths[key] = path.resolve()

    wdist_by_seed = discover_seed_metrics(
        args.wdist_out,
        "paper_s*/logs/paper_metrics_test_wdist.json",
    )
    mcc_by_seed = discover_seed_metrics(
        args.mcc_out,
        "paper_s*/logs/paper_metrics_test_mcc.json",
    )

    method_specs = {
        "wdist": ("proposed_wdist_30ep", wdist_by_seed, str(tune_paths["wdist"])),
        "mcc": ("proposed_mcc_30ep", mcc_by_seed, str(tune_paths["mcc"])),
    }
    rows: list[dict[str, Any]] = []
    for key in args.methods:
        method, by_seed, tune_json = method_specs[key]
        rows.append(
            aggregate_row(
                method=method,
                by_seed=by_seed,
                expected_seeds=args.seeds,
                tune_json=tune_json,
            )
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary_proposed.csv"
    write_csv(summary_path, rows)

    manifest = {
        "source_revision": git_revision(REPO_ROOT),
        "seeds_expected": list(args.seeds),
        "wdist_out": str(args.wdist_out),
        "mcc_out": str(args.mcc_out),
        "wdist_tune_json": str(tune_paths["wdist"]),
        "mcc_tune_json": str(tune_paths["mcc"]),
        "paper_no_cls_contract": expected_contract,
        "per_seed": {
            "wdist": wdist_by_seed,
            "mcc": mcc_by_seed,
        },
        "warnings": [],
        "summary_csv": str(summary_path),
    }
    (out_dir / "MANIFEST_proposed.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for row in rows:
        print(
            f"{row['method']}: MCC={row['mcc_mean']:.4f}±{row['mcc_std']:.4f}  "
            f"G-Mean={row['gmean_mean']:.4f}±{row['gmean_std']:.4f}"
        )
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
