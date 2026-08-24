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
    PAPER_NO_CLS_CONTRACT,
    PAPER_NO_CLS_ELONGATE_CONTRACT,
    RINGS_NO_CLS_CONTRACT,
    preflight_tune_json,
)
from tda_ml.reproducibility import RUN_STATUS_EMPTY_RESULT, RUN_STATUS_FAILED
from tda_ml.run_paths import assert_no_legacy_paper_run_namespace
from tda_ml.supervised_diagnostics import git_revision
from tda_ml.val_topo_cliff import (
    RINGS_VAL_TOPO_CLIFF_MAX,
    evaluate_run_cliff,
)

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
    assert_no_legacy_paper_run_namespace(out_base)
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


def resolve_aggregate_contract(
    *,
    experiment_contract: str,
    aniso_variant: str,
    tune_payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Return ``(contract_name, contract_dict)`` for tune JSON preflight."""
    if experiment_contract == "auto":
        has_dataset_type = "dataset_type" in tune_payload
        dataset_type = str(tune_payload.get("dataset_type", "")).strip().lower()
        if dataset_type == "thin_rings":
            experiment_contract = "rings"
        elif dataset_type == "mnist":
            experiment_contract = "mnist"
        elif (
            not has_dataset_type
            and "tangent_pca_k" in tune_payload
        ):
            # Legacy MNIST near-tangent best JSON omitted dataset_type but
            # recorded tangent_* keys — treat as explicit mnist Methods.
            experiment_contract = "mnist"
        elif has_dataset_type:
            raise ValueError(
                f"Unrecognized tune JSON dataset_type={dataset_type!r}; "
                "use --experiment-contract mnist|rings"
            )
        else:
            raise ValueError(
                "tune JSON missing dataset_type; refusing silent mnist default "
                "for --experiment-contract auto. Pass --experiment-contract "
                "mnist|rings explicitly, or re-tune so the best JSON records "
                "dataset_type (thin_rings|mnist)."
            )

    if experiment_contract == "rings":
        if aniso_variant != "elongate_barrier":
            raise ValueError(
                "rings Methods contract requires --aniso-variant elongate_barrier "
                f"(got {aniso_variant!r})"
            )
        return "rings", RINGS_NO_CLS_CONTRACT

    if experiment_contract == "mnist":
        contract = (
            PAPER_NO_CLS_CONTRACT
            if aniso_variant == "elongate_barrier"
            else PAPER_NO_CLS_ELONGATE_CONTRACT
        )
        return "mnist", contract

    raise ValueError(
        f"--experiment-contract must be auto|mnist|rings; got {experiment_contract!r}"
    )


def resolve_cliff_policy(
    *,
    seed_cliff_policy: str,
    contract_name: str,
) -> str:
    if seed_cliff_policy == "auto":
        return "strict" if contract_name == "rings" else "off"
    if seed_cliff_policy not in ("off", "strict", "exclude-failed"):
        raise ValueError(
            f"--seed-cliff-policy must be auto|off|strict|exclude-failed; "
            f"got {seed_cliff_policy!r}"
        )
    return seed_cliff_policy


def annotate_seed_cliffs(
    by_seed: dict[int, dict[str, Any]],
    *,
    cliff_max: float,
) -> dict[int, dict[str, Any]]:
    annotated: dict[int, dict[str, Any]] = {}
    for seed, row in by_seed.items():
        report = evaluate_run_cliff(Path(row["run_dir"]), cliff_max=cliff_max)
        annotated[seed] = {**row, **report}
    return annotated


def apply_cliff_policy(
    *,
    method: str,
    by_seed: dict[int, dict[str, Any]],
    expected_seeds: Sequence[int],
    policy: str,
    cliff_max: float,
) -> tuple[list[int], list[int], dict[int, dict[str, Any]]]:
    """Return ``(seeds_for_mean, failed_seeds, annotated_by_seed)``."""
    annotated = annotate_seed_cliffs(by_seed, cliff_max=cliff_max)
    if policy == "off":
        return list(expected_seeds), [], annotated

    failed = [
        s
        for s in expected_seeds
        if s in annotated and not annotated[s]["val_topo_cliff_passed"]
    ]
    passed = [
        s
        for s in expected_seeds
        if s in annotated and annotated[s]["val_topo_cliff_passed"]
    ]
    if policy == "strict" and failed:
        detail = ", ".join(
            f"seed={s} best_val_topo={annotated[s]['best_val_topo_loss']:.4f}"
            for s in failed
        )
        raise ValueError(
            f"{method}: val_topo cliff failed for seeds [{detail}] "
            f"(cliff_max={cliff_max}). Re-tune/re-run those seeds, or pass "
            "--seed-cliff-policy exclude-failed for a diagnostic survivor-only summary."
        )
    if policy == "exclude-failed":
        if len(passed) < 2:
            raise ValueError(
                f"{method}: after excluding cliff-failed seeds {failed}, "
                f"only {len(passed)} survivor(s); refusing aggregate "
                "(need >=2 for mean±std)"
            )
        return passed, failed, annotated
    return list(expected_seeds), failed, annotated


def aggregate_row(
    *,
    method: str,
    by_seed: dict[int, dict[str, Any]],
    expected_seeds: Sequence[int],
    tune_json: str,
    seeds_for_mean: Sequence[int] | None = None,
    cliff_failed_seeds: Sequence[int] | None = None,
    cliff_policy: str = "off",
    cliff_max: float | None = None,
) -> dict[str, Any]:
    mean_seeds = list(seeds_for_mean) if seeds_for_mean is not None else list(expected_seeds)
    missing = [s for s in mean_seeds if s not in by_seed]
    if missing:
        raise ValueError(
            f"{method}: missing seeds {missing}; refusing partial aggregate"
        )
    expected_tune = str(Path(tune_json).resolve())
    mismatched = []
    for seed in mean_seeds:
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
    seeds_present = [by_seed[s] for s in mean_seeds]

    mccs = [r["mcc"] for r in seeds_present]
    gmeans = [r["gmean"] for r in seeds_present]
    wdist = [r["wdist"] for r in seeds_present]
    n = len(seeds_present)
    failed = list(cliff_failed_seeds or [])
    notes = (
        f"{n}/{len(expected_seeds)} seeds in mean; "
        f"fixed tune weights from {expected_tune}; "
        "30ep val_topo ckpt; maha DBSCAN eval; "
        "wdist_* are diagnostics only (not main-table columns)"
    )
    if cliff_policy != "off":
        notes += (
            f"; cliff_policy={cliff_policy} cliff_max={cliff_max}; "
            f"cliff_failed_seeds={failed}"
        )
    return {
        "method": method,
        "mcc_mean": float(np.mean(mccs)),
        "mcc_std": sample_std(mccs),
        "gmean_mean": float(np.mean(gmeans)),
        "gmean_std": sample_std(gmeans),
        "wdist_mean": float(np.mean(wdist)),
        "wdist_std": sample_std(wdist),
        "notes": notes,
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
        default="elongate_barrier",
        help=(
            "Declared paper contract variant the tune JSONs must match "
            "(default elongate_barrier; elongate = ablation without aspect ceiling)."
        ),
    )
    p.add_argument(
        "--experiment-contract",
        choices=["auto", "mnist", "rings"],
        default="auto",
        help=(
            "Which declared contract the tune JSON must match. "
            "auto: thin_rings → RINGS_NO_CLS_CONTRACT, else MNIST paper contract."
        ),
    )
    p.add_argument(
        "--seed-cliff-policy",
        choices=["auto", "off", "strict", "exclude-failed"],
        default="auto",
        help=(
            "How to treat seeds that never cross the val_topo cliff. "
            "auto: strict for rings, off for MNIST. "
            "strict: hard-fail if any seed fails. "
            "exclude-failed: mean only over survivors (>=2)."
        ),
    )
    p.add_argument(
        "--val-topo-cliff-max",
        type=float,
        default=None,
        help=(
            "Cliff threshold for --seed-cliff-policy (default: "
            f"{RINGS_VAL_TOPO_CLIFF_MAX} when policy is active)."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tune_paths = {
        "wdist": Path(args.wdist_tune_json),
        "mcc": Path(args.mcc_tune_json),
    }
    # Resolve contract from the first requested method's tune JSON.
    first_key = args.methods[0]
    first_path = tune_paths[first_key]
    if not first_path.is_absolute():
        first_path = REPO_ROOT / first_path
    first_payload = json.loads(first_path.read_text(encoding="utf-8"))
    contract_name, expected_contract = resolve_aggregate_contract(
        experiment_contract=args.experiment_contract,
        aniso_variant=args.aniso_variant,
        tune_payload=first_payload,
    )
    cliff_policy = resolve_cliff_policy(
        seed_cliff_policy=args.seed_cliff_policy,
        contract_name=contract_name,
    )
    cliff_max = (
        float(args.val_topo_cliff_max)
        if args.val_topo_cliff_max is not None
        else float(RINGS_VAL_TOPO_CLIFF_MAX)
    )

    for key in args.methods:
        path = tune_paths[key]
        if not path.is_absolute():
            path = REPO_ROOT / path
        payload = preflight_tune_json(path, expected_contract=expected_contract)
        # Cross-check auto/explicit contract agreement across methods.
        other_name, _ = resolve_aggregate_contract(
            experiment_contract=args.experiment_contract,
            aniso_variant=args.aniso_variant,
            tune_payload=payload,
        )
        if other_name != contract_name:
            raise ValueError(
                f"Tune JSON contract mismatch across methods: {first_key}={contract_name}, "
                f"{key}={other_name}"
            )
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
    cliff_annotations: dict[str, Any] = {}
    for key in args.methods:
        method, by_seed, tune_json = method_specs[key]
        mean_seeds, failed_seeds, annotated = apply_cliff_policy(
            method=method,
            by_seed=by_seed,
            expected_seeds=args.seeds,
            policy=cliff_policy,
            cliff_max=cliff_max,
        )
        cliff_annotations[key] = {
            "seeds_in_mean": mean_seeds,
            "cliff_failed_seeds": failed_seeds,
            "per_seed": {
                str(s): {
                    "val_topo_cliff_passed": annotated[s]["val_topo_cliff_passed"],
                    "best_val_topo_loss": annotated[s]["best_val_topo_loss"],
                    "best_val_topo_epoch": annotated[s]["best_val_topo_epoch"],
                    "run_status": (
                        "completed"
                        if annotated[s]["val_topo_cliff_passed"]
                        else RUN_STATUS_EMPTY_RESULT
                    ),
                }
                for s in args.seeds
                if s in annotated
            },
        }
        # Keep discover payloads for manifest, but also surface cliff fields.
        if key == "wdist":
            wdist_by_seed = annotated
        else:
            mcc_by_seed = annotated
        rows.append(
            aggregate_row(
                method=method,
                by_seed=annotated,
                expected_seeds=args.seeds,
                tune_json=tune_json,
                seeds_for_mean=mean_seeds,
                cliff_failed_seeds=failed_seeds,
                cliff_policy=cliff_policy,
                cliff_max=cliff_max if cliff_policy != "off" else None,
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
        "experiment_contract": contract_name,
        "paper_no_cls_contract": expected_contract,
        "seed_cliff_policy": cliff_policy,
        "val_topo_cliff_max": cliff_max if cliff_policy != "off" else None,
        "cliff_annotations": cliff_annotations,
        "per_seed": {
            "wdist": wdist_by_seed,
            "mcc": mcc_by_seed,
        },
        "warnings": (
            [
                f"cliff_policy=exclude-failed; survivors-only mean "
                f"(failed={cliff_annotations.get('wdist', {}).get('cliff_failed_seeds')})"
            ]
            if cliff_policy == "exclude-failed"
            else []
        ),
        "summary_csv": str(summary_path),
        "run_status_note": (
            f"cliff-failed seeds recorded as {RUN_STATUS_EMPTY_RESULT}/{RUN_STATUS_FAILED}"
            if cliff_policy != "off"
            else None
        ),
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
