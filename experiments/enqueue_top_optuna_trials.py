#!/usr/bin/env python3
"""Create a refinement study by enqueueing top trials from a proxy study."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import optuna
from optuna.trial import TrialState

from tda_ml.supervised_diagnostics import git_revision

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PARAMS = ("w_aniso", "w_size", "w_topo", "lr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-storage", required=True)
    parser.add_argument("--source-study", required=True)
    parser.add_argument("--target-storage", required=True)
    parser.add_argument("--target-study", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument(
        "--rank-start",
        type=int,
        default=1,
        help="1-based first proxy rank to enqueue (default: 1).",
    )
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Append selected trials to an existing target study (recovery only).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError(f"top-k must be >= 1, got {args.top_k}")
    if args.rank_start < 1:
        raise ValueError(f"rank-start must be >= 1, got {args.rank_start}")
    if args.out_dir.exists() and not args.append_existing:
        raise FileExistsError(f"Refinement output already exists: {args.out_dir}")

    source = optuna.load_study(
        study_name=args.source_study,
        storage=args.source_storage,
    )
    complete = sorted(
        (trial for trial in source.trials if trial.state == TrialState.COMPLETE),
        key=lambda trial: float(trial.value),
    )
    rank_stop = args.rank_start - 1 + args.top_k
    if len(complete) < rank_stop:
        raise RuntimeError(
            f"Source study has {len(complete)} completed trials; "
            f"need ranks {args.rank_start}..{rank_stop}"
        )

    selected = complete[args.rank_start - 1 : rank_stop]
    for trial in selected:
        missing = [name for name in REQUIRED_PARAMS if name not in trial.params]
        if missing:
            raise ValueError(
                f"Source trial {trial.number} is missing parameters {missing}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=args.append_existing)
    target = optuna.create_study(
        study_name=args.target_study,
        storage=args.target_storage,
        direction="minimize",
        load_if_exists=args.append_existing,
    )
    for trial in selected:
        target.enqueue_trial(
            {name: float(trial.params[name]) for name in REQUIRED_PARAMS},
            user_attrs={
                "source_study": args.source_study,
                "source_trial_number": trial.number,
                "source_proxy_value": float(trial.value),
            },
        )

    manifest = {
        "run_status": "pending",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_entry": "experiments/enqueue_top_optuna_trials.py",
        "source_revision": git_revision(REPO_ROOT),
        "source_storage": args.source_storage,
        "source_study": args.source_study,
        "target_storage": args.target_storage,
        "target_study": args.target_study,
        "selection": "lowest proxy W-Dist",
        "top_k": args.top_k,
        "rank_start": args.rank_start,
        "rank_stop": rank_stop,
        "append_existing": args.append_existing,
        "selected_trials": [
            {
                "source_trial_number": trial.number,
                "source_proxy_value": float(trial.value),
                "params": {
                    name: float(trial.params[name]) for name in REQUIRED_PARAMS
                },
            }
            for trial in selected
        ],
        "fallbacks": [],
    }
    if args.append_existing:
        # Recovery append: keep the original preflight intact for provenance.
        preflight_name = f"REFINEMENT_PREFLIGHT_rank{args.rank_start}-{rank_stop}.json"
    else:
        preflight_name = "REFINEMENT_PREFLIGHT.json"
    preflight_path = args.out_dir / preflight_name
    if preflight_path.exists():
        raise FileExistsError(f"Refinement preflight already exists: {preflight_path}")
    preflight_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["selected_trials"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
