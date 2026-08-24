#!/usr/bin/env python3
"""Freshness gate for multiseed 30ep skip decisions.

Exit codes:
  0 — matching metrics exist (safe to skip)
  1 — no metrics (must run)
  2 — stale / ambiguous / corrupt (hard-fail; do not skip and do not silently reuse)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tda_ml.supervised_diagnostics import git_revision
from tda_ml.run_paths import assert_no_legacy_paper_run_namespace

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_tune_json(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def inspect_seed_metrics(
    *,
    out_base: Path,
    seed: int,
    tag: str,
    tune_json: Path,
    expected_revision: str | None = None,
) -> tuple[str, list[Path]]:
    """Return ``(status, paths)`` where status is fresh|missing|stale|ambiguous."""
    assert_no_legacy_paper_run_namespace(out_base, seed=seed)
    pattern = f"paper_s{seed}_*/logs/paper_metrics_test_{tag}.json"
    matches = sorted(out_base.glob(pattern))
    if not matches:
        return "missing", []

    expected_tune = _resolve_tune_json(str(tune_json))
    expected_rev = expected_revision or git_revision(REPO_ROOT)
    fresh: list[Path] = []
    stale_reasons: list[str] = []

    for metrics_path in matches:
        run_dir = metrics_path.parent.parent
        manifest_path = run_dir / "logs" / "run_manifest.json"
        if not manifest_path.is_file():
            stale_reasons.append(f"{metrics_path}: missing run_manifest.json")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("seed", -1)) != int(seed):
            stale_reasons.append(
                f"{metrics_path}: manifest seed {manifest.get('seed')!r} != {seed}"
            )
            continue
        manifest_tune = _resolve_tune_json(manifest.get("tune_json"))
        if manifest_tune != expected_tune:
            stale_reasons.append(
                f"{metrics_path}: tune_json {manifest_tune!r} != {expected_tune!r}"
            )
            continue
        rev = manifest.get("source_revision")
        if rev != expected_rev:
            stale_reasons.append(
                f"{metrics_path}: source_revision {rev!r} != {expected_rev!r}"
            )
            continue
        fresh.append(metrics_path)

    if len(fresh) == 1 and not stale_reasons:
        return "fresh", fresh
    if len(fresh) > 1:
        return "ambiguous", fresh
    if fresh and stale_reasons:
        return "ambiguous", fresh
    return "stale", matches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-base", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--tune-json", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    status, paths = inspect_seed_metrics(
        out_base=args.out_base,
        seed=args.seed,
        tag=args.tag,
        tune_json=args.tune_json,
    )
    if status == "fresh":
        print(f"[fresh] seed={args.seed} metrics={paths[0]}")
        return 0
    if status == "missing":
        print(f"[missing] seed={args.seed} (will run)")
        return 1
    detail = ", ".join(str(p) for p in paths)
    print(
        f"error: seed={args.seed} metrics are {status}: {detail}. "
        "Delete stale runs or align tune-json/revision before re-running.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
