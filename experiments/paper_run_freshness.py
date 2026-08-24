#!/usr/bin/env python3
"""Freshness gate for multiseed 30ep skip decisions.

Exit codes:
  0 — matching metrics exist (safe to skip)
  1 — no metrics (must run)
  2 — hard-fail: stale / ambiguous / corrupt / leftover ``pwr_s*`` namespace /
      val_topo cliff empty-result / any inspection error. Do not skip and do
      not start a new run. Exit 1 is reserved for genuine missing ``paper_s*``
      metrics so the 30ep driver never treats a refused tree as "missing".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tda_ml.run_paths import assert_no_legacy_paper_run_namespace
from tda_ml.supervised_diagnostics import git_revision
from tda_ml.val_topo_cliff import (
    RINGS_VAL_TOPO_CLIFF_MAX,
    evaluate_run_cliff,
)

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
    require_val_topo_cliff: bool = False,
    val_topo_cliff_max: float = RINGS_VAL_TOPO_CLIFF_MAX,
) -> tuple[str, list[Path]]:
    """Return ``(status, paths)`` where status is fresh|missing|stale|ambiguous.

    Cliff failure is never ``missing``. A completed run that missed the cliff
    is ``empty-result`` and raises so the CLI exits 2.
    """
    assert_no_legacy_paper_run_namespace(out_base)
    pattern = f"paper_s{seed}_*/logs/paper_metrics_test_{tag}.json"
    matches = sorted(out_base.glob(pattern))
    if not matches:
        return "missing", []

    expected_tune = _resolve_tune_json(str(tune_json))
    expected_rev = expected_revision or git_revision(REPO_ROOT)
    fresh: list[Path] = []
    stale_reasons: list[str] = []
    cliff_fail_reasons: list[str] = []

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
        if require_val_topo_cliff:
            try:
                cliff = evaluate_run_cliff(run_dir, cliff_max=val_topo_cliff_max)
            except (FileNotFoundError, ValueError) as exc:
                stale_reasons.append(f"{metrics_path}: cliff inspect failed ({exc})")
                continue
            if not cliff["val_topo_cliff_passed"]:
                cliff_fail_reasons.append(
                    f"{metrics_path}: val_topo cliff failed "
                    f"(best={cliff['best_val_topo_loss']:.4f} > {val_topo_cliff_max})"
                )
                continue
        fresh.append(metrics_path)

    if len(fresh) == 1 and not stale_reasons and not cliff_fail_reasons:
        return "fresh", fresh
    if len(fresh) > 1:
        return "ambiguous", fresh
    if fresh and (stale_reasons or cliff_fail_reasons):
        return "ambiguous", fresh
    if require_val_topo_cliff and not fresh and cliff_fail_reasons and not stale_reasons:
        raise RuntimeError(
            "val_topo cliff failed (empty-result), not missing: "
            + "; ".join(cliff_fail_reasons)
        )
    return "stale", matches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-base", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--tune-json", type=Path, required=True)
    p.add_argument(
        "--require-val-topo-cliff",
        action="store_true",
        help="Only treat cliff-passed runs as fresh (rings Methods).",
    )
    p.add_argument(
        "--val-topo-cliff-max",
        type=float,
        default=RINGS_VAL_TOPO_CLIFF_MAX,
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        status, paths = inspect_seed_metrics(
            out_base=args.out_base,
            seed=args.seed,
            tag=args.tag,
            tune_json=args.tune_json,
            require_val_topo_cliff=bool(args.require_val_topo_cliff),
            val_topo_cliff_max=float(args.val_topo_cliff_max),
        )
    except Exception as exc:
        # Fail closed. Exit 1 is reserved for genuine missing paper_s* metrics.
        print(f"error: freshness check failed: {exc}", file=sys.stderr)
        return 2
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
