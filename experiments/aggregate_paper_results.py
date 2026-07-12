#!/usr/bin/env python3
"""
Aggregate paper Table~\\ref{tab:comparison} metrics into ``summary_for_paper.csv``.

Reads test-split DBSCAN metrics (``logs/paper_metrics_test.json`` from
``evaluate_paper_protocol.py``) for proposed / ablation runs, and
``summary_baselines.csv`` from Phase 3.

Usage::

    uv run python experiments/aggregate_paper_results.py \\
        --proposed-dir outputs/supervised/20260627/055936_paper_reproduce_1week_tuned \\
        --wtopo0-dir outputs/paper_wtopo0 \\
        --baselines-csv outputs/supervised/20260623/155756_paper_baselines/summary_baselines.csv

Writes ``summary_for_paper.csv`` and ``MANIFEST.md`` under ``--out-dir``
(default: ``--proposed-dir``).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tda_ml.supervised_diagnostics import git_revision

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_SEEDS = [42, 123, 456, 789, 1024]

SUMMARY_COLUMNS = [
    "method",
    "mcc_mean",
    "mcc_std",
    "gmean_mean",
    "gmean_std",
    "wdist_mean",
    "wdist_std",
    "notes",
]

METHOD_ORDER = [
    "proposed_w_topo_pos",
    "proposed_w_topo_0",
    "euclidean_dbscan",
    "isolation_forest",
    "lof",
    "adbscan",
]

# ``evaluate_paper_protocol`` writes SplitMetrics fields (mcc, gmean, wdist).
# Mac-side stubs may use alternate names — try all aliases.
MCC_KEYS = ("mcc", "test_mcc", "mcc_mean", "mean_mcc")
GMEAN_KEYS = ("gmean", "g_mean", "test_gmean", "gmean_mean", "mean_gmean")
WDIST_KEYS = ("wdist", "w_dist", "test_wdist", "wdist_mean", "mean_wdist", "w_dist_mean")


@dataclass
class SeedPaperMetrics:
    seed: int | None
    run_dir: Path
    metrics_path: Path
    mcc: float
    gmean: float
    wdist: float
    n_clouds: int | None
    dbscan_eps: float | None
    dbscan_min_samples: int | None
    backend: str | None
    source_revision: str | None


def sample_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


def _first_key(payload: dict[str, Any], keys: Sequence[str], *, label: str) -> float:
    for key in keys:
        if key in payload and payload[key] is not None:
            return float(payload[key])
    raise KeyError(f"Missing {label} in JSON (tried {list(keys)}); keys={sorted(payload)}")


def load_paper_metrics_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") not in (None, "test"):
        raise ValueError(f"Expected test-split metrics in {path}; got split={payload.get('split')!r}")
    return payload


def parse_seed_paper_metrics(run_dir: Path, metrics_path: Path) -> SeedPaperMetrics:
    payload = load_paper_metrics_json(metrics_path)
    manifest_path = run_dir / "logs" / "run_manifest.json"
    seed: int | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("seed") is not None:
            seed = int(manifest["seed"])

    return SeedPaperMetrics(
        seed=seed,
        run_dir=run_dir.resolve(),
        metrics_path=metrics_path.resolve(),
        mcc=_first_key(payload, MCC_KEYS, label="mcc"),
        gmean=_first_key(payload, GMEAN_KEYS, label="gmean"),
        wdist=_first_key(payload, WDIST_KEYS, label="wdist"),
        n_clouds=int(payload["n_clouds"]) if payload.get("n_clouds") is not None else None,
        dbscan_eps=float(payload["dbscan_eps"]) if payload.get("dbscan_eps") is not None else None,
        dbscan_min_samples=(
            int(payload["dbscan_min_samples"]) if payload.get("dbscan_min_samples") is not None else None
        ),
        backend=str(payload["backend"]) if payload.get("backend") is not None else None,
        source_revision=str(payload["source_revision"]) if payload.get("source_revision") else None,
    )


def discover_run_dirs(out_base: Path) -> list[Path]:
    """Return run directories with ``logs/paper_metrics_test.json``."""
    found: list[Path] = []
    progress = out_base / "progress_summary.csv"
    if progress.is_file():
        with progress.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("backend") not in (None, "", "ellphi"):
                    continue
                run_dir = Path(row["run_dir"])
                if (run_dir / "logs" / "paper_metrics_test.json").is_file():
                    found.append(run_dir.resolve())

    if not found:
        for pattern in ("eph_s*/logs/paper_metrics_test.json", "backend_ellphi_seed*/logs/paper_metrics_test.json"):
            for metrics_path in sorted(out_base.glob(pattern)):
                found.append(metrics_path.parent.parent.resolve())
        for metrics_path in sorted(out_base.glob("reproduce_*/logs/paper_metrics_test.json")):
            found.append(metrics_path.parent.parent.resolve())

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for rd in found:
        if rd not in seen:
            seen.add(rd)
            unique.append(rd)
    return unique


def aggregate_seed_metrics(
    seeds: list[SeedPaperMetrics],
    *,
    method: str,
    notes: str,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError(f"No seed metrics for method={method!r}")
    mccs = [s.mcc for s in seeds]
    gmeans = [s.gmean for s in seeds]
    wdist = [s.wdist for s in seeds]
    n_clouds = seeds[0].n_clouds
    cloud_note = f"test n_clouds={n_clouds} per seed" if n_clouds is not None else "test split"
    return {
        "method": method,
        "mcc_mean": float(np.mean(mccs)),
        "mcc_std": sample_std(mccs),
        "gmean_mean": float(np.mean(gmeans)),
        "gmean_std": sample_std(gmeans),
        "wdist_mean": float(np.mean(wdist)),
        "wdist_std": sample_std(wdist),
        "notes": notes or f"{len(seeds)} seeds; DBSCAN test metrics; {cloud_note}",
    }


def load_proposed_row(
    out_base: Path,
    *,
    method: str,
    default_notes: str,
    expected_seeds: Sequence[int],
) -> tuple[dict[str, Any] | None, list[SeedPaperMetrics], list[str]]:
    warnings: list[str] = []
    run_dirs = discover_run_dirs(out_base)
    if not run_dirs:
        warnings.append(f"{method}: no run dirs with paper_metrics_test.json under {out_base}")
        return None, [], warnings

    per_seed: list[SeedPaperMetrics] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "logs" / "paper_metrics_test.json"
        try:
            per_seed.append(parse_seed_paper_metrics(run_dir, metrics_path))
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"{method}: skip {run_dir}: {exc}")

    if not per_seed:
        warnings.append(f"{method}: no parseable paper_metrics_test.json under {out_base}")
        return None, [], warnings

    found_seeds = {s.seed for s in per_seed if s.seed is not None}
    missing = [s for s in expected_seeds if s not in found_seeds]
    if missing:
        warnings.append(f"{method}: missing seeds {missing} (found {sorted(found_seeds)})")

    row = aggregate_seed_metrics(per_seed, method=method, notes=default_notes)
    return row, per_seed, warnings


def load_baselines_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Baselines CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for col in SUMMARY_COLUMNS:
            if col not in row:
                row[col] = ""
    return rows


def write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in SUMMARY_COLUMNS})


def format_pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_manifest(
    path: Path,
    *,
    summary_csv: Path,
    proposed_dir: Path,
    wtopo0_dir: Path | None,
    baselines_csv: Path,
    rows: Sequence[dict[str, Any]],
    proposed_seeds: list[SeedPaperMetrics],
    wtopo0_seeds: list[SeedPaperMetrics],
    warnings: Sequence[str],
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Paper experiment manifest",
        "",
        f"- Generated: {now}",
        f"- Git HEAD: `{git_revision(REPO_ROOT)}`",
        f"- Summary CSV: `{summary_csv}`",
        f"- Proposed runs: `{proposed_dir}`",
        f"- w_topo=0 ablation: `{wtopo0_dir}`" if wtopo0_dir else "- w_topo=0 ablation: (not provided)",
        f"- Baselines CSV: `{baselines_csv}`",
        "",
        "## Metrics source",
        "",
        "- Proposed / ablation: `evaluate_paper_protocol.py` → `logs/paper_metrics_test.json`",
        "  (keys: `mcc`, `gmean`, `wdist` from DBSCAN test evaluation)",
        "- Baselines: `evaluate_paper_baselines.py` → `summary_baselines.csv`",
        "",
        "## summary_for_paper.csv",
        "",
        "| method | MCC | G-Mean | W-Dist | notes |",
        "|--------|-----|--------|--------|-------|",
    ]

    for row in rows:
        method = row.get("method", "")
        if not row.get("mcc_mean"):
            lines.append(f"| {method} | — | — | — | pending |")
            continue
        lines.append(
            f"| {method} | {format_pm(float(row['mcc_mean']), float(row['mcc_std']))} "
            f"| {format_pm(float(row['gmean_mean']), float(row['gmean_std']))} "
            f"| {format_pm(float(row['wdist_mean']), float(row['wdist_std']))} "
            f"| {row.get('notes', '')} |"
        )

    def _seed_section(title: str, seeds: Sequence[SeedPaperMetrics]) -> list[str]:
        if not seeds:
            return [f"## {title}", "", "(not available)", ""]
        out = [f"## {title}", ""]
        for s in sorted(seeds, key=lambda x: (x.seed is None, x.seed or 0)):
            hparam = ""
            if s.dbscan_eps is not None and s.dbscan_min_samples is not None:
                hparam = f" eps={s.dbscan_eps}, min_samples={s.dbscan_min_samples}"
            seed_label = s.seed if s.seed is not None else "?"
            out.append(
                f"- seed {seed_label}: `{s.run_dir}` — "
                f"MCC={s.mcc:.4f}, G-Mean={s.gmean:.4f}, W-Dist={s.wdist:.4f}{hparam}"
            )
        out.append("")
        return out

    lines.extend(_seed_section("Proposed per-seed (w_topo > 0)", proposed_seeds))
    lines.extend(_seed_section("Ablation per-seed (w_topo = 0)", wtopo0_seeds))

    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--proposed-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "supervised" / "20260627" / "055936_paper_reproduce_1week_tuned",
        help="Main proposed runs (12ep × 5 seed, ellphi).",
    )
    p.add_argument(
        "--wtopo0-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "paper_wtopo0",
        help="w_topo=0 ablation output tree (optional until Phase 2 completes).",
    )
    p.add_argument(
        "--baselines-csv",
        type=Path,
        default=REPO_ROOT / "outputs" / "supervised" / "20260623" / "155756_paper_baselines" / "summary_baselines.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write summary_for_paper.csv and MANIFEST.md here (default: --proposed-dir).",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    p.add_argument(
        "--skip-wtopo0",
        action="store_true",
        help="Do not require w_topo=0 ablation row.",
    )
    p.add_argument(
        "--require-proposed",
        action="store_true",
        help="Exit non-zero if proposed runs are missing.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    proposed_dir = args.proposed_dir.resolve()
    wtopo0_dir = args.wtopo0_dir.resolve() if args.wtopo0_dir else None
    baselines_csv = args.baselines_csv.resolve()
    out_dir = (args.out_dir or proposed_dir).resolve()
    warnings: list[str] = []

    proposed_row, proposed_seeds, w1 = load_proposed_row(
        proposed_dir,
        method="proposed_w_topo_pos",
        default_notes="proposed; ellphi DBSCAN test metrics; val-tuned eps/min_samples per seed",
        expected_seeds=args.seeds,
    )
    warnings.extend(w1)

    wtopo0_row: dict[str, Any] | None = None
    wtopo0_seeds: list[SeedPaperMetrics] = []
    if not args.skip_wtopo0 and wtopo0_dir is not None:
        wtopo0_row, wtopo0_seeds, w2 = load_proposed_row(
            wtopo0_dir,
            method="proposed_w_topo_0",
            default_notes="ablation w_topo=0; ellphi DBSCAN test metrics; val-tuned eps/min_samples per seed",
            expected_seeds=args.seeds,
        )
        warnings.extend(w2)
    elif not args.skip_wtopo0:
        warnings.append("proposed_w_topo_0: --wtopo0-dir not set")

    baseline_rows = load_baselines_rows(baselines_csv)
    baseline_by_method = {r["method"]: r for r in baseline_rows}

    rows_by_method: dict[str, dict[str, Any]] = {}
    if proposed_row:
        rows_by_method["proposed_w_topo_pos"] = proposed_row
    if wtopo0_row:
        rows_by_method["proposed_w_topo_0"] = wtopo0_row
    for method in ("euclidean_dbscan", "isolation_forest", "lof", "adbscan"):
        if method in baseline_by_method:
            rows_by_method[method] = baseline_by_method[method]

    ordered_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        if method in rows_by_method:
            ordered_rows.append(rows_by_method[method])
        else:
            warnings.append(f"metrics pending: {method}")
            ordered_rows.append({"method": method, "notes": "pending"})

    summary_path = out_dir / "summary_for_paper.csv"
    manifest_path = out_dir / "MANIFEST.md"
    write_summary_csv(summary_path, ordered_rows)
    write_manifest(
        manifest_path,
        summary_csv=summary_path,
        proposed_dir=proposed_dir,
        wtopo0_dir=wtopo0_dir if not args.skip_wtopo0 else None,
        baselines_csv=baselines_csv,
        rows=ordered_rows,
        proposed_seeds=proposed_seeds,
        wtopo0_seeds=wtopo0_seeds,
        warnings=warnings,
    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {manifest_path}")
    for row in ordered_rows:
        if row.get("mcc_mean"):
            print(
                f"  {row['method']}: MCC={float(row['mcc_mean']):.4f}±{float(row['mcc_std']):.4f}  "
                f"G-Mean={float(row['gmean_mean']):.4f}±{float(row['gmean_std']):.4f}  "
                f"W-Dist={float(row['wdist_mean']):.4f}±{float(row['wdist_std']):.4f}"
            )
        else:
            print(f"  {row['method']}: pending")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    if args.require_proposed and proposed_row is None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
