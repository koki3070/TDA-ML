#!/usr/bin/env python3
"""
Paper-aligned baseline evaluation.

Same data split as ``eval_paper.py`` under the paper
config (typically ``paper_mnist_h1``), seeds
42 / 123 / 456 / 789 / 1024: tune hyperparameters on validation clouds, report
MCC / G-Mean on test. Topo W-Dist is computed for diagnostics only and is
**not** a main-table column.

Methods:
  1. Euclidean DBSCAN (sklearn on raw coordinates)
  2. Isolation Forest
  3. Local Outlier Factor (LOF)
  4. ADBSCAN (local PCA ellipses + Mahalanobis ``apply_anisotropic_dbscan``;
     no learned corrections — not compute-matched to the proposed 30ep model)

Usage::

    uv run python experiments/eval_baselines.py \\
        --base-config paper_mnist_h1 \\
        --out-dir outputs/paper_baselines
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

from experiments.eval_paper import (
    CloudMetrics,
    _aggregate_classification_metrics,
    _aggregate_cloud_metrics,
    build_split_loader,
    dbscan_labels_to_outlier_pred,
)
from tda_ml.config import deep_update, load_config
from tda_ml.dbscan_eval import valid_clean_inliers
from tda_ml.local_pca import local_pca_ellipse_params as _local_pca_ellipse_params_torch
from tda_ml.metrics import (
    compute_recall_specificity_gmean_mcc,
    compute_recall_specificity_gmean_mcc_wdist,
)
from tda_ml.preflight import preflight_baseline_eval
from tda_ml.reproducibility import (
    RUN_STATUS_COMPLETED,
    baseline_grids_from_config,
    record_fallback,
    reproducibility_settings,
    write_json,
)
from tda_ml.supervised_diagnostics import git_revision
from tda_ml.topo_wdist import (
    TopoWdistOptions,
    compute_topo_wdist,
    topo_wdist_options_from_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_SEEDS = [42, 123, 456, 789, 1024]
LOCAL_PCA_K = 10


@dataclass
class CloudSample:
    points: np.ndarray
    labels_gt: np.ndarray
    clean_pc: np.ndarray
    adbscan_params: np.ndarray | None = None


@dataclass
class SeedResult:
    seed: int
    method: str
    hparams: dict[str, Any]
    val_mcc: float
    test_recall: float
    test_specificity: float
    test_gmean: float
    test_mcc: float
    test_wdist: float
    n_test_clouds: int


def local_pca_ellipse_params(points: np.ndarray, k: int = LOCAL_PCA_K) -> np.ndarray:
    """ADBSCAN ellipses via shared ``tda_ml.local_pca`` (normalize_axes=True)."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be (N, 2); got {points.shape}")
    if points.shape[0] < k:
        raise ValueError(f"Need at least k={k} points; got n={points.shape[0]}")
    x = torch.from_numpy(points.astype(np.float32))
    return _local_pca_ellipse_params_torch(x, k=k, normalize_axes=True).numpy()


def load_clouds(config: dict[str, Any], split: str, device: torch.device) -> list[CloudSample]:
    loader = build_split_loader(config, split, device)
    clouds: list[CloudSample] = []
    for data, labels, clean_pc in loader:
        data_np = data.numpy()
        labels_np = labels.numpy()
        clean_np = clean_pc.numpy()
        for b in range(data_np.shape[0]):
            points = data_np[b]
            params = local_pca_ellipse_params(points)
            clouds.append(
                CloudSample(
                    points=points,
                    labels_gt=labels_np[b],
                    clean_pc=clean_np[b],
                    adbscan_params=params,
                )
            )
    return clouds


def cloud_metrics_from_pred(
    labels_gt: np.ndarray,
    pred: np.ndarray,
    points: np.ndarray,
    clean_pc: np.ndarray,
) -> CloudMetrics:
    gt_inliers = valid_clean_inliers(clean_pc)
    recall, specificity, gmean, mcc, wdist = compute_recall_specificity_gmean_mcc_wdist(
        labels_gt,
        pred,
        points=points,
        gt_inliers=gt_inliers,
    )
    return CloudMetrics(recall, specificity, gmean, mcc, wdist)


def evaluate_euclidean_dbscan(
    cloud: CloudSample,
    *,
    eps: float,
    min_samples: int,
) -> CloudMetrics:
    db_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(cloud.points)
    pred = dbscan_labels_to_outlier_pred(db_labels)
    return cloud_metrics_from_pred(cloud.labels_gt, pred, cloud.points, cloud.clean_pc)


def evaluate_isolation_forest(
    cloud: CloudSample,
    *,
    contamination: float,
    random_state: int,
) -> CloudMetrics:
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    sk_pred = iso.fit_predict(cloud.points)
    pred = (sk_pred == -1).astype(np.int64)
    return cloud_metrics_from_pred(cloud.labels_gt, pred, cloud.points, cloud.clean_pc)


def evaluate_lof(
    cloud: CloudSample,
    *,
    n_neighbors: int,
    contamination: float,
) -> CloudMetrics:
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
    )
    sk_pred = lof.fit_predict(cloud.points)
    pred = (sk_pred == -1).astype(np.int64)
    return cloud_metrics_from_pred(cloud.labels_gt, pred, cloud.points, cloud.clean_pc)


def evaluate_adbscan(
    cloud: CloudSample,
    *,
    eps: float,
    min_samples: int,
    topo_options: TopoWdistOptions | None = None,
) -> CloudMetrics:
    """
    ADBSCAN baseline metrics for one cloud.

    ``topo_options=None`` is the val grid-search phase: selection uses MCC only,
    and the ellipse-filtration topo W-Dist does not depend on (eps, min_samples),
    so it is deliberately NOT computed there (``wdist=None``, never aggregated).
    The test phase passes explicit ``topo_options`` and reports the real W-Dist.
    """
    from tda_ml.dbscan import apply_anisotropic_dbscan

    if cloud.adbscan_params is None:
        raise ValueError("adbscan_params missing")
    db_labels = apply_anisotropic_dbscan(
        cloud.points,
        cloud.adbscan_params,
        eps=eps,
        min_samples=min_samples,
        metric="max",
        backend="mahalanobis",
    )
    pred = dbscan_labels_to_outlier_pred(db_labels)
    recall, specificity, gmean, mcc = compute_recall_specificity_gmean_mcc(
        cloud.labels_gt, pred
    )
    if topo_options is None:
        # Val MCC grid: ellipse W-Dist is independent of (eps, min_samples) and
        # is not a selection objective — leave unset rather than NaN-placeholder.
        return CloudMetrics(recall, specificity, gmean, mcc, wdist=None)
    wdist = float(
        compute_topo_wdist(
            cloud.points, cloud.adbscan_params, cloud.clean_pc, topo_options
        )
    )
    return CloudMetrics(recall, specificity, gmean, mcc, wdist)


def grid_search_clouds(
    clouds: Sequence[CloudSample],
    evaluate_fn: Callable[..., CloudMetrics],
    param_combos: Sequence[dict[str, Any]],
    *,
    desc: str,
    allow_skip_degenerate_grid_cells: bool = False,
    manifest_ref: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    # Select by max mean MCC among successful cells. Negative MCC is a valid
    # outcome (method worse than chance on this data); only hard-fail when every
    # cell errored or the combo list is empty.
    best_mcc: float | None = None
    best_params: dict[str, Any] | None = None
    grid_log: list[dict[str, Any]] = []
    n_ok = 0

    for params in tqdm(param_combos, desc=desc, leave=False):
        per_cloud: list[CloudMetrics] = []
        error: str | None = None
        for cloud in clouds:
            try:
                per_cloud.append(evaluate_fn(cloud, **params))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                per_cloud = []
                break
        if error is not None:
            entry = {**params, "status": "failed", "error": error}
            grid_log.append(entry)
            if allow_skip_degenerate_grid_cells:
                if manifest_ref is not None:
                    record_fallback(
                        manifest_ref,
                        "baseline_grid_cell_skip",
                        f"{desc} {params}: {error}",
                    )
                continue
            raise RuntimeError(
                f"Baseline grid cell failed ({desc} {params}): {error}. "
                "Set reproducibility.allow_skip_degenerate_grid_cells=true to opt in "
                "to skipping failed cells."
            )
        _, _, _, mcc = _aggregate_classification_metrics(per_cloud)
        n_ok += 1
        grid_log.append(
            {**params, "status": "ok", "mean_mcc": mcc, "n_clouds": len(per_cloud)}
        )
        if best_mcc is None or mcc > best_mcc:
            best_mcc = mcc
            best_params = dict(params)

    if n_ok == 0 or best_params is None or best_mcc is None:
        raise RuntimeError(
            f"Grid search failed for all hparams ({desc}); log={grid_log[:5]}"
        )
    return best_params, best_mcc, grid_log


def dbscan_param_combos(
    eps_values: Sequence[float],
    min_samples_values: Sequence[int],
) -> list[dict[str, Any]]:
    return [
        {"eps": float(eps), "min_samples": int(ms)}
        for eps in eps_values
        for ms in min_samples_values
    ]


def contamination_param_combos(contamination_values: Sequence[float]) -> list[dict[str, Any]]:
    return [{"contamination": float(c)} for c in contamination_values]


def lof_param_combos(
    n_neighbors_values: Sequence[int],
    contamination_values: Sequence[float],
) -> list[dict[str, Any]]:
    return [
        {"n_neighbors": int(n), "contamination": float(c)}
        for n in n_neighbors_values
        for c in contamination_values
    ]


def evaluate_method_on_seed(
    method: str,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    *,
    eps_values: Sequence[float],
    min_samples_values: Sequence[int],
    contamination_values: Sequence[float],
    lof_n_neighbors_values: Sequence[int],
    allow_skip_degenerate_grid_cells: bool = False,
    manifest_ref: dict[str, Any] | None = None,
) -> tuple[SeedResult, list[dict[str, Any]]]:
    val_clouds = load_clouds(config, "val", device)
    test_clouds = load_clouds(config, "test", device)

    if method == "euclidean_dbscan":
        combos = dbscan_param_combos(eps_values, min_samples_values)
        best_params, val_mcc, grid_log = grid_search_clouds(
            val_clouds,
            evaluate_euclidean_dbscan,
            combos,
            desc=f"seed{seed} euclidean_dbscan val",
            allow_skip_degenerate_grid_cells=allow_skip_degenerate_grid_cells,
            manifest_ref=manifest_ref,
        )
        test_fn: Callable[..., CloudMetrics] = evaluate_euclidean_dbscan
    elif method == "isolation_forest":
        combos = [
            {"contamination": float(c), "random_state": seed}
            for c in contamination_values
        ]
        best_params, val_mcc, grid_log = grid_search_clouds(
            val_clouds,
            evaluate_isolation_forest,
            combos,
            desc=f"seed{seed} isolation_forest val",
            allow_skip_degenerate_grid_cells=allow_skip_degenerate_grid_cells,
            manifest_ref=manifest_ref,
        )
        test_fn = evaluate_isolation_forest
    elif method == "lof":
        combos = lof_param_combos(lof_n_neighbors_values, contamination_values)
        best_params, val_mcc, grid_log = grid_search_clouds(
            val_clouds,
            evaluate_lof,
            combos,
            desc=f"seed{seed} lof val",
            allow_skip_degenerate_grid_cells=allow_skip_degenerate_grid_cells,
            manifest_ref=manifest_ref,
        )
        test_fn = evaluate_lof
    elif method == "adbscan":
        combos = dbscan_param_combos(eps_values, min_samples_values)
        best_params, val_mcc, grid_log = grid_search_clouds(
            val_clouds,
            evaluate_adbscan,
            combos,
            desc=f"seed{seed} adbscan val",
            allow_skip_degenerate_grid_cells=allow_skip_degenerate_grid_cells,
            manifest_ref=manifest_ref,
        )
        test_fn = evaluate_adbscan
        # Real ellipse-filtration W-Dist only on test (grid phase selects by MCC).
        test_extra_kwargs = {"topo_options": topo_wdist_options_from_config(config)}
    else:
        raise ValueError(f"Unknown method: {method!r}")

    if method != "adbscan":
        test_extra_kwargs = {}

    per_test: list[CloudMetrics] = []
    for cloud in test_clouds:
        per_test.append(test_fn(cloud, **best_params, **test_extra_kwargs))
    recall, specificity, gmean, mcc, wdist = _aggregate_cloud_metrics(per_test)

    return SeedResult(
        seed=seed,
        method=method,
        hparams=best_params,
        val_mcc=val_mcc,
        test_recall=recall,
        test_specificity=specificity,
        test_gmean=gmean,
        test_mcc=mcc,
        test_wdist=wdist,
        n_test_clouds=len(per_test),
    ), grid_log


def config_for_seed(base_config: str, seed: int) -> dict[str, Any]:
    cfg = load_config(base_config, project_root=REPO_ROOT)
    return deep_update(cfg, {"data": {"seed": int(seed)}})


def sample_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


def write_summary_csv(
    out_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "method",
        "mcc_mean",
        "mcc_std",
        "gmean_mean",
        "gmean_std",
        "wdist_mean",
        "wdist_std",
        "notes",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def aggregate_method_results(seed_results: Sequence[SeedResult]) -> dict[str, Any]:
    mccs = [r.test_mcc for r in seed_results]
    gmeans = [r.test_gmean for r in seed_results]
    wdist = [r.test_wdist for r in seed_results]
    method = seed_results[0].method
    return {
        "method": method,
        "mcc_mean": float(np.mean(mccs)),
        "mcc_std": sample_std(mccs),
        "gmean_mean": float(np.mean(gmeans)),
        "gmean_std": sample_std(gmeans),
        "wdist_mean": float(np.mean(wdist)),
        "wdist_std": sample_std(wdist),
        "notes": (
            f"5 seeds; val hparam selection by max mean cloud MCC; "
            f"test n_clouds={seed_results[0].n_test_clouds} per seed; "
            "wdist_* columns are diagnostics only (not main-table); "
            "ADBSCAN uses fixed local-PCA ellipses (no 30ep training)"
        ),
    }


METHOD_ORDER = [
    "euclidean_dbscan",
    "isolation_forest",
    "lof",
    "adbscan",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # No implicit default: the n100 paper comparison must pass the paper config
    # (e.g. paper_mnist_h1); baselines share its
    # data settings and evaluation.dbscan / evaluation.baselines grids.
    p.add_argument("--base-config", type=str, required=True)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "paper_baselines")
    p.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    p.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_ORDER,
        default=METHOD_ORDER,
    )
    p.add_argument(
        "--eps-values",
        type=float,
        nargs="+",
        default=None,
        help="Override DBSCAN eps grid (default: evaluation.dbscan.eps_values from config).",
    )
    p.add_argument(
        "--min-samples-values",
        type=int,
        nargs="+",
        default=None,
        help="Override DBSCAN min_samples grid (default: config).",
    )
    p.add_argument(
        "--contamination-values",
        type=float,
        nargs="+",
        default=None,
        help="Override IF/LOF contamination grid (default: evaluation.baselines).",
    )
    p.add_argument(
        "--lof-n-neighbors",
        type=int,
        nargs="+",
        default=None,
        help="Override LOF n_neighbors grid (default: config).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.base_config, project_root=REPO_ROOT)
    preflight_baseline_eval(cfg, project_root=REPO_ROOT, out_dir=out_dir)
    grids = baseline_grids_from_config(cfg)
    rep = reproducibility_settings(cfg)
    manifest_ref: dict[str, Any] = {
        "fallback_status": "none",
        "fallbacks": [],
    }

    eps_values = args.eps_values if args.eps_values is not None else grids["eps_values"]
    min_samples_values = (
        args.min_samples_values
        if args.min_samples_values is not None
        else grids["min_samples_values"]
    )
    contamination_values = (
        args.contamination_values
        if args.contamination_values is not None
        else grids["contamination_values"]
    )
    lof_n_neighbors = (
        args.lof_n_neighbors if args.lof_n_neighbors is not None else grids["lof_n_neighbors"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = {
        "run_status": RUN_STATUS_COMPLETED,
        "source_revision": git_revision(REPO_ROOT),
        "base_config": args.base_config,
        "seeds": list(args.seeds),
        "methods": list(args.methods),
        "selection": "max mean cloud MCC on validation split",
        "metrics": "compute_recall_specificity_gmean_mcc_wdist",
        "local_pca_k": LOCAL_PCA_K,
        "grids": {
            "eps_values": list(eps_values),
            "min_samples_values": list(min_samples_values),
            "contamination_values": list(contamination_values),
            "lof_n_neighbors": list(lof_n_neighbors),
        },
        "reproducibility": rep,
        "fallback_status": manifest_ref["fallback_status"],
    }
    write_json(out_dir / "MANIFEST_baselines.json", manifest)

    all_seed_results: dict[str, list[SeedResult]] = {m: [] for m in args.methods}

    for seed in args.seeds:
        config = config_for_seed(args.base_config, seed)
        seed_dir = out_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for method in args.methods:
            print(f"\n=== seed={seed} method={method} ===")
            result, grid_log = evaluate_method_on_seed(
                method,
                config,
                seed,
                device,
                eps_values=eps_values,
                min_samples_values=min_samples_values,
                contamination_values=contamination_values,
                lof_n_neighbors_values=lof_n_neighbors,
                allow_skip_degenerate_grid_cells=rep["allow_skip_degenerate_grid_cells"],
                manifest_ref=manifest_ref,
            )
            all_seed_results[method].append(result)

            payload = {
                **asdict(result),
                "grid_log": grid_log,
            }
            out_json = seed_dir / f"{method}.json"
            out_json.write_text(json.dumps(payload, indent=2) + "\n")
            print(
                f"  val_mcc={result.val_mcc:.4f}  test_mcc={result.test_mcc:.4f}  "
                f"test_gmean={result.test_gmean:.4f}  test_wdist={result.test_wdist:.4f}  "
                f"hparams={result.hparams}"
            )

    summary_rows = [
        aggregate_method_results(all_seed_results[m])
        for m in args.methods
        if all_seed_results[m]
    ]
    summary_path = out_dir / "summary_baselines.csv"
    write_summary_csv(summary_path, summary_rows)

    manifest["summary_csv"] = str(summary_path)
    manifest["per_seed_json"] = str(out_dir / "seed{seed}/{method}.json")
    manifest["fallback_status"] = manifest_ref.get("fallback_status", "none")
    manifest["fallbacks"] = manifest_ref.get("fallbacks", [])
    write_json(out_dir / "MANIFEST_baselines.json", manifest)

    print(f"\nWrote {summary_path}")
    print(f"Wrote {out_dir / 'MANIFEST_baselines.json'}")
    for row in summary_rows:
        print(
            f"  {row['method']}: MCC={row['mcc_mean']:.4f}±{row['mcc_std']:.4f}  "
            f"G-Mean={row['gmean_mean']:.4f}±{row['gmean_std']:.4f}  "
            f"W-Dist={row['wdist_mean']:.4f}±{row['wdist_std']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
