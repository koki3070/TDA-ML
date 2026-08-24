#!/usr/bin/env python3
"""Fair one-class peers on thin rings with near-ring (radial) outliers.

Same information as proposed (``w_class=0``): fit on train-clean coordinates
only; score mixed clouds by coordinates; never train a labelled classifier.
Isolation Forest / Euclidean DBSCAN on mixed-only clouds are **out of scope**
here (already reported in paper baselines); pass ``--methods`` only if needed.

Threshold rule (unified; declared in the written report):
  val_mcc (default, rule C)
      freeze a global score cutoff that maximises mean per-cloud MCC on
      **val mixed**. Matches proposed Maha-DBSCAN ``(eps, min_samples)``
      selection. Test labels are used only for the final table.
  known_ratio (rule B)
      per cloud, flag the most outlier-like fraction ``k`` where
      ``k = num_outliers / (max_points + num_outliers)``. Uses the ratio
      only, never point labels.
  inlier_recall (rule A)
      cutoff so clean-val inlier recall >= ``--recall-tau`` (default 0.95).

Usage::

    # smoke (seconds)
    uv run python experiments/eval_fair_oneclass_rings.py --smoke

    # paper rings contract, 5 seeds
    uv run python experiments/eval_fair_oneclass_rings.py \\
        --base-config paper_rings \\
        --out-dir outputs/fair_oneclass_rings
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tda_ml.config import deep_update, load_config  # noqa: E402
from tda_ml.dbscan_eval import (  # noqa: E402
    dbscan_labels_to_outlier_pred,
    valid_clean_inliers,
)
from tda_ml.oneclass_baselines import (  # noqa: E402
    THRESHOLD_RULES,
    ThresholdRule,
    contamination_ratio,
    evaluate_predictions,
    fit_autoencoder,
    fit_deep_svdd,
    fit_elliptic,
    fit_gmm,
    fit_lof_novelty,
    fit_ocsvm_rbf,
    fit_svdd_primal,
    pool_points,
    predict_clouds,
    resolve_threshold,
    score_clouds,
)
from tda_ml.reproducibility import (  # noqa: E402
    RUN_STATUS_COMPLETED,
    baseline_grids_from_config,
    write_json,
)
from tda_ml.ring_dataset import (  # noqa: E402
    TEST_INDEX_OFFSET,
    ThinRingsDataset,
    ring_kwargs_from_config,
)
from tda_ml.seed_utils import set_global_seed  # noqa: E402
from tda_ml.supervised_diagnostics import git_revision  # noqa: E402
from tda_ml.visualization import INLIER_COLOR, OUTLIER_COLOR  # noqa: E402

PAPER_SEEDS = [42, 123, 456, 789, 1024]
FAIR_METHODS = (
    "ocsvm_rbf",
    "svdd_primal",
    "elliptic",
    "gmm",
    "autoencoder",
    "deep_svdd",
    "lof_novelty",
)
LOWINFO_METHODS = ("isolation_forest_mixed", "euclidean_dbscan_mixed")


def sample_std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


def collect_split(
    ds: ThinRingsDataset, n: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    noisy, labels, cleans = [], [], []
    for i in range(n):
        pts, lab, clean = ds[i]
        noisy.append(pts.numpy().astype(np.float64))
        labels.append(lab.numpy().astype(int))
        cleans.append(valid_clean_inliers(clean.numpy().astype(np.float64)))
    return noisy, labels, cleans


def make_rings_ds(
    cfg: dict[str, Any], n: int, *, index_offset: int, seed: int
) -> ThinRingsDataset:
    dcfg = cfg["data"]
    return ThinRingsDataset(
        n,
        max_points=int(dcfg["max_points"]),
        num_outliers=int(dcfg["num_outliers"]),
        noise_std=float(dcfg["noise_std"]),
        noise_seed=int(seed),
        index_offset=int(index_offset),
        **ring_kwargs_from_config(dcfg),
    )


def fit_fair_models(
    name: str,
    clean_train: np.ndarray,
    clean_val: np.ndarray,
    *,
    seed: int,
    rng: np.random.Generator,
    device: torch.device,
    deep_epochs: int,
) -> tuple[Any, dict[str, Any]]:
    extra: dict[str, Any] = {}
    if name == "ocsvm_rbf":
        return fit_ocsvm_rbf(clean_train), extra
    if name == "svdd_primal":
        return fit_svdd_primal(clean_train), extra
    if name == "elliptic":
        return fit_elliptic(clean_train, seed=seed), extra
    if name == "gmm":
        scorer, n_comp = fit_gmm(clean_train, clean_val, seed=seed)
        extra["n_components"] = n_comp
        return scorer, extra
    if name == "autoencoder":
        return (
            fit_autoencoder(
                clean_train,
                seed=seed,
                epochs=deep_epochs,
                rng=rng,
                device=device,
            ),
            extra,
        )
    if name == "deep_svdd":
        pre = max(1, deep_epochs // 3)
        svdd = max(1, deep_epochs - pre)
        return (
            fit_deep_svdd(
                clean_train,
                seed=seed,
                pretrain_epochs=pre,
                svdd_epochs=svdd,
                rng=rng,
                device=device,
            ),
            extra,
        )
    if name == "lof_novelty":
        return fit_lof_novelty(clean_train), extra
    raise ValueError(name)


def evaluate_if_mixed(
    mixed: list[np.ndarray],
    labels: list[np.ndarray],
    cleans: list[np.ndarray],
    *,
    contamination: float,
    seed: int,
    compute_wdist: bool,
) -> dict[str, Any]:
    preds = []
    for pts in mixed:
        iso = IsolationForest(
            contamination=float(contamination),
            random_state=int(seed),
            n_estimators=100,
        )
        sk = iso.fit_predict(pts)
        preds.append((sk == -1).astype(np.int64))
    return evaluate_predictions(preds, labels, mixed, cleans, compute_wdist=compute_wdist)


def evaluate_euclid_dbscan(
    mixed: list[np.ndarray],
    labels: list[np.ndarray],
    cleans: list[np.ndarray],
    *,
    eps: float,
    min_samples: int,
    compute_wdist: bool,
) -> dict[str, Any]:
    preds = []
    for pts in mixed:
        db = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(pts)
        preds.append(dbscan_labels_to_outlier_pred(db))
    return evaluate_predictions(preds, labels, mixed, cleans, compute_wdist=compute_wdist)


def select_if_contamination(
    val_mixed: list[np.ndarray],
    val_y: list[np.ndarray],
    grid: list[float],
    seed: int,
) -> tuple[float, float]:
    best_c = float(grid[0])
    best_mcc = -2.0
    for c in grid:
        metrics = evaluate_if_mixed(
            val_mixed, val_y, val_mixed, contamination=c, seed=seed, compute_wdist=False
        )
        if metrics["mcc"] > best_mcc:
            best_mcc = float(metrics["mcc"])
            best_c = float(c)
    return best_c, best_mcc


def select_dbscan_hparams(
    val_mixed: list[np.ndarray],
    val_y: list[np.ndarray],
    eps_values: list[float],
    min_samples_values: list[int],
) -> tuple[dict[str, Any], float]:
    best: dict[str, Any] | None = None
    best_mcc = -2.0
    for eps in eps_values:
        for ms in min_samples_values:
            metrics = evaluate_euclid_dbscan(
                val_mixed,
                val_y,
                val_mixed,
                eps=eps,
                min_samples=ms,
                compute_wdist=False,
            )
            if metrics["mcc"] > best_mcc:
                best_mcc = float(metrics["mcc"])
                best = {"eps": float(eps), "min_samples": int(ms)}
    if best is None:
        raise RuntimeError("Euclid DBSCAN val grid produced no cell")
    return best, best_mcc


def plot_qualitative(
    out_path: Path,
    clouds: list[np.ndarray],
    preds: list[np.ndarray],
    labels: list[np.ndarray],
    title: str,
    n_show: int = 4,
) -> None:
    n = min(n_show, len(clouds))
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4), squeeze=False)
    for j in range(n):
        pts, pred, y = clouds[j], preds[j], labels[j]
        ax_gt, ax_pr = axes[0, j], axes[1, j]
        ax_gt.scatter(
            pts[y == 0, 0], pts[y == 0, 1], c=INLIER_COLOR, s=12, label="inlier"
        )
        ax_gt.scatter(
            pts[y == 1, 0], pts[y == 1, 1], c=OUTLIER_COLOR, s=12, label="outlier"
        )
        ax_gt.set_title(f"GT {j}")
        ax_pr.scatter(
            pts[pred == 0, 0],
            pts[pred == 0, 1],
            c=INLIER_COLOR,
            s=12,
            label="pred in",
        )
        ax_pr.scatter(
            pts[pred == 1, 0],
            pts[pred == 1, 1],
            c=OUTLIER_COLOR,
            s=12,
            label="pred noise",
        )
        ax_pr.set_title(f"pred {j}")
        for ax in (ax_gt, ax_pr):
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(loc="upper right", fontsize=7)
    axes[1, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fmt_pm(mean: float, std: float) -> str:
    if mean is None or not np.isfinite(mean) or std is None or not np.isfinite(std):
        raise ValueError(
            f"refusing TBD/non-finite summary for paper report: mean={mean!r} std={std!r}"
        )
    return f"{mean:.3f} ± {std:.3f}"


def fmt_count(value: Any) -> str:
    if value is None or value == "TBD":
        raise ValueError(f"refusing TBD/missing count for paper report: {value!r}")
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"refusing non-numeric count for paper report: {value!r}") from exc


def load_proposed_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("mcc_mean", "mcc_std", "gmean_mean", "gmean_std")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(
            f"--proposed-metrics missing {missing}; refusing TBD/placeholder report"
        )
    for key in required:
        if not np.isfinite(float(payload[key])):
            raise ValueError(
                f"--proposed-metrics {key}={payload[key]!r} is non-finite"
            )
    return payload


def write_report(
    path: Path,
    *,
    protocol: dict[str, Any],
    summary: list[dict[str, Any]],
    proposed: dict[str, Any] | None = None,
) -> None:
    rule = protocol["threshold_rule"]
    rule_text = {
        "val_mcc": (
            "**採用ルール C（val_mcc）**: val mixed 上で平均 per-cloud MCC を"
            "最大にするスコア閾値を選び、test に固定する。提案の Maha-DBSCAN が"
            " val MCC で `(eps, min_samples)` を選ぶのと同じ検出ヘッド契約。"
            "test ラベルは最終指標のみ。"
        ),
        "known_ratio": (
            "**採用ルール B（known_ratio）**: 各 mixed 雲でスコア上位"
            f" k={protocol['k']:.4f} をノイズとする。比の値だけ使い、点ラベルは使わない。"
        ),
        "inlier_recall": (
            "**採用ルール A（inlier_recall）**: clean val 上でインライア再現率"
            f" ≥ τ={protocol['recall_tau']} となるスコア閾値を固定。"
        ),
    }[rule]
    lines = [
        "# Fair one-class baselines（thin rings / ring_radial）",
        "",
        "## 閾値ルール（最初に明記）",
        "",
        rule_text,
        "",
        "## 1. 情報量の定義",
        "",
        "使える:",
        "",
        "- train の clean 参照点群 `X_clean`（インライア座標のみ）",
        "- 評価時の mixed 点群 `X_mixed`（座標のみ）",
        "- 既知のノイズ比 k（ルール B のとき。点ラベルではない）",
        "",
        "使えない:",
        "",
        "- 学習時の per-point in/out ラベル（教師あり二値分類は禁止）",
        "- test ラベルによる閾値最適化（リーク）",
        "- 提案内部（ellphi, PH 損失, 学習楕円）",
        "",
        "データ: `ThinRingsDataset` / `outlier_mode=ring_radial`"
        "（円周の法線方向オフセット 0.03–0.06）。"
        f" inliers={protocol['max_points']}, outliers={protocol['num_outliers']},"
        f" k={protocol['k']:.4f}。",
        "",
        "## 2. 同条件か",
        "",
        "| 手法 | 学習 | テスト | 同条件 | 備考 |",
        "|------|------|--------|:------:|------|",
        "| ocsvm_rbf | train clean | mixed 座標 | yes |"
        " RBF OneClassSVM。RBF 核では kernel SVDD と同等 |",
        "| svdd_primal | train clean | mixed 座標 | yes |"
        " 入力空間の球（中心=clean 平均）。核 SVDD とは別物 |",
        "| elliptic | train clean | mixed 座標 | yes | EllipticEnvelope |",
        "| gmm | train clean; 成分数は clean val 尤度 | mixed 座標 | yes | |",
        "| autoencoder | train clean | 再構成誤差 | yes | 2D MLP、ラベルなし |",
        "| deep_svdd | train clean | 超球距離 | yes | Ruff 簡易実装 |",
        "| lof_novelty | train clean (`novelty=True`) | mixed 座標 | yes | 任意枠 |",
        "| Isolation Forest / Euclid DBSCAN (mixed-only) | — | — | **対象外** |"
        " 既存 paper baselines / Methods 表を参照。本スクリプトでは回さない |",
        "| Proposed | train clean PD 教師 | mixed 座標 + 学習楕円 DBSCAN |"
        " 比較対象 | 本スクリプトでは再学習しない |",
        "",
        "ambient one-class は全 train 雲の clean をプールするため、雲ごとの環位置を"
        "持たない（提案は train 時に雲単位の clean PD を見る）。これは同条件枠の"
        "限界として明記する。",
        "",
        "## 3. 結果表",
        "",
        "指標は test の雲平均。複数シードは mean ± sample std (ddof=1)。",
        "W-Dist は除去後点群と clean 参照の Euclidean Alpha H1"
        "（提案の ellphi 濾過 W-Dist ではない）。",
        "",
        "| 手法 | 枠 | MCC ↑ | G-Mean ↑ | W-Dist ↓ | TP | TN | FP | FN |",
        "|------|----|------:|---------:|---------:|---:|---:|---:|---:|",
    ]
    for row in summary:
        wdist = (
            "—"
            if row.get("wdist_mean") is None
            else fmt_pm(row["wdist_mean"], row["wdist_std"])
        )
        lines.append(
            f"| {row['method']} | {row['tier']} | "
            f"{fmt_pm(row['mcc_mean'], row['mcc_std'])} | "
            f"{fmt_pm(row['gmean_mean'], row['gmean_std'])} | "
            f"{wdist} | "
            f"{fmt_count(row.get('tp'))} | {fmt_count(row.get('tn'))} | "
            f"{fmt_count(row.get('fp'))} | {fmt_count(row.get('fn'))} |"
        )
    lines.extend(
        [
            "",
            "## 4. 提案手法",
            "",
        ]
    )
    if proposed is None:
        lines.extend(
            [
                "本スクリプトは提案を学習しない。提案列を出すときは "
                "`--proposed-metrics` に `eval_paper` / 集計 JSON "
                "（`mcc_mean`, `mcc_std`, `gmean_mean`, `gmean_std` 必須）を渡す。"
                "欠落時はプレースホルダを書かず、この節を数値なしで終える。",
                "",
            ]
        )
    else:
        wdist = (
            "—"
            if proposed.get("wdist_mean") is None
            else fmt_pm(float(proposed["wdist_mean"]), float(proposed["wdist_std"]))
        )
        lines.extend(
            [
                "同じ `thin_rings` / `ring_radial` 契約。数値は `--proposed-metrics` "
                "（生成物。gitignored `docs/` からのハードコードはしない）。",
                "",
                "| 手法 | MCC ↑ | G-Mean ↑ | W-Dist ↓ |",
                "|------|------:|---------:|---------:|",
                f"| Proposed | {fmt_pm(float(proposed['mcc_mean']), float(proposed['mcc_std']))} "
                f"| {fmt_pm(float(proposed['gmean_mean']), float(proposed['gmean_std']))} "
                f"| {wdist} |",
                "",
            ]
        )
    lines.extend(
        [
            f"seeds={protocol['seeds']}, n_train={protocol['n_train']},"
            f" n_val={protocol['n_val']}, n_test={protocol['n_test']}.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-config",
        default="paper_rings",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "fair_oneclass_rings",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    p.add_argument(
        "--threshold-rule",
        choices=THRESHOLD_RULES,
        default="val_mcc",
        help="Unified detection-head rule (default: C / val_mcc).",
    )
    p.add_argument("--recall-tau", type=float, default=0.95)
    p.add_argument("--max-train-points", type=int, default=30000)
    p.add_argument("--deep-epochs", type=int, default=30)
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(FAIR_METHODS),
        help="Default: fair same-info peers only (no mixed-only IF/DBSCAN).",
    )
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument(
        "--num-outliers",
        type=int,
        default=None,
        help="Override data.num_outliers (e.g. 100 for 50/50 with max_points=100).",
    )
    p.add_argument(
        "--proposed-metrics",
        type=Path,
        default=None,
        help=(
            "Optional JSON from paper eval/aggregate for the Proposed row. "
            "Must contain finite mcc_mean/mcc_std/gmean_mean/gmean_std. "
            "Omit to skip the Proposed numbers (no TBD placeholders)."
        ),
    )
    p.add_argument("--skip-wdist", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny split / few epochs / one seed (wiring check).",
    )
    p.add_argument("--n-qualitative", type=int, default=4)
    return p.parse_args()


def apply_smoke(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.seeds = args.seeds[:1]
    args.n_train = args.n_train or 24
    args.n_val = args.n_val or 8
    args.n_test = args.n_test or 8
    args.deep_epochs = min(args.deep_epochs, 4)
    args.max_train_points = min(args.max_train_points, 4000)
    args.skip_wdist = True


def run_one_seed(
    *,
    cfg: dict[str, Any],
    seed: int,
    methods: list[str],
    rule: ThresholdRule,
    tau: float,
    max_train_points: int,
    deep_epochs: int,
    compute_wdist: bool,
    n_qualitative: int,
    out_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    set_global_seed(seed)
    dcfg = cfg["data"]
    n_train = int(dcfg["train_size"])
    n_val = int(dcfg["val_size"])
    n_test = int(dcfg["test_size"])
    k = contamination_ratio(int(dcfg["max_points"]), int(dcfg["num_outliers"]))

    train_ds = make_rings_ds(cfg, n_train, index_offset=0, seed=seed)
    val_ds = make_rings_ds(cfg, n_val, index_offset=n_train, seed=seed)
    test_ds = make_rings_ds(cfg, n_test, index_offset=TEST_INDEX_OFFSET, seed=seed)

    print(f"collect splits train={n_train} val={n_val} test={n_test}", flush=True)
    tr_n, tr_y, tr_c = collect_split(train_ds, n_train)
    va_n, va_y, va_c = collect_split(val_ds, n_val)
    te_n, te_y, te_c = collect_split(test_ds, n_test)
    del tr_n, tr_y  # mixed train and its labels are unused (no labelled fit)

    rng = np.random.default_rng(seed)
    clean_train = pool_points(tr_c, max_train_points, rng)
    clean_val = pool_points(va_c, min(max_train_points, 20000), rng)
    print(f"pooled train clean {clean_train.shape}", flush=True)

    seed_dir = out_dir / f"seed{seed}"
    fig_dir = seed_dir / "figures"
    seed_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    fair = [m for m in methods if m in FAIR_METHODS]
    for name in fair:
        print(f"fit {name}", flush=True)
        scorer, extra = fit_fair_models(
            name,
            clean_train,
            clean_val,
            seed=seed,
            rng=rng,
            device=device,
            deep_epochs=deep_epochs,
        )
        va_scores = score_clouds(scorer, va_n)
        te_scores = score_clouds(scorer, te_n)
        va_clean_scores = score_clouds(scorer, va_c)
        chosen = resolve_threshold(
            rule,
            val_mixed_scores=va_scores,
            val_mixed_labels=va_y,
            val_clean_scores=va_clean_scores,
            k=k,
            tau=tau,
        )
        te_preds = predict_clouds(
            te_scores,
            rule=rule,
            threshold=chosen["threshold"],
            k=chosen["k"],
        )
        test = evaluate_predictions(
            te_preds, te_y, te_n, te_c, compute_wdist=compute_wdist
        )
        row = {
            "seed": seed,
            "method": name,
            "tier": "fair_clean_train_only",
            "hparams": {**extra, **chosen},
            "test": test,
        }
        rows.append(row)
        print(
            f"  {name}: test MCC={test['mcc']:.4f} G-Mean={test['gmean']:.4f}",
            flush=True,
        )
        plot_qualitative(
            fig_dir / f"{name}.png",
            te_n,
            te_preds,
            te_y,
            title=f"{name} seed={seed}",
            n_show=n_qualitative,
        )

    if "isolation_forest_mixed" in methods:
        print("isolation_forest_mixed (lower-information tier)", flush=True)
        grids = baseline_grids_from_config(cfg)
        if rule == "known_ratio":
            cont, val_mcc = k, None
        else:
            cont, val_mcc = select_if_contamination(
                va_n, va_y, list(grids["contamination_values"]), seed
            )
        test = evaluate_if_mixed(
            te_n, te_y, te_c, contamination=cont, seed=seed, compute_wdist=compute_wdist
        )
        rows.append(
            {
                "seed": seed,
                "method": "isolation_forest_mixed",
                "tier": "lower_info_mixed_only",
                "hparams": {"contamination": cont, "val_mcc": val_mcc},
                "test": test,
            }
        )
        print(f"  IF mixed: test MCC={test['mcc']:.4f}", flush=True)

    if "euclidean_dbscan_mixed" in methods:
        print("euclidean_dbscan_mixed (lower-information; val MCC grid)", flush=True)
        grids = baseline_grids_from_config(cfg)
        hp, val_mcc = select_dbscan_hparams(
            va_n,
            va_y,
            [float(x) for x in grids["eps_values"]],
            [int(x) for x in grids["min_samples_values"]],
        )
        test = evaluate_euclid_dbscan(
            te_n, te_y, te_c, compute_wdist=compute_wdist, **hp
        )
        rows.append(
            {
                "seed": seed,
                "method": "euclidean_dbscan_mixed",
                "tier": "lower_info_mixed_only",
                "hparams": {**hp, "val_mcc": val_mcc, "note": "no score; val MCC grid"},
                "test": test,
            }
        )
        print(f"  Euclid DBSCAN: test MCC={test['mcc']:.4f} {hp}", flush=True)

    write_json(seed_dir / "metrics.json", {"seed": seed, "rows": rows})
    return rows


def aggregate(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = []
    for r in all_rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    out = []
    for method in methods:
        subset = [r for r in all_rows if r["method"] == method]
        mccs = [float(r["test"]["mcc"]) for r in subset]
        gs = [float(r["test"]["gmean"]) for r in subset]
        wd_raw = [r["test"]["wdist"] for r in subset]
        wd_ok = [float(w) for w in wd_raw if w is not None]
        row = {
            "method": method,
            "tier": subset[0]["tier"],
            "n_seeds": len(subset),
            "mcc_mean": float(np.mean(mccs)),
            "mcc_std": sample_std(mccs),
            "gmean_mean": float(np.mean(gs)),
            "gmean_std": sample_std(gs),
            "wdist_mean": float(np.mean(wd_ok)) if wd_ok else None,
            "wdist_std": sample_std(wd_ok) if wd_ok else None,
            "tp": int(round(float(np.mean([r["test"]["tp"] for r in subset])))),
            "tn": int(round(float(np.mean([r["test"]["tn"] for r in subset])))),
            "fp": int(round(float(np.mean([r["test"]["fp"] for r in subset])))),
            "fn": int(round(float(np.mean([r["test"]["fn"] for r in subset])))),
            "tp_tn_fp_fn_note": "confusion counts = mean over seeds, rounded",
        }
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "method",
        "tier",
        "n_seeds",
        "mcc_mean",
        "mcc_std",
        "gmean_mean",
        "gmean_std",
        "wdist_mean",
        "wdist_std",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    args = parse_args()
    apply_smoke(args)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.base_config, project_root=REPO_ROOT)
    if str(cfg["data"].get("dataset_type", "")).lower() != "thin_rings":
        raise ValueError(
            "this driver is rings-only; got "
            f"dataset_type={cfg['data'].get('dataset_type')!r}"
        )
    if args.num_outliers is not None:
        cfg = deep_update(cfg, {"data": {"num_outliers": int(args.num_outliers)}})
    if args.n_train is not None:
        cfg = deep_update(cfg, {"data": {"train_size": int(args.n_train)}})
    if args.n_val is not None:
        cfg = deep_update(cfg, {"data": {"val_size": int(args.n_val)}})
    if args.n_test is not None:
        cfg = deep_update(cfg, {"data": {"test_size": int(args.n_test)}})

    dcfg = cfg["data"]
    k = contamination_ratio(int(dcfg["max_points"]), int(dcfg["num_outliers"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rule: ThresholdRule = args.threshold_rule  # type: ignore[assignment]

    protocol = {
        "threshold_rule": rule,
        "recall_tau": float(args.recall_tau),
        "k": k,
        "max_points": int(dcfg["max_points"]),
        "num_outliers": int(dcfg["num_outliers"]),
        "n_train": int(dcfg["train_size"]),
        "n_val": int(dcfg["val_size"]),
        "n_test": int(dcfg["test_size"]),
        "seeds": list(args.seeds),
        "base_config": args.base_config,
        "dataset_type": "thin_rings",
        "outlier_mode": str(dcfg["outlier_mode"]),
        "max_train_points": int(args.max_train_points),
        "deep_epochs": int(args.deep_epochs),
        "skip_wdist": bool(args.skip_wdist),
        "smoke": bool(args.smoke),
        "usable": ["X_clean train", "X_mixed coordinates", "scalar k if rule B"],
        "forbidden": [
            "point in/out labels at fit",
            "test labels for threshold",
            "ellphi / PH loss / learned ellipses",
        ],
        "ocsvm_vs_svdd": (
            "sklearn OneClassSVM(kernel='rbf') is kernel SVDD for a Gaussian "
            "kernel; svdd_primal is a distinct input-space sphere."
        ),
    }
    write_json(
        out_dir / "MANIFEST.json",
        {
            "run_status": RUN_STATUS_COMPLETED,
            "source_revision": git_revision(REPO_ROOT),
            "protocol": protocol,
        },
    )

    all_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        cfg_seed = deep_update(cfg, {"data": {"seed": int(seed)}})
        print(f"\n=== seed={seed} ===", flush=True)
        all_rows.extend(
            run_one_seed(
                cfg=cfg_seed,
                seed=int(seed),
                methods=list(args.methods),
                rule=rule,
                tau=float(args.recall_tau),
                max_train_points=int(args.max_train_points),
                deep_epochs=int(args.deep_epochs),
                compute_wdist=not args.skip_wdist,
                n_qualitative=int(args.n_qualitative),
                out_dir=out_dir,
                device=device,
            )
        )

    summary = aggregate(all_rows)
    write_csv(out_dir / "summary.csv", summary)
    write_json(out_dir / "summary.json", {"protocol": protocol, "rows": summary})
    proposed = (
        load_proposed_metrics(args.proposed_metrics)
        if args.proposed_metrics is not None
        else None
    )
    write_report(
        out_dir / "REPORT.md",
        protocol=protocol,
        summary=summary,
        proposed=proposed,
    )
    print(f"\nWrote {out_dir / 'summary.csv'}", flush=True)
    print(f"Wrote {out_dir / 'REPORT.md'}", flush=True)
    for row in summary:
        print(
            f"  {row['method']}: MCC={row['mcc_mean']:.4f}±{row['mcc_std']:.4f}  "
            f"G-Mean={row['gmean_mean']:.4f}±{row['gmean_std']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
