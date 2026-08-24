# 再現性ガイド

この文書は **読者向け** です。実験の再現手順、データと成果物の置き場所、図がコードのどこに対応するかをまとめています。内部手順・監査メモなど（従来 `docs/` に置いていたもの）は **この公開リポジトリの git ツリーには含めません**（ローカルや別管理で保持してください）。

## リポジトリに含まれる範囲（目安）

- **含む:** `tda_ml/`、**`configs/` 直下の正本 YAML**（`base.yaml` と `reproduce` / `dev` / `prod` / `test_fast`、論文比較用の `paper_n100_o20_nocls_h1_ellphi_lpca_power` / `tune_n100_o20_nocls_h1_ellphi_lpca_power`、および Methods opt-in の `methods_n100_o20_nocls_*`）、`tests/`、追跡されている `scripts/`、論文・再現用 `experiments/`（下記）、および `README.md` / `REPRODUCIBILITY.md` / `pyproject.toml` / `uv.lock` / `LICENSE` / `CITATION.cff` などのメタデータ。
- **含めない:** `docs/` 以下（**ローカル実験メモ**；公開方針で git に入れる場合は別途決定）、`configs/archive/`（履歴用 YAML を置く場合は **ローカルのみ**）、`outputs/`、`data/`、`.cursor/` など。`load_config("archive/...")` は、手元に `configs/archive/*.yaml` を置いた場合にのみ使えます。

### 論文比較（W-Dist / MCC 二目的）で使う `experiments/`

**論文主表の提案:** W-Dist tune 重みの 30ep 5-seed（`run_paper_30ep_multiseed.sh wdist`）。
**主張:** Euclidean DBSCAN / ADBSCAN と **同程度の外れ値除去性能**（MCC / G-Mean；5 seed の mean ± sample std による**記述的**比較。同等性検定は行わない）。主表に Topo W. 列は載せない。
**比較の非対称:** ADBSCAN は学習なしの局所 PCA 楕円ベースライン。提案法は同一データで 30ep 学習する（計算資源・パラメータ更新は対等ではない）。
**正本 config:** `paper_n100_o20_nocls_h1_ellphi_lpca_power`（`w_class=0`, `homology_dimensions=[1]`, `aniso_mode=elongate`）。
出力先は `WDIST_OUT` / `MCC_OUT` / `LOG_ROOT`（既定: `outputs/supervised/paper30_*`）で明示する（生成物は git に含めない）。

| 区分 | パス |
|------|------|
| 本番 5-seed | `run_paper_30ep_multiseed.sh`, `run_paper_30ep.py`, `aggregate_paper_multiseed.py` |
| paper eval | `eval_paper.py`（`best_model.pth` のみ） |
| ベースライン | `eval_baselines.py` |
| チューニング（重みの出所） | `tune_wdist.py`, `tune_mcc.py`, `tune_wdist_parallel.sh`, `tune_mcc_parallel.sh`, `tune_objectives.sh` |

**実行記録:** 各 run の `source_revision`（git HEAD）は `logs/run_manifest.json` および `paper_metrics_*.json` に記録。未コミットのまま実行した場合、リモート clone では数値が再現できない。

### 正本の起動（学習 → paper eval）

checkpoint は **`best_model.pth`（`selection=val_topo`）のみ**。欠落は hard-fail（サイレント代替なし）。YAML 役割は `configs/README.md`。

**1. Tune（主表は W-Dist；YAML の data.seed=42。Optuna sampler seed は worker ごとに異なる）**

```bash
MODE=wdist bash experiments/tune_objectives.sh
```

**2. 本番 30ep × 5-seed（tune JSON 必須）→ 内部で paper eval**

```bash
bash experiments/run_paper_30ep_multiseed.sh wdist
```

**3. ベースライン（ADBSCAN 等）**

```bash
uv run python experiments/eval_baselines.py \
  --base-config paper_n100_o20_nocls_h1_ellphi_lpca_power \
  --out-dir outputs/paper_baselines
```

単一 seed・手動 eval:

```bash
uv run python experiments/run_paper_30ep.py \
  --tune-json outputs/tune/wdist/best_wdist_ellphi.json \
  --seed 42

uv run python experiments/eval_paper.py \
  --run-dir outputs/supervised/.../paper_s42_<stamp> \
  --base-config paper_n100_o20_nocls_h1_ellphi_lpca_power \
  --split val
```

運用補助: `experiments/launch_detached_screen.sh`（screen 経由の長時間ジョブ）。論文主表の入口ではない。

### W-Dist 契約（場所ごとの定義）

| 経路 | homology | 距離 / filtration | 備考 |
|------|----------|-------------------|------|
| 訓練 `TopologicalLoss` / eval `compute_topo_wdist` | config の `homology_dimensions`（主表は `[1]`） | ellipse filtration + Wasserstein-2²（torch_topological） | 教師は `loss.teacher_mode`（主表 `local_pca`） |
| Gudhi `persistence.compute_w_distance` | H1-only | Euclidean Alpha / 点座標 | legacy baseline 用。主表の ellipse W-Dist とは別物 |
| `metrics` の W-Dist | 上記どちらかを明示引数で選択 | 引数不足は **hard-fail**（黙って 0 にしない） | |

主表・チューニングの preflight は `homology_dimensions=[1]`、`teacher_mode=local_pca`、`prob_weighting=false`、`aniso_mode=elongate`、`distance_backend=ellphi`、`size_mode=power`、`w_class=0.0`、`teacher_local_pca_k=10`、`teacher_local_pca_normalize_axes=true` の明示を要求する。欠落や不一致は実行前に hard-fail する。本番 30ep は `--tune-json`（H1-only Optuna best）必須で、YAML 埋め込みの旧重みでは起動しない。

**退化ガード variant（主表外・Methods opt-in）:** near-tangent データでは素の `elongate` が短軸→0 まで潰し、ellphi tangency が hard-fail し得る。対策として `aniso_mode: elongate_barrier`（`PAPER_NO_CLS_BARRIER_CONTRACT`、`aniso_barrier_threshold=6.0`、`distance_backend=ellphi`）を **YAML で明示したときだけ**使う。公開正本: `methods_n100_o20_nocls_h1_ellphi_lpca_power_neartangent_barrier`（`BASE_CONFIG=...`）。Euclidean-teacher 対比列は `methods_n100_o20_nocls_h1_maha_euclid_power`。暗黙の切替はしない。主表の ADBSCAN 比較・本番 30ep 経路には使わない。

## 命名移行（2026-07 / PR #7）— 破壊的

論文本番の run / tune 出力の識別子を短縮した。**旧ツリーは集計・freshness 対象外。再実行必須。** 暗黙に旧 path を拾わない（[Computational Reproducibility skill](https://github.com/t-uda/skills/blob/main/skills/computational-reproducibility/SKILL.md)）。

| 種別 | 旧 | 新 |
|------|----|----|
| 本番 run-dir | `pwr_s{seed}_*` | `paper_s{seed}_*` |
| 本番 out base | `outputs/supervised/pwr30_*` | `outputs/supervised/paper30_*` |
| tune out | `outputs/tune/pwr_wdist` / `pwr_mcc` | `outputs/tune/wdist` / `mcc` |
| best JSON | `best_elongate_wdist_*.json` | `best_wdist_*.json` |
| metrics tag | `power_{wdist,mcc}_valtopo_paper_eval` | `wdist` / `mcc` |
| Optuna study | `elongate_local_pca_power_*` | `tune_{wdist,mcc}_*` |

- `aggregate_paper_multiseed.py` / `paper_run_freshness.py` は同一 `out_base` に旧 `pwr_s*` がある場合、または `pwr_s*` と `paper_s*` が混在する場合 **hard-fail** する。
- 旧 `config_id` `teacher_local_pca_power_seed{N}` は引き続き slug `pwr_s{N}`（新 `paper_s{N}` とは別名前空間）。再実行は新名前空間で行う。

## 環境

- **Python**: `.python-version` を参照（現状 3.12）。
- **依存関係**: **uv** で管理。リポジトリのルートで:

  ```bash
  ./scripts/ensure_pytorch_topological.sh
  ./scripts/ensure_ellphi_repo.sh
  uv sync --all-groups   # ローカル検証用に dev（pytest, ruff）を含める
  ```

  `torch_topological` は **`pytorch-topological/` の path 依存**（`pyproject.toml`）であり、通常の `git clone` だけではディレクトリが揃いません。コミットは **`third_party/pytorch_topological.ref`** に固定し、**`scripts/ensure_pytorch_topological.sh`** がその ref に checkout します（CI と同じ）。

  `ellphi` も **`ellphi_repo/` の path 依存**です。fork の pin は **`third_party/ellphi.ref`**、取得は **`scripts/ensure_ellphi_repo.sh`**（CI と同じ）。`ellphi_repo/` 自体は git に含めません。PyPI の `ellphi==0.1.2` だけでは **`ellphi.grad`（学習用）が不足**するため、clone 後は ensure 必須です。

  各 run の `logs/run_manifest.json` には `reproducibility.ellphi_repo` として **pin 済み SHA** と **インストール済み checkout SHA** が記録されます。不一致時は preflight が hard-fail します。

- **PyTorch / CUDA**: 数値結果はデバイスや dtype によって変わり得ます。`tda_ml/main.py` の学習ループを使う場合、有効な設定は各実行の `logs/` 配下の `runtime_profile.json` などに記録されます。

## データ（MNIST）

- MNIST は git にコミットしません（`data/` は無視対象）。
- 学習・paper eval ともデータ根は **リポジトリ根の `data/`**（`tda_ml.config.default_data_root()`）であり、プロセスの cwd には依存しません。初回アクセス時に `torchvision` 経由でそこにダウンロードされます。
- 初回はインターネットに到達できるようにするか、キャッシュ済みの MNIST を自分でリポジトリ根の `data/` に置いてください。
- 設定 YAML の役割分担は **`configs/README.md`** を参照（共有プロファイル + 論文用 `paper_*` / `tune_*` + Methods `methods_*`）。探索用の旧設定はローカルで `configs/archive/` に置けるが、公開クローンには同梱されない。

## チェックポイントと実行出力

- 論文・backend 比較とも、学習成果物は実行ごとのディレクトリ以下に書き出されます。典型例は `logs/metrics.csv`、`logs/runtime_profile.json`、`logs/run_manifest.json`、`best_model.pth`、可視化が有効なら `images/` などです。
- paper eval / tune の評価 checkpoint は **`best_model.pth` のみ**（`resolve_val_topo_checkpoint`）。別名への切替は不可。
- **`outputs/`** は git の対象外です。論文用に実行ツリーを保存する場合は、原稿や付録で **コミットハッシュ・シード・使用した設定名** とあわせてパスを示すと追跡しやすいです。`progress_summary.csv` には絶対パスが入るため、共有時のプライバシーに注意してください。

## 厳格な再現性インフラ（preflight / manifest）

[Computational Reproducibility](https://github.com/t-uda/skills/blob/main/skills/computational-reproducibility/SKILL.md) に沿い、本リポジトリでは **暗黙 fallback を禁止**し、実行前検証と manifest 記録で監査可能にしています（実装: `tda_ml/preflight.py`, `tda_ml/reproducibility.py`）。

### 実行前 preflight

| 入口 | 出力 | 内容 |
|------|------|------|
| tune study | `STUDY_PREFLIGHT.json` | ベース config・DBSCAN grid・依存関係 |
| tune 本番 run | `RUN_PREFLIGHT.json` | tune JSON の objective 種別（MCC / W-Dist）・checkpoint 選択方針 |
| 学習 (`tda_ml.main`) | `logs/run_manifest.json` | preflight 通過後に学習開始；失敗時は `not-run` |
| paper eval | run-dir 内 checkpoint 存在確認 | `best_model.pth` 必須（サイレント代替なし） |

preflight 失敗時は **学習を開始せず** `run_status: not-run` を記録します。

### `run_status` 語彙

| 値 | 意味 |
|----|------|
| `pending` | manifest 作成直後（preflight 未実施） |
| `not-run` | preflight 失敗により学習未開始 |
| `running` | preflight 通過後、学習実行中 |
| `completed` | 正常終了 |
| `failed` | early-abort 等で異常終了 |
| `skipped` / `empty-result` / `zero-result` | 集計・評価スクリプト側の失敗語彙 |

`not-run` は preflight 専用です。学習中・中断 run を `not-run` と混同しないでください。

### `configs/base.yaml` の `reproducibility.*`（strict 既定）

| フラグ | 既定 | 効果 |
|--------|------|------|
| `strict_topo_samples` | `true` | 位相損失の NaN / 退化サンプルで hard-fail |
| `allow_nan_batch_skip` | `false` | NaN loss バッチの skip は opt-in のみ（manifest に記録） |
| `allow_empty_cloud_fallback` | `false` | 空点群の暗黙代替禁止 |
| `allow_skip_degenerate_grid_cells` | `false` | DBSCAN grid 退化セルの skip は opt-in のみ |
| `allow_otsu_threshold_fallback` | `false` | Otsu 閾値の暗黙 fallback 禁止 |
| `allow_legacy_loss_keys` | `false` | `training.lambda_*` からの重み読み取り禁止；`loss.w_*` を明示 |

opt-in fallback を有効にした場合、`run_manifest.json` の `fallbacks` 配列にイベントが追記されます。

### 明示必須の評価 grid

`evaluation.dbscan.eps_values` / `min_samples_values` / `backend` は **config に必須**です（Python 側の implicit default は削除済み）。ベースライン評価用の `evaluation.baselines.*` も同様です。

### チェックポイント選択（本番プロトコル）

既定は **`training.selection.metric: val_topo`**（`configs/base.yaml`）。学習中は `val_topo_loss` 最小の `best_model.pth` を保存し、DBSCAN grid は学習後の paper eval で一度だけ実行します。チューニング用に `wdist` / `dbscan_mcc` を選ぶ場合は docstring（`tda_ml/model_selection.py`）を参照してください。

### 出力パス規約

`tda_ml/run_paths.py` が `outputs/supervised` / `outputs/supervised_no_cls` / `outputs/tune` 配下の slug・タイムスタンプ命名を統一します。新規実験フォルダは `scripts/new_experiment.sh` を使用してください。

## 副次: バックエンドパイプライン比較（論文主表ではない）

`run_backend_multiseed.py` は **二次比較 / CI smoke** 用です。論文主表の入口ではありません。

```bash
uv run python experiments/run_backend_multiseed.py \
  --base-config reproduce \
  --epochs 50 \
  --seeds 42 123 456 789 1024 \
  --backends mahalanobis ellphi \
  --out-base outputs/backend_compare
```

再開・ロック・期待成果物は `README.md` の Backend pipeline comparison 節を参照。

### バックエンド比較と outlier 確率の重み（非対称）

- `run_backend_multiseed.py` のマルチシード・バックエンド比較は、**距離バックエンドだけを切り替えた純粋な ablation ではありません**（学習パイプライン全体の比較です）。
- 位相損失では **`mahalanobis`** が outlier **確率による重み付け**を距離行列に織り込める一方、**`ellphi`** では未実装のため `prob_weighting=false` を明示する。`true` は黙って無視せず hard-fail する。
- 結果は **同一スケジュール・同一設定表面**（典型: `configs/reproduce.yaml`）上の **2 本のフルパイプライン**として読み、距離実装だけの効果に還元しないでください。

位相損失用の距離行列は `model.topology_loss.distance_backend` ごとに別定義です。**`mahalanobis`** では、学習で予測した **outlier 確率 `probs`** を距離の重み付けに織り込めます（`tda_ml.topology.compute_anisotropic_distance_matrix`）。**`ellphi`** では楕円の接触距離のみを用い、`run_backend_multiseed.py` が `prob_weighting=false` を明示します。未実装の確率重みを要求すると `tda_ml.distance_backend.compute_distance_matrix_batch` が hard-fail します。

したがって、`run_backend_multiseed.py` で同じ YAML を回しても、**位相損失が見ている距離空間はバックエンド間で同一ではありません**。ここでは「同一のデータ・スケジュール・設定表面での再現パイプライン比較」を意図しており、**両バックエンドが数学的に完全に同型の重み付き距離目的関数を共有する**という読み方はしません。`ellphi` 側に Mahalanobis の確率重みに相当する項を無理に足す予定はなく、比較の解釈は本節および `README.md` の Known constraints に従ってください。

### ellphi + power：二目的チューニング（実験メモ）

no_cls・local_pca 教師・`size_mode=power` スタックでは、`tune_objectives.sh` が **W-Dist 最小**と **DBSCAN MCC 最大**の 2 本の Optuna study を実行し、`run_paper_30ep_multiseed.sh` が固定した best 重みで 30ep 本番を実行します。

要点: **学習 topo loss と教師 PD は ellphi**；**MCC のチューニング objective と paper eval の DBSCAN は mahalanobis**（filtration 時刻をクラスタリング距離に使わない）。

**重み固定プロトコル（重要）:** ハイパーパラメータ探索（Optuna）は **seed 42 の 20ep proxy で 1 回だけ**行い、得られた best 重み（`w_topo` / `w_aniso` / `w_size` / `lr`）を **5 つのデータ seed（42/123/456/789/1024）すべての 30ep 本番に固定**して適用します。**データ seed ごとの再チューニングは行いません。** 論文の mean ± std はこの固定重みの下でのデータ seed 間ばらつきです。

**H1-only 移行:** `homology_dimensions=[1]` 導入前に生成した tune JSON（旧 `0709_*` など）は目的関数が異なるため再利用しません。新しい best JSON は H1-only 契約（homology、teacher、probability weighting、anisotropy mode）を記録し、本番 preflight は契約キーの欠落・不一致を hard-fail します。H1-only スタックで二目的を再チューニングした後、その重みで 30ep 本番を再学習してください。

## 教師あり学習の目的関数（論文 Methods 用）

本線 `tda_ml/` の学習は **点ラベル BCE** と **clean 点群の $H_1$ 持久図との Wasserstein 教師** を併用します（`configs/reproduce.yaml` 系）。

\[
\mathcal{L}
= w_{\mathrm{class}}\mathcal{L}_{\mathrm{BCE}}
+ w_{\mathrm{topo}}\, W_2^2\!\bigl(\mathrm{PD}_{H_1}(D^{\theta,p}),\,\mathrm{PD}_{H_1}(X_{\mathrm{clean}})\bigr)
+ w_{\mathrm{size}}\mathcal{L}_{\mathrm{size}}
+ w_{\mathrm{aniso}}\mathcal{L}_{\mathrm{aniso}}.
\]

**楕円パラメータ（主表の既定）** — 局所 PCA の $a_{i,\mathrm{base}}, b_{i,\mathrm{base}}, \theta_{i,\mathrm{base}}$ に対し、学習可能な補正 $\Delta a_i,\Delta b_i,\Delta\theta_i$（`topology_head` 出力）で:

\[
a_i = a_{i,\mathrm{base}}\, e^{\Delta a_i},\quad
b_i = b_{i,\mathrm{base}}\, e^{\Delta b_i},\quad
\theta_i = \theta_{i,\mathrm{base}} + \tanh(\Delta\theta_i)\frac{\pi}{2}.
\]

clip や sigmoid による軸倍率の暗黙クリップは行わない。ellphi 等で退化が起きた run は `run_status: failed` として記録する（[Computational Reproducibility skill](https://github.com/t-uda/skills/blob/main/skills/computational-reproducibility/SKILL.md)）。

**正則化（`tda_ml/losses.py`）** — 主表は `size_mode: power`:

\[
\mathcal{L}_{\mathrm{size}}
= \frac{1}{N}\sum_i \left(\frac{M_i^2 + m_i^2}{\mathrm{ref}}\right)^{\gamma},\quad
M_i=\max(a_i,b_i),\; m_i=\min(a_i,b_i),
\]

（`size_ref` \(=\mathrm{ref}\)、`size_power` \(=\gamma\)；主表は ref=1.34, γ=1.5）。
`size_mode: quadratic`（\(\frac{1}{N}\sum_i (M_i^2+m_i^2)\)）は非主表の共有プロファイル用。

主表 power 30ep config（`paper_n100_o20_nocls_h1_ellphi_lpca_power`）では
`homology_dimensions: [1]`（H1-only Wasserstein）と `aniso_mode: elongate` を用いる。
ellphi 退化（NaN 共分散・接線距離未定義など）は
`run_status: failed` とする（[Computational Reproducibility skill](https://github.com/t-uda/skills/blob/main/skills/computational-reproducibility/SKILL.md)）。

非主表 ablation では `aniso_mode: linear`
（$\mathcal{L}_{\mathrm{aniso}} = \frac{1}{N}\sum_i R_i$、$R_i=M_i/m_i$）や
`aniso_mode: barrier`
（$\mathcal{L}_{\mathrm{aniso}} = \frac{10}{N}\sum_i \mathrm{ReLU}(R_i-\tau)^2$）を使う。
これらの ablation config は公開ツリーに含めず、ローカル `configs/archive/` のみ。

**Mahalanobis 距離**（`tda_ml/topology.py`）は outlier 確率 $p_i$ により二乗距離を
$1/\bigl((1-p_i)(1-p_j)\bigr)$ で重み付け（`INLIER_PROB_MIN` で下限クリップ）。

**数値安定化のみの定数**（モデリング床ではない）は `tda_ml/numerical_eps.py` に集約し、付録で列挙します。encoder の `clamp(0.2)` や legacy の `+10^{-4}` といった**論文に無い床は削除済み**（issue #59）。中心一致を含む ellphi 退化は補正せず hard-fail します。

## 図・定性出力

学習中の楕円・点スナップショットは `tda_ml.trainer` → `tda_ml.visualization.visualize` が
`<run_dir>/images/` に書き出します。論文用の静的図アセットはローカルの `docs/`（git 外）で管理します。

楕円パラメータ → 共分散 → 描画は `tda_ml/visualization.py` と `tda_ml/geometry.py` を参照してください。

## 自動テスト

ローカルでは:

```bash
uv run python -m unittest discover -s tests -v
# または
uv run pytest
```

PR および **`main` と `feature/**` への push** のたびに、CI では **ruff**、テスト、軽量な
**backend smoke**（`run_backend_multiseed.py` を `--epochs 1`・1 seed・`mahalanobis`）が走ります。
**論文本番（30ep × 5 seed）は CI では実行しません。**

## 論文提出時のスナップショット

論文投稿時は、提出結果と一致する **git コミットを固定**し、`CITATION.cff` およびリポジトリ URL を、正本の組織リポジトリ（公式が `uda-lab/TDA-ML` のときはそれ）と揃えて引用してください。
