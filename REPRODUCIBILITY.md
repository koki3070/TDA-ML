# 再現性ガイド

この文書は **読者向け** です。実験の再現手順、データと成果物の置き場所、図がコードのどこに対応するかをまとめています。内部手順・監査メモなど（従来 `docs/` に置いていたもの）は **この公開リポジトリの git ツリーには含めません**（ローカルや別管理で保持してください）。

## リポジトリに含まれる範囲（目安）

- **含む:** `tda_ml/`、**`configs/` 直下の正本 YAML**（`base` / `reproduce` / `dev` / `prod` / `test_fast`、主表 `paper_rings` / `tune_rings`、ablation `paper_rings_notopo*`、別幾何テンプレ `methods_mnist_neartangent`）、`tests/`、`scripts/ensure_*`、論文・再現・再チューニング用 `experiments/`（`launch_detached_screen.sh` 含む）、および `README.md` / `REPRODUCIBILITY.md` / `pyproject.toml` / `uv.lock` / `LICENSE` / `CITATION.cff` などのメタデータ。
- **含めない:** `docs/` 以下（**ローカル実験メモ**；公開方針で git に入れる場合は別途決定）、`configs/archive/`（履歴用 YAML を置く場合は **ローカルのみ**）、`scratch/`、`outputs/`、`data/`、`.cursor/` など。`load_config("archive/...")` は、手元に `configs/archive/*.yaml` を置いた場合にのみ使えます。

### 論文比較（W-Dist / MCC 二目的）で使う `experiments/`

**論文主表の提案:** W-Dist tune 重みの 30ep 5-seed（`run_paper_30ep_multiseed.sh wdist`）、`thin_rings` + `ring_radial`。
**主張:** (i) clean PD 教師付きの提案は、同一 test 雲だけの教師なし（ADBSCAN / IF / LOF / Euclid DBSCAN）より動径外れの **MCC が高い**（5 seed の mean ± sample std；記述的比較。同等性検定は行わない）。(ii) 同一レシピで `w_topo=0` にすると向きが倒れ、接線ロックや aniso 強化でも PH-on に届かない。主表に Topo W. 列は載せない。
**比較の非対称:** 教師なしは学習時の clean 参照を使わない。提案法は学習時のみ clean PD を教師にし、test ではラベルを使わない。ADBSCAN は学習なしの局所 PCA 楕円ベースライン。
**正本 config:** `paper_rings`（`w_class=0`, `homology_dimensions=[0, 1]`, `dataset_type=thin_rings`, `outlier_mode=ring_radial`, `aniso_mode=elongate_barrier`, `aniso_barrier_threshold=6.0`, `teacher_local_pca_major_scale=0.083`, `require_val_topo_cliff=true`）。
出力先は `WDIST_OUT` / `MCC_OUT` / `LOG_ROOT`（既定: `outputs/supervised/paper30_*`）で明示する（生成物は git に含めない）。

| 区分 | パス |
|------|------|
| 本番 5-seed | `run_paper_30ep_multiseed.sh`, `run_paper_30ep.py`, `aggregate_paper_multiseed.py` |
| paper eval | `eval_paper.py`（`best_model.pth` のみ） |
| ベースライン | `eval_baselines.py`, `eval_fair_oneclass_rings.py` |
| チューニング（重みの出所） | `tune_wdist.py`, `tune_mcc.py`, `tune_wdist_parallel.sh`, `tune_mcc_parallel.sh`, `tune_objectives.sh` |
| 長時間ジョブ補助 | `launch_detached_screen.sh`（SSH 切断でも Optuna / 30ep を継続） |

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

**3. ベースライン（ADBSCAN 等）と fair 教師なし**

```bash
uv run python experiments/eval_baselines.py \
  --base-config paper_rings \
  --out-dir outputs/paper_baselines

uv run python experiments/eval_fair_oneclass_rings.py \
  --base-config paper_rings \
  --out-dir outputs/fair_oneclass_rings
```

単一 seed・手動 eval:

```bash
uv run python experiments/run_paper_30ep.py \
  --tune-json outputs/tune/wdist/best_wdist_ellphi.json \
  --seed 42

uv run python experiments/eval_paper.py \
  --run-dir outputs/supervised/.../paper_s42_<stamp> \
  --base-config paper_rings \
  --split val
```

長時間の Optuna / 30ep は `experiments/launch_detached_screen.sh` で screen 経由起動できる（SSH 切断対策。論文主表の入口そのものではない）。

### 別データ / 別幾何での再チューニング

rings 主表の `--tune-json` を別幾何に流用しない。手順の型:

1. `methods_mnist_neartangent` または `tune_rings` をコピーし、`data.*` と契約キー（homology / teacher / aniso / major_scale 等）を明示する。
2. `BASE_CONFIG=<その YAML 名>` で `bash experiments/tune_objectives.sh wdist`（必要なら `both`）。
3. 得た best JSON を `--tune-json` に渡し `run_paper_30ep_multiseed.sh` / `run_paper_30ep.py` で本番。

公開テンプレ: `methods_mnist_neartangent`（MNIST near-tangent + H1）。

### W-Dist 契約（場所ごとの定義）

| 経路 | homology | 距離 / filtration | 備考 |
|------|----------|-------------------|------|
| 訓練 `TopologicalLoss` / eval `compute_topo_wdist` | config の `homology_dimensions`（主表は `[0, 1]`） | ellipse filtration + Wasserstein-2²（torch_topological） | 教師は `loss.teacher_mode`（主表 `local_pca`） |
| Gudhi `persistence.compute_w_distance` | H1-only | Euclidean Alpha / 点座標 | legacy baseline 用。主表の ellipse W-Dist とは別物 |
| `metrics` の W-Dist | 上記どちらかを明示引数で選択 | 引数不足は **hard-fail**（黙って 0 にしない） | |

主表・チューニングの preflight は `RINGS_NO_CLS_CONTRACT`（`homology_dimensions=[0, 1]`、`dataset_type=thin_rings`、`outlier_mode=ring_radial` および宣言された `ring_*`、`teacher_mode=local_pca`、`prob_weighting=false`、`aniso_mode=elongate_barrier`、`aniso_barrier_threshold=6.0`、`distance_backend=ellphi`、`size_mode=power`、`w_class=0.0`、`teacher_local_pca_k=10`、`teacher_local_pca_normalize_axes=true`、`teacher_local_pca_major_scale=0.083`）の明示を要求する。欠落や不一致は実行前に hard-fail する。本番 30ep は `--tune-json`（rings Optuna best）必須で、YAML 埋め込みの旧重みでは起動しない。

## 命名移行（PR #7）と leftover `pwr_s*` — 破壊的

旧本番 run-dir は `pwr_s{seed}_*`。現行は `paper_s{seed}_*`。
`aggregate_paper_multiseed.py` / `paper_run_freshness.py` は同一 `out_base` に旧 `pwr_s*` がある場合、または `pwr_s*` と `paper_s*` が混在する場合 **hard-fail** する。検査は **`--seed` 単位ではなく `out_base` 全体**。freshness CLI はこの拒否と val_topo cliff 失敗（empty-result）を **exit 2** にする（exit 1 = 真の metrics 欠落のみ）。30ep driver が旧 tree / cliff 失敗を missing と誤認して学習を始めない。

## val_topo cliff と model_seed 再試行（宣言済み Methods）

rings 主表は `require_val_topo_cliff=true`（既定 `val_topo_cliff_max=0.3`）。崖を越えなかった run は `empty-result` であり、集計に入れない。
`run_paper_30ep.py` は `CLIFF_INIT_RESTARTS`（rings 既定 8）個の異なる `training.model_seed` を試し、各 attempt を manifest（`cliff_init_attempt`, `cliff_init_model_seeds`）に記録する。これは暗黙フォールバックではなく宣言された再初期化である。失敗 attempt の tree が成功 run と混在する場合、freshness は **ambiguous（exit 2）** にする。


**教師スケール:** local-PCA 教師は向き・アスペクトのため `normalize_axes=true`（単位長軸）のあと、rings では val-clean の raw LPCA major 中央値 **`teacher_local_pca_major_scale=0.083`** で major を揃える。単位教師（major=1）のままでは filtration が膨らみ向き選好が歪む。このキーを欠く旧 tune JSON は契約不一致で再利用不可（現行スタックで再チューニングが必要）。

**異方性天井（Methods）:** 主表は `aniso_mode: elongate_barrier`（`aniso_barrier_threshold=6.0`）。素の `elongate` は伸長のみで aspect 上限がなく、短軸→0 まで潰すと ellphi tangency が hard-fail し得る。`elongate_barrier` は伸長報酬に加え、閾値超過分へ二次罰を足す（針状退化を抑える）。plain `elongate` は ablation 契約として残す。暗黙の切替はしない。

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

## データ（thin_rings）

- 論文主表のデータは **合成 thin rings**（`tda_ml/ring_dataset.py`）。ダウンロード不要で、設定の `data.seed` から決定的に生成されます。
- 副次の `reproduce.yaml` / CI smoke は従来どおり共有プロファイルを使い、必要なら `data/` 配下のキャッシュを参照します（`data/` は git 対象外）。
- 設定 YAML の役割分担は **`configs/README.md`** を参照（共有プロファイル + 論文用 `paper_rings` / `tune_rings`）。旧設定はローカルで `configs/archive/` に置けるが、公開クローンには同梱されない。

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
| `empty-result`（学習後） | rings Methods で `require_val_topo_cliff` 未達（悪い盆地；MCC〜0） |

`not-run` は preflight 専用です。学習中・中断 run を `not-run` と混同しないでください。

### Rings 主表の val_topo cliff

細いリング＋半径方向外れでは、学習が **相転移（崖）** を跨ぐか悪い盆地に留まるかが seed 依存になる。  
`training.require_val_topo_cliff=true` かつ明示の `val_topo_cliff_max`（rings YAML は `0.3`）のとき:

- 学習終了後に `best_val_topo_loss` が閾値を超えれば `ValTopoCliffError` → manifest `empty-result`（`logs/VAL_TOPO_CLIFF.json`）
- Optuna tune では同条件で `TrialPruned`（COMPLETE に入れない）
- 集計 `aggregate_paper_multiseed.py` は `--experiment-contract rings`（または auto）で `RINGS_NO_CLS_CONTRACT`、`--seed-cliff-policy auto` → `strict`（未達 seed があると hard-fail）。診断のみ `--seed-cliff-policy exclude-failed`
- 本番ドライバは `--cliff-init-restarts` で別の `training.model_seed` を再試行する（既定 8）

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

`tda_ml/run_paths.py` が `outputs/supervised` / `outputs/supervised_no_cls` / `outputs/tune` 配下の slug・タイムスタンプ命名を統一します。

## 副次: バックエンド smoke（論文主表ではない）

`run_backend_multiseed.py` は **二次比較 / CI smoke** 用です。論文主表の入口ではありません。
学習 PD filtration は **ellphi のみ**です。

```bash
uv run python experiments/run_backend_multiseed.py \
  --base-config reproduce \
  --epochs 1 \
  --seeds 42 \
  --backends ellphi \
  --out-base outputs/ci_smoke
```

再開・ロック・期待成果物は `README.md` の Backend pipeline smoke 節を参照。

### 学習 PD と DBSCAN の距離は別物

- **学習 topo loss / 教師 PD / topo W-Dist:** `distance_backend=ellphi` のみ。`mahalanobis` を要求すると hard-fail。
- **MCC のチューニング objective と paper eval の DBSCAN:** `mahalanobis`（点間クラスタリング距離。filtration 時刻ではない）。

### ellphi + power：二目的チューニング（実験メモ）

no_cls・local_pca 教師・`size_mode=power` スタックでは、`tune_objectives.sh` が **W-Dist 最小**と **DBSCAN MCC 最大**の 2 本の Optuna study を実行し、`run_paper_30ep_multiseed.sh` が固定した best 重みで 30ep 本番を実行します。

要点: **学習 topo loss と教師 PD は ellphi**；**MCC のチューニング objective と paper eval の DBSCAN は mahalanobis**。

**重み固定プロトコル（重要）:** ハイパーパラメータ探索（Optuna）は **seed 42 の 20ep proxy で 1 回だけ**行い、得られた best 重み（`w_topo` / `w_aniso` / `w_size` / `lr`）を **5 つのデータ seed（42/123/456/789/1024）すべての 30ep 本番に固定**して適用します。**データ seed ごとの再チューニングは行いません。** 論文の mean ± std はこの固定重みの下でのデータ seed 間ばらつきです。

**現行主表は rings H0+H1:** `RINGS_NO_CLS_CONTRACT`（`homology_dimensions=[0, 1]`、`teacher_local_pca_major_scale=0.083`）。単位教師（major=1）下での向き診断を受けて H0 を足した経緯はあるが、**現行の論文主表は H1-only ではない**。旧 MNIST H1-only / 旧スケールの tune JSON は目的関数・契約が異なるため再利用しない。新しい best JSON は rings 契約（homology、teacher、major_scale、probability weighting、anisotropy mode、rings 幾何）を記録し、本番 preflight は契約キーの欠落・不一致を hard-fail する。rings スタックで二目的を再チューニングした後、その重みで 30ep 本番を再学習してください。

## 教師あり学習の目的関数（論文 Methods 用）

本線 `tda_ml/` の学習は **点ラベル BCE** と **clean 点群の持久図との Wasserstein 教師** を併用します。論文主表は `w_{\mathrm{class}}=0` かつ `homology_dimensions=[0, 1]`（$H_0$ と $H_1$ の和）です。共有プロファイル `configs/reproduce.yaml` は分類項を残します。

\[
\mathcal{L}
= w_{\mathrm{class}}\mathcal{L}_{\mathrm{BCE}}
+ w_{\mathrm{topo}}\, W_2^2\!\bigl(\mathrm{PD}_{H_{0,1}}(D^{\theta,p}),\,\mathrm{PD}_{H_{0,1}}(X_{\mathrm{clean}})\bigr)
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

主表 power 30ep config（`paper_rings`）では
`homology_dimensions: [0, 1]` と `aniso_mode: elongate_barrier`
（`aniso_barrier_threshold: 6.0`）を用いる。
`teacher_local_pca_major_scale: 0.083` で教師長軸を rings の clean 中央値に揃える。
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

### 楕円の向き診断（定性主張のゲート）

`loss.aniso_mode=elongate` は伸長のみを報酬とし**向きを拘束しない**ため、アスペクト比が高くても
長軸がストローク法線方向を向いた checkpoint が生じ得ます。この場合「ストローク方向の異方性を獲得した」
という定性主張は成立しません。近円形楕円（宣言定数 `ORIENTATION_MIN_AXIS_GAP` 未満）では
向きが未定義です。論文本番経路（`run_paper_30ep.py` と paper preflight）は診断のためには
変更していません。向き診断用のローカルスクリプトは公開ツリーに含めません。

#### 測定済みの知見（seed 42、4 エポック診断ラン、20 雲）

| `homology_dimensions` | epoch 1 | epoch 2 | best (val_topo) | median inlier aspect |
|---|---|---|---|---|
| `[1]`（単位教師 major=1 当時） | 11.4° tangent | 71.0° **normal** | 78.1° **normal** | 6.31 |
| `[0, 1]`（同・暫定契約） | 4.7° tangent | 16.4° tangent | 18.6° tangent | 2.82 |
| `[0]` | 3.6° tangent | 6.0° tangent | 3.7° tangent | 3.23 |

単位教師（major=1）下では旧 H1-only が epoch 2 で法線向きへ反転して戻らず、H0 を含めると接線向きを維持した。上表はスケール修正前の診断記録である。**現行の論文主表契約は `[0, 1]` + `teacher_local_pca_major_scale=0.083`（thin rings）**。

**Optuna 探索帯（向き制約・再校正前）:** 第1回 H0+H1 study（`outputs/tune/wdist_h01/`）では
W-Dist 最小帯（`w_topo~0.08–0.10`, `lr~5e-4`）の val_topo-best が median |Δθ|≈40°（mixed）に
なり、接線 trial（≤30°）は `w_topo` 中央値≈0.02・`lr`≤3.2e-4 に集まりました。第2回 study
（`outputs/tune/wdist_h01_lowtopo/`）は目的関数を変えず、宣言済みの探索帯だけ
`w_topo∈[0.005,0.055]`, `lr∈[1e-4,3.5e-4]` に絞ります。rings 主表の tune 帯
（`tune_wdist.py` の narrow band）はこの校正を引き継ぐ。

これらはいずれも診断記録であり、損失コード・モデル・checkpoint 選択は変更しません。

## 自動テスト

ローカルでは:

```bash
uv run python -m unittest discover -s tests -v
# または
uv run pytest
```

PR および **`main` と `feature/**` への push** のたびに、CI では **ruff**、テスト、軽量な
**backend smoke**（`run_backend_multiseed.py` を `--epochs 1`・1 seed・`ellphi`）が走ります。
**論文本番（30ep × 5 seed）は CI では実行しません。**

## 論文提出時のスナップショット

論文投稿時は、提出結果と一致する **git コミットを固定**し、`CITATION.cff` およびリポジトリ URL を、正本の組織リポジトリ（公式が `uda-lab/TDA-ML` のときはそれ）と揃えて引用してください。
