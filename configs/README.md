# Configuration layout

Canonical YAML files live **in this directory** (deep-merged with `base.yaml` by
`tda_ml.config.load_config`).

## Naming (paper / tune)

`{role}_n{N}_o{O}_nocls_h1_{backend}_{teacher}_{size}`

| Token | Meaning |
|-------|---------|
| `paper` / `tune` | Production 30ep vs Optuna tune base |
| `n100` | `data.max_points=100` |
| `o20` | `data.num_outliers=20` |
| `nocls` | `loss.w_class=0` |
| `h1` | `homology_dimensions=[1]` |
| `ellphi` / `maha` | `distance_backend` |
| `lpca` / `euclid` | `teacher_mode` (`local_pca` / `euclidean`) |
| `power` | `size_mode=power` |

`aniso_mode=elongate` is the paper contract default and is **not** put in the
filename (it is a loss mode, not the experiment identity).

## 正本（公開・論文）

| File | Role |
|------|------|
| `base.yaml` | Shared defaults; always merged first. Declared keys only (no silent trainer defaults). |
| `reproduce.yaml` | Secondary backend pipeline comparison (`run_backend_multiseed.py` / CI smoke). |
| `dev.yaml` | Small MNIST subset for local wiring (non-paper). |
| `prod.yaml` | Longer CPU profile (non-paper). |
| `test_fast.yaml` | Quick checks / CI. |
| `paper_n100_o20_nocls_h1_ellphi_lpca_power.yaml` | **Paper production** (30ep, `w_class=0`, H1-only, local_pca, ellphi, power). |
| `tune_n100_o20_nocls_h1_ellphi_lpca_power.yaml` | Shared Optuna tune base for **both** W-Dist and MCC studies. |

## 置かないもの

Probe / ablation / dated experiment YAML → local `configs/archive/` only
(gitignored). Do not reintroduce rings / contam / raw_axes / near-tangent
barrier / Euclidean-teacher baseline configs here unless they become a claimed
Methods path with a matching production YAML.

Library support without public paper YAML: `tda_ml/ring_dataset.py`
(`dataset_type=thin_rings`) and `tda_ml/tangent_outliers.py` remain importable
for opt-in Methods experiments; they are **not** the main-table path unless a
public YAML above selects them. Barrier contract
(`aniso_mode=elongate_barrier`) is code-supported via explicit YAML declaration;
no public Methods YAML ships for it.

## Keys read by the training stack

`tda_ml.main` and `Trainer` use the following (other YAML keys are ignored).
Missing required keys **hard-fail** (no silent method defaults).

| Section | Key | Used by | Notes |
|---------|-----|---------|--------|
| `meta` | `config_id` | `main` | Run directory prefix `<config_id>_<timestamp>`. |
| `model` | `point_dim`, `feature_dim` | `main` | Passed to `AnisotropicOutlierClassifier`. |
| `model` | `threshold` | `Trainer` | Classification threshold. |
| `model` | `topology_loss.distance_backend` | `Trainer` | `mahalanobis` or `ellphi` (required). |
| `model` | `topology_loss.homology_dimensions` | `Trainer` / topo W-Dist | Required list. |
| `model` | `topology_loss.prob_weighting` | `Trainer` | Required bool; `ellphi` requires `false`. |
| `model` | `topology_loss.ellphi_differentiable` | `Trainer` | Required bool. |
| `loss` | `w_class`, `w_topo`, `w_aniso`, `w_size` | `Trainer` | Required under `loss.*`. |
| `loss` | `teacher_mode` | `Trainer` / topo W-Dist | Required (`euclidean` or `local_pca`). |
| `loss` | `topo_eps_scale`, `topo_scale_mode` | `Trainer` | Required filtration alignment. |
| `loss` | `teacher_local_pca_*` | `Trainer` | Required when `teacher_mode=local_pca`. |
| `loss` | `pos_weight`, `aniso_mode`, `size_mode` | `Trainer` | Mode-specific extras required when that mode is selected. |
| `training` | `lr`, `epochs`, `grad_clip_value`, `visualize_every`, `warmup_epochs` | `main` / `Trainer` | |
| `training` | `selection.metric` | `Trainer` | Required (paper: `val_topo`). |
| `data` | `seed`, sizes, `outlier_mode`, … | `main` | Explicit geometry; hard-fail if absent. |
| `outputs` | `base_dir` | `main` | Parent of per-run trees. |
| `reproducibility` | `*` | preflight / trainer | Strict defaults in `base.yaml`. |
| `init_checkpoint` | | `main` | Optional warm-start. |

## Non-paper scripts

Optional extras under `pyproject.toml` (`experiments`, `images`) support tune /
plotting. They are not the paper production path.
