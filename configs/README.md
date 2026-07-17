# Configuration layout

Canonical YAML files live **in this directory** (deep-merged with `base.yaml` by `tda_ml.config.load_config`):

| File | Role |
|------|------|
| `base.yaml` | Shared defaults; always merged first. |
| `reproduce.yaml` | Paper / official multi-seed reproduction (`run_backend_multiseed.py`). |
| `dev.yaml` | Small MNIST subset for local wiring checks (non-official). |
| `prod.yaml` | Longer CPU profile (non-official). |
| `test_fast.yaml` | Small settings for quick checks and CI smoke. |
| `elongate_n100_no_cls_full120_teacher_local_pca.yaml` | Paper no_cls production (H1-only, local_pca teacher). |
| `elongate_n100_no_cls_tune_local_pca_ellphi_power_mcc.yaml` | Optuna tune base for power + ellphi (W-Dist / MCC studies). |
| `elongate_n100_no_cls_full120_baseline.yaml` | no_cls baseline contrast (optional table column). |

**Probe / ablation / dated experiment YAML** (scaleinv, topo12, raw_axes, H1 launch variants, etc.) belongs under local `configs/archive/` and is **not** part of the publishable surface. Load with `load_config("archive/<stem>")` when present.

## Keys read by the training stack

`tda_ml.main` and `Trainer` use the following (other YAML keys are ignored).

| Section | Key | Used by | Notes |
|---------|-----|---------|--------|
| `meta` | `config_id` | `main` | Run directory prefix `<config_id>_<timestamp>`. |
| `model` | `point_dim`, `feature_dim` | `main` | Passed to `AnisotropicOutlierClassifier`. |
| `model` | `threshold` | `Trainer` | Classification threshold. |
| `model` | `topology_loss.distance_backend` | `Trainer` | `mahalanobis` or `ellphi` (set by multiseed driver per run). |
| `model` | `topology_loss.ellphi_differentiable` | `Trainer` | Default `true`; multiseed driver reads from merged config. |
| `loss` | `w_class`, `w_topo`, `w_aniso` | `Trainer` | Loss weights (fallback: `training.lambda_*`). |
| `loss` | `w_size` | `Trainer` | Size regularization scale (fallback: `training.lambda_size` / `lambda_major` / `lambda_minor`). |
| `loss` | `pos_weight`, `aniso_mode` | `Trainer` | BCE positive weight; anisotropy penalty mode. |
| `training` | `lr`, `epochs`, `grad_clip_value`, `visualize_every`, `warmup_epochs` | `main` / `Trainer` | |
| `training` | `lambda_major`, `lambda_minor`, `lambda_size` | `Trainer` | Ellipse size loss (overrides `w_size` when set). |
| `training` | `barrier_threshold`, `rotation_augmentation` | `Trainer` | Anisotropy barrier / data aug. |
| `training` | `use_amp`, `amp_dtype` | `Trainer` | CUDA AMP (optional). |
| `data` | `seed`, `train_size`, `val_size`, `test_size` | `main` | MNIST index splits (not `num_samples`). |
| `data` | `max_points`, `num_outliers`, `batch_size`, `num_workers`, … | `main` | DataLoader / dataset. |
| `outputs` | `base_dir` | `main` | Parent of per-run trees. |
| `outputs` | `log_dir`, `image_dir` | — | **Overwritten** by `main` to `<run_dir>/logs` and `/images`. |
| `outputs` | `save_every` | `main` | Periodic checkpoint interval. |
| `device` | | `main` | `auto`, `cpu`, `cuda`, or `mps`. |
| `reproducibility` | `deterministic_algorithms` | `main` | Passed to `set_global_seed`. |
| `init_checkpoint` | | `main` | Optional warm-start (top-level or via CLI). |

## Non-official scripts

Modules under `tda_ml/experiments/`, `reproduce_pd_animation.py`, and `robustness_sweep.py` may require **your own checkpoints** and optional extras. They are not part of the canonical `run_backend_multiseed.py` path.
