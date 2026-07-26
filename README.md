# TDA-ML

Clean-room, reproducibility-focused implementation for anisotropic topological
denoising (Letters / applied-math reference code).

Reader-facing protocol, manifests, and failure semantics:
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

Computational discipline follows
[computational-reproducibility](https://github.com/t-uda/skills/blob/main/skills/computational-reproducibility/SKILL.md)
(no silent fallback; declared numerical constants in `tda_ml/numerical_eps.py`).

## Claim (main table)

Proposed method (W-Dist-tuned weights, 30 epochs × 5 data seeds) shows
**comparable** outlier-removal performance to Euclidean DBSCAN and ADBSCAN on
**MCC / G-Mean** (descriptive mean ± sample std over seeds; no equivalence test).
Main table does **not** include a Topo W. column.

ADBSCAN uses fixed local-PCA ellipses (no training). The proposed method trains
for 30 epochs on the same data; the comparison is not compute-matched.

## Setup

```bash
./scripts/ensure_pytorch_topological.sh
./scripts/ensure_ellphi_repo.sh
uv sync
# development (tests, ruff):
uv sync --all-groups
# Optuna tune drivers:
uv sync --extra experiments
```

`torch_topological` and `ellphi` are **local path dependencies** (pinned under
`third_party/*.ref`). A plain `git clone` is not enough; run the ensure scripts
before `uv sync`. See `third_party/README.md` when bumping pins.

## Paper production path (primary)

Main table: **W-Dist-tuned weights**, 30 epochs × 5 data seeds, `w_class=0`,
H1-only, local-PCA teacher, ellphi distance, checkpoint `best_model.pth`
(`selection=val_topo`), then val DBSCAN grid → test MCC / G-Mean.

Config: `paper_n100_o20_nocls_h1_ellphi_lpca_power`.

```bash
# 1) Tune once (Optuna sampler seeds differ per worker; data seed in YAML is 42).
#    Default MODE=both also runs the secondary MCC study; main table needs wdist.
MODE=wdist bash experiments/run_tune_local_pca_power_objectives.sh

# 2) Fixed W-Dist weights → 30ep × 5 seeds → paper eval (+ baselines separately)
bash experiments/run_teacher_local_pca_power_30ep_multiseed.sh wdist

uv run python experiments/evaluate_paper_baselines.py \
  --base-config paper_n100_o20_nocls_h1_ellphi_lpca_power \
  --out-dir outputs/paper_baselines
```

Details and contract keys: `REPRODUCIBILITY.md` / `configs/README.md`.
Generated artifacts stay under `outputs/` (not committed).

## Backend pipeline comparison (secondary / CI)

`experiments/run_backend_multiseed.py` compares **full training pipelines** under
`configs/reproduce.yaml`. It is **not** the paper main-table entrypoint and
**not** a pure distance-backend ablation.

```bash
# CI-style smoke
uv run python experiments/run_backend_multiseed.py \
  --base-config reproduce \
  --epochs 1 --seeds 42 --backends mahalanobis \
  --out-base outputs/smoke

# Full secondary comparison (local / own runner; not CI)
uv run python experiments/run_backend_multiseed.py \
  --base-config reproduce \
  --epochs 50 --seeds 42 123 456 789 1024 \
  --backends mahalanobis ellphi \
  --out-base outputs/backend_compare
```

Expected under `--out-base`: `progress_summary.csv`, `backend_stats.csv`, and
per-run `*/logs/metrics.csv`. Resume skips completed `(backend, seed, epochs)`
keys; use `--rerun-completed` to force. A lock file under `--out-base` aborts if
another process is active.

`tda_ml.main` is an internal trainer entry used by the drivers above; do not
treat direct invocation as the public protocol.

## Continuous integration

PRs and pushes to `main` / `feature/**` run `ruff`, tests, and the **1-epoch
mahalanobis smoke** above. Paper 30ep × 5-seed production is not run in CI.

## Known constraints

- Fixed paper seed set: `42 123 456 789 1024`.
- Runtime depends on device / threads / dtype; each run records a manifest.
- For topological loss, **mahalanobis** may use outlier-probability weighting;
  **ellphi** is geometry-only and requires `prob_weighting=false` (hard-fail
  otherwise). Do not read backend comparison as an isolated metric swap.

## License / Attribution

MIT — see `LICENSE`.

- **ellphi (differentiable tangency):** pinned fork via
  `./scripts/ensure_ellphi_repo.sh` (`third_party/ellphi.ref`). PyPI
  `ellphi==0.1.2` alone lacks the training `ellphi.grad` API.
- **pytorch-topological:** via `./scripts/ensure_pytorch_topological.sh`.
