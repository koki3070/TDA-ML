# TDA-ML

Clean-room, reproducibility-focused implementation for anisotropic topological
denoising on **thin synthetic rings** with radial outliers (Letters / applied-math
reference code).

Reader-facing protocol, manifests, and failure semantics:
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

Computational discipline follows
[computational-reproducibility](https://github.com/t-uda/skills/blob/main/skills/computational-reproducibility/SKILL.md)
(no silent fallback; declared numerical constants in `tda_ml/numerical_eps.py`).

## Claim (main table)

On thin rings + radial outliers (`thin_rings` / `ring_radial`), the proposed
method (clean-PD teacher, W-Dist-tuned weights, 30 epochs × 5 data seeds) shows
**higher** outlier-removal **MCC** than unsupervised peers evaluated on the same
test clouds (ADBSCAN, Isolation Forest, LOF, Euclid DBSCAN; descriptive
mean ± sample std; no equivalence test). Main table does **not** include a
Topo W. column.

Within the same recipe, setting `w_topo=0` collapses orientation unless a
declared tangent lock is applied; locked PH-off still trails PH-on (see
ablation YAMLs and `REPRODUCIBILITY.md`).

ADBSCAN uses fixed local-PCA ellipses (no training). The proposed method trains
for 30 epochs with a clean reference PD; the unsupervised comparison is not
compute-matched and does not use that teacher at test time.

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
H0+H1, thin rings + radial outliers, `elongate_barrier`, local-PCA teacher
(major scale 0.083), ellphi distance, checkpoint `best_model.pth`
(`selection=val_topo` + val_topo cliff gate), then val DBSCAN grid → test MCC /
G-Mean.

Config: `paper_rings`.

```bash
# 1) Tune once (Optuna sampler seeds differ per worker; data seed in YAML is 42).
#    Default MODE=both also runs the secondary MCC study; main table needs wdist.
MODE=wdist bash experiments/tune_objectives.sh

# 2) Fixed W-Dist weights → 30ep × 5 seeds → paper eval (+ baselines separately)
bash experiments/run_paper_30ep_multiseed.sh wdist

uv run python experiments/eval_baselines.py \
  --base-config paper_rings \
  --out-dir outputs/paper_baselines

# Fair unsupervised peers (IF / LOF / Euclid / ADBSCAN) on the same rings contract:
uv run python experiments/eval_fair_oneclass_rings.py \
  --base-config paper_rings \
  --out-dir outputs/fair_oneclass_rings
```

Details and contract keys: `REPRODUCIBILITY.md` / `configs/README.md`.
Generated artifacts stay under `outputs/` (not committed).

## Backend pipeline smoke (secondary / CI)

`experiments/run_backend_multiseed.py` is a **secondary** driver for the shared
`configs/reproduce.yaml` profile. It is **not** the paper main-table entrypoint.
Training PD filtration is **ellphi only**; paper MCC / DBSCAN still uses
mahalanobis as a clustering distance.

```bash
# CI-style smoke
uv run python experiments/run_backend_multiseed.py \
  --base-config reproduce \
  --epochs 1 --seeds 42 --backends ellphi \
  --out-base outputs/smoke
```

Expected under `--out-base`: `progress_summary.csv`, `backend_stats.csv`, and
per-run `*/logs/metrics.csv`. Resume skips completed `(backend, seed, epochs)`
keys; use `--rerun-completed` to force. A lock file under `--out-base` aborts if
another process is active.

`tda_ml.main` is an internal trainer entry used by the drivers above; do not
treat direct invocation as the public protocol.

## Continuous integration

PRs and pushes to `main` / `feature/**` run `ruff`, tests, and the **1-epoch
ellphi smoke** above. Paper 30ep × 5-seed production is not run in CI.

## Known constraints

- Fixed paper seed set: `42 123 456 789 1024`.
- Rings production uses `require_val_topo_cliff` (bad basins → `empty-result`;
  multiseed driver retries `training.model_seed`).
- Runtime depends on device / threads / dtype; each run records a manifest.
- Training PD / topo W-Dist: **`ellphi` only** (`prob_weighting=false`).
- Paper MCC DBSCAN: **`mahalanobis`** clustering distance (not filtration).

## License / Attribution

MIT — see `LICENSE`.

- **ellphi (differentiable tangency):** pinned fork via
  `./scripts/ensure_ellphi_repo.sh` (`third_party/ellphi.ref`). PyPI
  `ellphi==0.1.2` alone lacks the training `ellphi.grad` API.
- **pytorch-topological:** via `./scripts/ensure_pytorch_topological.sh`.
