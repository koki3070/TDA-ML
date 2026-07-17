# TDA-ML Reproducibility Repository

This repository provides a reproducibility-focused implementation for anisotropic topological denoising experiments.

For **data layout, checkpoints, figure-related scripts, and CI tests**, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (reader-facing).

## Setup

Use `uv` as the canonical dependency manager.

`torch_topological` is a **local path dependency** (`pyproject.toml` → `pytorch-topological/`). That directory is not always present after a plain `git clone`; sync it to the pinned commit in `third_party/pytorch_topological.ref` (same as CI):

```bash
./scripts/ensure_pytorch_topological.sh
```

`ellphi` is also a **local path dependency** (`pyproject.toml` → `ellphi_repo/`). Sync the pinned fork (differentiable tangency grad API) before `uv sync`:

```bash
./scripts/ensure_ellphi_repo.sh
```

Optional: if this repository records `pytorch-topological` as a git submodule gitlink, `git submodule update --init --recursive` may populate the tree first; the ensure script still verifies the pinned commit. See `third_party/README.md` when bumping the pin.

Then install Python dependencies:

```bash
uv sync
```

For development dependencies:

```bash
uv sync --all-groups
```

Optional package extras (e.g. `robustness_sweep`, HomCloud animation, explicit Pillow):

```bash
uv sync --extra experiments --extra repro-pd-animation --extra images
```

If you use `pip` instead of `uv`, run `./scripts/ensure_pytorch_topological.sh` and `./scripts/ensure_ellphi_repo.sh` from the repository root, then install from `pyproject.toml` with `pip install .` or `pip install -e .` for an editable install; add optional extras when needed (for example `pip install -e ".[experiments,repro-pd-animation,images]"`). There is no `requirements.txt`; dependency pins live in `uv.lock` for `uv` users.

## Minimal Reproduction

Use the following command as the official reproduction entrypoint:

```bash
uv run python experiments/run_backend_multiseed.py \
  --base-config reproduce \
  --epochs 50 \
  --seeds 42 123 456 789 1024 \
  --backends mahalanobis ellphi \
  --out-base outputs/backend_compare
```

Run this command from the repository root.

Expected artifacts:

- `outputs/backend_compare/progress_summary.csv`
- `outputs/backend_compare/backend_stats.csv`
- `outputs/backend_compare/*/logs/metrics.csv`

Direct execution of `tda_ml/main.py` is non-official and not part of the canonical reproduction path.

### Resume Behavior

- `run_backend_multiseed.py` skips completed runs using the key `(backend, seed, epochs)`.
- Changing `--epochs` creates a different run key and will not be skipped.
- Use `--rerun-completed` to force rerun even when the same key already exists.
- `progress_summary.csv` header is validated on startup. If header mismatch occurs, use a fresh `--out-base` or fix the CSV manually.

### Operational Notes

- `run_backend_multiseed.py` creates a lock file under `--out-base` and aborts if another process is active there.
- `progress_summary.csv` stores `run_dir` as provided by `--out-base`; avoid sharing logs with personal absolute paths if privacy is a concern.
- Keep `metrics.csv` schema compatible with the current trainer output (`val_mcc`, `val_recall`, `val_loss` are required).

## Quick Smoke Run

For a fast wiring check before the full run:

```bash
uv run python experiments/run_backend_multiseed.py \
  --base-config reproduce \
  --epochs 1 \
  --seeds 42 \
  --backends mahalanobis \
  --out-base outputs/smoke
```

## Continuous integration

On pull requests and pushes to `main` / `feature/**`, CI runs:

- `ruff check`
- `pytest`
- A **repro smoke** only: `run_backend_multiseed.py` with `--epochs 1`, one seed, and `mahalanobis` (see `.github/workflows/ruff.yml`).

The full official command (50 epochs × five seeds × two backends) is **not** executed in CI. Run that locally or in your own runner when validating paper numbers.

## Known Constraints

- The official comparison uses a fixed five-seed set: `42 123 456 789 1024`.
- Runtime and numeric behavior can vary by `device`, `thread`, and `dtype`; effective values should be logged per run.
- External implementations are reference-only and are not redistributed in this repository.
- Scripts under `tda_ml/experiments/` are **non-official** and require your own checkpoints (see `configs/README.md`).
- `run_backend_multiseed.py` multiseed backend comparison is **not** a pure distance-backend-only ablation; it compares full training pipelines, not an isolated backend switch.
- For topological loss, **`mahalanobis`** can incorporate predicted **outlier-probability weighting** in the distance matrix; **`ellphi`** is geometry-only and requires `prob_weighting=false`.
- Read outcomes as **two full pipelines** under the same schedule and config surface, not as isolating distance-backend effects alone.

### Backend comparison: outlier-probability weighting

For **topological loss**, the batched distance matrix is built per `model.topology_loss.distance_backend`. With **`mahalanobis`**, the implementation can incorporate **predicted outlier probabilities** when forming pairwise distances (see `tda_ml.topology.compute_anisotropic_distance_matrix`). With **`ellphi`**, tangency distances are computed from ellipse geometry only; **probability-based weighting is not implemented**, so the backend comparison driver explicitly sets `prob_weighting=false`. Requesting it now hard-fails rather than silently ignoring `probs` (see `tda_ml.distance_backend.compute_distance_matrix_batch`). Hyperparameters in `configs/reproduce.yaml` are shared across backends, but **the induced metric for topological loss is not identical across backends** by design: the intended read is a reproducible pipeline comparison (same data, schedule, and config surface), not a claim that both backends optimize the exact same weighted distance objective. Elliptic contact distances are left as defined by `ellphi`; no synthetic “prob-equivalent” weighting is applied on the ellphi path.

## License / Attribution

This project is licensed under the MIT License. See `LICENSE`.

External implementations are reference-only and are not redistributed in this repository.

- **ellphi (training / differentiable tangency):** requires the pinned fork checked out via `./scripts/ensure_ellphi_repo.sh` (`third_party/ellphi.ref` → `ellphi_repo/`). PyPI `ellphi==0.1.2` alone does **not** include the `ellphi.grad` API used for training. Fork: [koki3070/ellphi](https://github.com/koki3070/ellphi) (based on [t-uda/ellphi](https://github.com/t-uda/ellphi)).
- **pytorch-topological:** [aidos-lab/pytorch-topological](https://github.com/aidos-lab/pytorch-topological) via `./scripts/ensure_pytorch_topological.sh`.

Upstream references:

- `https://github.com/t-uda/ellphi`
- `https://github.com/aidos-lab/pytorch-topological`
