# Third-party pins

## `pytorch_topological.ref`

Pins the commit checked out under `pytorch-topological/` for the `torch_topological` path dependency in `pyproject.toml`.

- **Fetch / sync:** `./scripts/ensure_pytorch_topological.sh` (used by CI and `README.md`).
- **Bump upstream:** choose a new commit on [aidos-lab/pytorch-topological](https://github.com/aidos-lab/pytorch-topological), update the SHA in `pytorch_topological.ref`, run the ensure script, then `uv sync` and `uv run ruff check .`.

## `ellphi.ref`

Pins the commit checked out under `ellphi_repo/` for the `ellphi` path dependency in `pyproject.toml`. The pinned fork adds the differentiable `pdist_tangency_grad` API used by `tda_ml/ellphi_torch.py`.

- **Fetch / sync:** `./scripts/ensure_ellphi_repo.sh` (used by CI and `README.md`).
- **Bump fork:** choose a new commit on [koki3070/ellphi](https://github.com/koki3070/ellphi) (based on [t-uda/ellphi](https://github.com/t-uda/ellphi)), update the SHA in `ellphi.ref`, run the ensure script, then `uv sync` and `uv run pytest`.

`.gitmodules` may list the same repositories; the ref files are the canonical pins when the parent repo does not record submodule gitlinks.
