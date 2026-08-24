# Configuration layout

Canonical YAML files live **in this directory** (deep-merged with `base.yaml` by
`tda_ml.config.load_config`). Load by stem: `load_config("paper_mnist_h1")`.

## Naming

Short stems encode **role + experiment identity**. Contract details (ellphi,
local_pca, power, n/o sizes) live **inside** the YAML / preflight — not in the
filename.

| Stem | Role |
|------|------|
| `paper_mnist_h1` | Main-table 30ep (MNIST + uniform, H1-only) |
| `tune_mnist_h1` | Optuna base (W-Dist and MCC studies) |
| `methods_mnist_neartangent` | Opt-in near-tangent + barrier (retune template) |
| `reproduce` / `dev` / `prod` / `test_fast` | CI smoke / light local / CPU profile / quick check |

Thin rings main table lands in the follow-up PR (`paper_rings` / `tune_rings`).

## 正本（本 PR・MNIST）

| File | Role |
|------|------|
| `base.yaml` | Shared defaults (merged first). |
| `reproduce.yaml` | CI / secondary smoke (`run_backend_multiseed.py`). |
| `dev.yaml` | Small subset, few epochs (local wiring). |
| `prod.yaml` | Longer CPU profile (non-paper). |
| `test_fast.yaml` | Short local verification. |
| `paper_mnist_h1.yaml` | **Paper production** (MNIST + uniform, H1, local_pca, ellphi, power). |
| `tune_mnist_h1.yaml` | Shared Optuna tune base. |

## Methods / 別幾何の再チューニング

| File | Role |
|------|------|
| `methods_mnist_neartangent.yaml` | Near-tangent + H1 + `elongate_barrier`. Not main table. |

## 置かないもの

Probe / dated YAML → `configs/archive/` only (gitignored). Training PD is
**ellphi only**; paper MCC DBSCAN uses **mahalanobis** clustering distance.
