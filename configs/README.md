# Configuration layout

Canonical YAML files live **in this directory** (deep-merged with `base.yaml` by
`tda_ml.config.load_config`). Load by stem: `load_config("paper_rings")`.

## Naming

Short stems encode **role + experiment identity**. Contract details (ellphi,
local_pca, power, barrier τ, major_scale, n/o sizes) live **inside** the YAML /
preflight — not in the filename.

| Stem | Role |
|------|------|
| `paper_rings` | Main-table 30ep production |
| `tune_rings` | Optuna base (W-Dist and MCC studies) |
| `paper_rings_notopo` | Ablation: `w_topo=0` |
| `paper_rings_notopo_lock` | Ablation: PH-off + orientation lock |
| `methods_mnist_neartangent` | Opt-in template for other geometry (retune) |
| `reproduce` / `dev` / `prod` / `test_fast` | CI smoke / light local / CPU profile / quick check |

## 正本（rings 再現）

| File | Role |
|------|------|
| `base.yaml` | Shared defaults (merged first). |
| `reproduce.yaml` | CI / secondary smoke (`run_backend_multiseed.py`). |
| `dev.yaml` | Small subset, few epochs (local wiring). |
| `prod.yaml` | Longer CPU profile (non-paper). |
| `test_fast.yaml` | Short local verification. |
| `paper_rings.yaml` | **Paper production** (H0+H1, thin rings, barrier, major=0.083, cliff). |
| `tune_rings.yaml` | Shared Optuna tune base. |

## Ablation（論文 claim (ii)）

| File | Role |
|------|------|
| `paper_rings_notopo.yaml` | PH-off (`w_topo=0`); `selection=dbscan_mcc`. |
| `paper_rings_notopo_lock.yaml` | PH-off + `freeze_ellipse_angle` + `enforce_a_ge_b`. |

## Methods / 別幾何の再チューニング

| File | Role |
|------|------|
| `methods_mnist_neartangent.yaml` | MNIST near-tangent + H1 + barrier. Not main table. Copy + edit `data.*` for new data; retune with `BASE_CONFIG=...`. |

Do not reuse rings `--tune-json` on a different geometry (preflight hard-fails).

## 置かないもの

Probe / dated YAML → `configs/archive/` only (gitignored).

## Keys (training stack)

Missing required keys **hard-fail**. See `REPRODUCIBILITY.md` for the rings
contract (`RINGS_NO_CLS_CONTRACT`). Training PD backend is **ellphi only**;
paper MCC DBSCAN uses **mahalanobis** clustering distance.
