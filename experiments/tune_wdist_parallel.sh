#!/usr/bin/env bash
# Optuna tune: local_pca + size_mode=power, objective = val topo W-Dist.
#
# Train per trial: val_topo ckpt. Score: mean val topo W-Dist (ellphi teacher PD).
# Search bands match tune_mcc power protocol (--narrow-search).
#
# Usage:
#   bash experiments/tune_wdist_parallel.sh [N_WORKERS] [N_TRIALS] [TUNE_EPOCHS] [BACKEND]
#
# Detached:
#   bash experiments/launch_detached_screen.sh tune_wdist \
#     outputs/tune/wdist/launcher.log \
#     experiments/tune_wdist_parallel.sh 4 24 20 ellphi

set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

N_WORKERS="${1:-4}"
N_TRIALS="${2:-24}"
TUNE_EPOCHS="${3:-20}"
BACKEND="${4:-ellphi}"
BASE_CONFIG="${BASE_CONFIG:-tune_rings}"
OUT_BASE="${OUT_BASE:-outputs/tune/wdist}"
STUDY_NAME="${STUDY_NAME:-tune_wdist_rings_${BACKEND}}"
STORAGE="sqlite:///${OUT_BASE}/study.db"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-12}"
TRIALS_PER_WORKER="${TRIALS_PER_WORKER:-${N_TRIALS}}"

mkdir -p "${OUT_BASE}"
cat > "${OUT_BASE}/PURPOSE.md" <<EOF
# H0+H1 + near-tangent + elongate_barrier W-Dist tune (topo=${BACKEND})

Stack: local_pca teacher, \`size_mode=power\`, \`homology_dimensions=[0, 1]\`,
\`outlier_mode=local_pca_tangent\` (jitter ±30°, clearance 0.08),
\`teacher_local_pca_major_scale=0.4\`, \`aniso_mode=elongate_barrier\`
(\`aniso_barrier_threshold=6.0\`).
Train ckpt: val_topo. Objective: mean val **topo W-Dist**.

Search constraint (declared method, not a silent filter): narrow band
\`w_topo∈[0.008,0.045]\`, \`w_aniso∈[0.05,0.15]\`,
\`w_size∈[0.10,0.40]\`, \`lr∈[1.2e-4,3.0e-4]\`.

Launch: \`bash experiments/tune_wdist_parallel.sh ${N_WORKERS} ${N_TRIALS} ${TUNE_EPOCHS} ${BACKEND}\`
EOF

echo "Power W-Dist tune: ${N_WORKERS} workers, ${N_TRIALS} trials, ${TUNE_EPOCHS}ep, topo=${BACKEND}"
echo "OUT_BASE=${OUT_BASE}"

# Create the Optuna study once before spawning workers: concurrent
# create_study(load_if_exists=True) on a fresh sqlite file races inside the
# alembic schema migration ("table alembic_version already exists") and kills
# the losing worker at startup.
STUDY_NAME="${STUDY_NAME}" STORAGE="${STORAGE}" uv run python - <<'PY'
import os

import optuna

optuna.create_study(
    study_name=os.environ["STUDY_NAME"],
    storage=os.environ["STORAGE"],
    direction="minimize",
    load_if_exists=True,
)
print(f"study initialized: {os.environ['STUDY_NAME']}")
PY

pids=()
for i in $(seq 0 $((N_WORKERS - 1))); do
  SEED=$((42 + i))
  OMP_NUM_THREADS=${THREADS_PER_WORKER} \
  MKL_NUM_THREADS=${THREADS_PER_WORKER} \
  OPENBLAS_NUM_THREADS=${THREADS_PER_WORKER} \
  nohup uv run python -u experiments/tune_wdist.py \
    --base-config "${BASE_CONFIG}" \
    --n-trials "${TRIALS_PER_WORKER}" \
    --max-complete-trials "${N_TRIALS}" \
    --n-startup-trials 8 \
    --tune-epochs "${TUNE_EPOCHS}" \
    --backend "${BACKEND}" \
    --size-mode power \
    --narrow-search \
    --out-base "${OUT_BASE}" \
    --storage "${STORAGE}" \
    --study-name "${STUDY_NAME}" \
    --seed "${SEED}" \
    > "${OUT_BASE}/worker_${i}.log" 2>&1 &
  pids+=($!)
  echo "  worker ${i}: PID $!, log ${OUT_BASE}/worker_${i}.log"
  sleep 1
done

for pid in "${pids[@]}"; do wait "${pid}"; done

uv run python -u experiments/tune_wdist.py \
  --base-config "${BASE_CONFIG}" \
  --n-trials "${N_TRIALS}" \
  --max-complete-trials "${N_TRIALS}" \
  --tune-epochs "${TUNE_EPOCHS}" \
  --backend "${BACKEND}" \
  --size-mode power \
  --narrow-search \
  --out-base "${OUT_BASE}" \
  --storage "${STORAGE}" \
  --study-name "${STUDY_NAME}" \
  --write-best

echo "Done: ${OUT_BASE}/best_wdist_${BACKEND}.json"
