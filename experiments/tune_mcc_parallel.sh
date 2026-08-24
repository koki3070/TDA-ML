#!/usr/bin/env bash
# Optuna tune: local_pca + size_mode=power, objective = val DBSCAN MCC.
#
# Distance backends by role:
#   BACKEND        (arg 4, default ellphi)      = training topo-loss (ellipse tangency).
#   DBSCAN_BACKEND (arg 5, default mahalanobis) = clustering distance for MCC objective.
#
# Train per trial: val_topo ckpt (fast). Score: one DBSCAN grid on val at trial end.
#
# Usage:
#   bash experiments/tune_mcc_parallel.sh [N_WORKERS] [N_TRIALS] [TUNE_EPOCHS] [BACKEND] [DBSCAN_BACKEND]
#
# Detached:
#   bash experiments/launch_detached_screen.sh tune_mcc_maha \
#     outputs/tune/mcc_maha/launcher.log \
#     experiments/tune_mcc_parallel.sh 4 24 20 ellphi mahalanobis

set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

N_WORKERS="${1:-4}"
N_TRIALS="${2:-24}"
TUNE_EPOCHS="${3:-20}"
BACKEND="${4:-ellphi}"
DBSCAN_BACKEND="${5:-mahalanobis}"
BASE_CONFIG="${BASE_CONFIG:-tune_n100_o20_nocls_h1_ellphi_lpca_power}"
OUT_BASE="${OUT_BASE:-outputs/tune/mcc_dbscan_${DBSCAN_BACKEND}}"
STUDY_NAME="${STUDY_NAME:-tune_mcc_${BACKEND}_dbscan_${DBSCAN_BACKEND}}"
STORAGE="sqlite:///${OUT_BASE}/study.db"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-12}"
TRIALS_PER_WORKER="${TRIALS_PER_WORKER:-${N_TRIALS}}"

mkdir -p "${OUT_BASE}"
cat > "${OUT_BASE}/PURPOSE.md" <<EOF
# local_pca + power size MCC tune (topo=${BACKEND}, DBSCAN=${DBSCAN_BACKEND})

Stack: local_pca teacher, \`size_mode=power\`.
Train topo-loss backend: ${BACKEND} (ellipse tangency filtration).
DBSCAN objective backend: ${DBSCAN_BACKEND} (point-to-point clustering distance).
Search: w_topo, w_aniso, w_size, lr (narrow; w_size floor 0.1 in script).
Train ckpt: val_topo. Trial objective: val DBSCAN MCC (grid once per trial).

Rationale: ellphi tangency time is a filtration parameter, not a clustering
distance; Mahalanobis is the geometrically meaningful DBSCAN metric.

Launch: \`bash experiments/tune_mcc_parallel.sh ${N_WORKERS} ${N_TRIALS} ${TUNE_EPOCHS} ${BACKEND} ${DBSCAN_BACKEND}\`
EOF

echo "Power MCC tune: ${N_WORKERS} workers, ${N_TRIALS} trials, ${TUNE_EPOCHS}ep, topo=${BACKEND}, DBSCAN=${DBSCAN_BACKEND}"
echo "OUT_BASE=${OUT_BASE}"

# Create the Optuna study once before spawning workers: concurrent
# create_study(load_if_exists=True) on a fresh sqlite file races inside the
# alembic schema migration ("table alembic_version already exists") and kills
# the losing worker at startup. Same guard as the W-Dist launcher.
STUDY_NAME="${STUDY_NAME}" STORAGE="${STORAGE}" uv run python - <<'PY'
import os

import optuna

optuna.create_study(
    study_name=os.environ["STUDY_NAME"],
    storage=os.environ["STORAGE"],
    direction="maximize",
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
  nohup uv run python -u experiments/tune_mcc.py \
    --base-config "${BASE_CONFIG}" \
    --n-trials "${TRIALS_PER_WORKER}" \
    --max-complete-trials "${N_TRIALS}" \
    --n-startup-trials 8 \
    --tune-epochs "${TUNE_EPOCHS}" \
    --backend "${BACKEND}" \
    --dbscan-backend "${DBSCAN_BACKEND}" \
    --size-mode power \
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

uv run python -u experiments/tune_mcc.py \
  --base-config "${BASE_CONFIG}" \
  --n-trials "${N_TRIALS}" \
  --max-complete-trials "${N_TRIALS}" \
  --tune-epochs "${TUNE_EPOCHS}" \
  --backend "${BACKEND}" \
  --dbscan-backend "${DBSCAN_BACKEND}" \
  --size-mode power \
  --out-base "${OUT_BASE}" \
  --storage "${STORAGE}" \
  --study-name "${STUDY_NAME}" \
  --write-best

echo "Done: ${OUT_BASE}/best_mcc_${BACKEND}_dbscan_${DBSCAN_BACKEND}.json"
