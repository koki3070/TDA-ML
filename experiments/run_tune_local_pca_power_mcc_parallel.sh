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
#   bash experiments/run_tune_local_pca_power_mcc_parallel.sh [N_WORKERS] [N_TRIALS] [TUNE_EPOCHS] [BACKEND] [DBSCAN_BACKEND]
#
# Detached:
#   bash experiments/launch_detached_screen.sh tune_power_mcc_maha \
#     outputs/tune/pwr_mcc_maha/launcher.log \
#     experiments/run_tune_local_pca_power_mcc_parallel.sh 4 24 20 ellphi mahalanobis

set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

N_WORKERS="${1:-4}"
N_TRIALS="${2:-24}"
TUNE_EPOCHS="${3:-20}"
BACKEND="${4:-ellphi}"
DBSCAN_BACKEND="${5:-mahalanobis}"
BASE_CONFIG="elongate_n100_no_cls_tune_local_pca_ellphi_power_mcc"
OUT_BASE="${OUT_BASE:-outputs/tune/pwr_mcc_dbscan_${DBSCAN_BACKEND}}"
STUDY_NAME="elongate_local_pca_power_mcc_${BACKEND}_dbscan_${DBSCAN_BACKEND}"
STORAGE="sqlite:///${OUT_BASE}/study.db"
THREADS_PER_WORKER=12

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

Launch: \`bash experiments/run_tune_local_pca_power_mcc_parallel.sh ${N_WORKERS} ${N_TRIALS} ${TUNE_EPOCHS} ${BACKEND} ${DBSCAN_BACKEND}\`
EOF

echo "Power MCC tune: ${N_WORKERS} workers, ${N_TRIALS} trials, ${TUNE_EPOCHS}ep, topo=${BACKEND}, DBSCAN=${DBSCAN_BACKEND}"
pids=()
for i in $(seq 0 $((N_WORKERS - 1))); do
  SEED=$((42 + i))
  OMP_NUM_THREADS=${THREADS_PER_WORKER} \
  MKL_NUM_THREADS=${THREADS_PER_WORKER} \
  OPENBLAS_NUM_THREADS=${THREADS_PER_WORKER} \
  nohup uv run python -u experiments/tune_elongate_mcc.py \
    --base-config "${BASE_CONFIG}" \
    --n-trials "${N_TRIALS}" \
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

uv run python -u experiments/tune_elongate_mcc.py \
  --base-config "${BASE_CONFIG}" \
  --n-trials "${N_TRIALS}" \
  --tune-epochs "${TUNE_EPOCHS}" \
  --backend "${BACKEND}" \
  --dbscan-backend "${DBSCAN_BACKEND}" \
  --size-mode power \
  --out-base "${OUT_BASE}" \
  --storage "${STORAGE}" \
  --study-name "${STUDY_NAME}" \
  --write-best

echo "Done: ${OUT_BASE}/best_elongate_mcc_${BACKEND}_dbscan_${DBSCAN_BACKEND}.json"
