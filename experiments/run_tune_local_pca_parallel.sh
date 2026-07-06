#!/usr/bin/env bash
# Optuna tune: local_pca ellphi teacher, no_cls, val_topo checkpoint, topo W-Dist objective.
#
# Usage:
#   bash experiments/run_tune_local_pca_parallel.sh [N_WORKERS] [N_TRIALS] [TUNE_EPOCHS] [BACKEND]
#
# Defaults: 8 workers, 50 trials, 20 epochs, ellphi.
#
# Detached (SSH-safe):
#   bash experiments/launch_detached_screen.sh tune_local_pca_v3 \
#     outputs/tune_elongate_ellphi_local_pca_v3/launcher.log \
#     experiments/run_tune_local_pca_parallel.sh 8 50 20 ellphi

set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

N_WORKERS="${1:-8}"
N_TRIALS="${2:-50}"
TUNE_EPOCHS="${3:-20}"
BACKEND="${4:-ellphi}"
BASE_CONFIG="elongate_n100_no_cls_tune_local_pca"

if [[ "${BACKEND}" == "mahalanobis" ]]; then
  OUT_BASE="outputs/tune_elongate_local_pca_v3"
else
  OUT_BASE="outputs/tune_elongate_${BACKEND}_local_pca_v3"
fi

STUDY_NAME="elongate_local_pca_topo_wdist_${BACKEND}_v3"
STORAGE="sqlite:///${OUT_BASE}/study.db"
THREADS_PER_WORKER=12

mkdir -p "${OUT_BASE}"
cat > "${OUT_BASE}/PURPOSE.md" <<EOF
# Optuna tune v3 — local_pca ellphi teacher (${BACKEND})

Protocol:
- Base config: \`${BASE_CONFIG}\` (no_cls, local_pca teacher, val_topo checkpoint)
- \`save_every=1\` — per-epoch checkpoints kept
- Train checkpoint: **val_topo_loss minimum** (\`best_model.pth\`)
- Objective: mean **topo W-Dist** on val (learned ellipses vs local_pca teacher PD)
- Study: \`${STUDY_NAME}\`

Launch:
\`bash experiments/run_tune_local_pca_parallel.sh ${N_WORKERS} ${N_TRIALS} ${TUNE_EPOCHS} ${BACKEND}\`
EOF

echo "Tune v3 local_pca: ${N_WORKERS} workers, ${N_TRIALS} trials, ${TUNE_EPOCHS}ep, backend=${BACKEND}"
echo "OUT_BASE=${OUT_BASE}"
echo "BASE_CONFIG=${BASE_CONFIG}"
echo "STUDY=${STUDY_NAME}"

pids=()
for i in $(seq 0 $((N_WORKERS - 1))); do
  SEED=$((42 + i))
  OMP_NUM_THREADS=${THREADS_PER_WORKER} \
  MKL_NUM_THREADS=${THREADS_PER_WORKER} \
  OPENBLAS_NUM_THREADS=${THREADS_PER_WORKER} \
  nohup uv run python -u experiments/tune_elongate_wdist.py \
    --base-config "${BASE_CONFIG}" \
    --n-trials "${N_TRIALS}" \
    --n-startup-trials 12 \
    --tune-epochs "${TUNE_EPOCHS}" \
    --backend "${BACKEND}" \
    --out-base "${OUT_BASE}" \
    --storage "${STORAGE}" \
    --study-name "${STUDY_NAME}" \
    --seed "${SEED}" \
    > "${OUT_BASE}/worker_${i}.log" 2>&1 &
  pids+=($!)
  echo "  worker ${i}: PID $!, seed ${SEED}, log ${OUT_BASE}/worker_${i}.log"
  sleep 1
done

echo "Waiting for ${N_WORKERS} workers..."
for pid in "${pids[@]}"; do
  wait "${pid}" || echo "  worker PID ${pid} exited non-zero"
done

echo "Writing best params..."
uv run python -u experiments/tune_elongate_wdist.py \
  --base-config "${BASE_CONFIG}" \
  --n-trials "${N_TRIALS}" \
  --tune-epochs "${TUNE_EPOCHS}" \
  --backend "${BACKEND}" \
  --out-base "${OUT_BASE}" \
  --storage "${STORAGE}" \
  --study-name "${STUDY_NAME}" \
  --write-best

echo "Done: ${OUT_BASE}/best_elongate_wdist_${BACKEND}.json"
echo "Apply to configs: uv run python experiments/apply_tune_best_to_configs.py --json ${OUT_BASE}/best_elongate_wdist_${BACKEND}.json"
