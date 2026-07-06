#!/usr/bin/env bash
# Re-tune launcher (v2 protocol): save_every=1 + train_topo best checkpoint → val W-Dist.
#
# Usage:
#   bash experiments/run_retune_parallel.sh [N_WORKERS] [N_TRIALS] [TUNE_EPOCHS] [BACKEND]
#
# Defaults: 8 workers, 50 trials, 20 epochs, ellphi.
#
# Examples:
#   bash experiments/run_retune_parallel.sh 8 50 20 ellphi
#   bash experiments/run_retune_parallel.sh 8 50 20 mahalanobis
#
# Detached (SSH-safe):
#   bash experiments/launch_detached_screen.sh tune_ellphi_v2 \
#     outputs/tune_elongate_ellphi_v2/launcher.log \
#     experiments/run_retune_parallel.sh 8 50 20 ellphi

set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

N_WORKERS="${1:-8}"
N_TRIALS="${2:-50}"
TUNE_EPOCHS="${3:-20}"
BACKEND="${4:-ellphi}"

if [[ "${BACKEND}" == "mahalanobis" ]]; then
  OUT_BASE="outputs/tune_elongate_v2"
else
  OUT_BASE="outputs/tune_elongate_${BACKEND}_v2"
fi

STUDY_NAME="elongate_topo_ckpt_wdist_${BACKEND}_v2"
STORAGE="sqlite:///${OUT_BASE}/study.db"
THREADS_PER_WORKER=12

mkdir -p "${OUT_BASE}"
cat > "${OUT_BASE}/PURPOSE.md" <<EOF
# Optuna re-tune v2 (${BACKEND})

Protocol (aligned with 30ep paper eval):
- \`save_every=1\` — checkpoint every epoch
- Objective: val W-Dist at **train_topo_loss minimum** saved checkpoint
- Base config: \`elongate_n100\` (w_class=1, 20ep proxy)
- Study: \`${STUDY_NAME}\`

Legacy v1: \`outputs/tune_elongate${BACKEND:+_${BACKEND}}\` (final_model @ ep20, no per-epoch ckpt)

Launch: \`bash experiments/run_retune_parallel.sh ${N_WORKERS} ${N_TRIALS} ${TUNE_EPOCHS} ${BACKEND}\`
EOF

echo "Re-tune v2: ${N_WORKERS} workers, ${N_TRIALS} trials, ${TUNE_EPOCHS}ep, backend=${BACKEND}"
echo "OUT_BASE=${OUT_BASE}"
echo "STUDY=${STUDY_NAME}"
echo "Storage: ${STORAGE}"

pids=()
for i in $(seq 0 $((N_WORKERS - 1))); do
  SEED=$((42 + i))
  OMP_NUM_THREADS=${THREADS_PER_WORKER} \
  MKL_NUM_THREADS=${THREADS_PER_WORKER} \
  OPENBLAS_NUM_THREADS=${THREADS_PER_WORKER} \
  nohup uv run python -u experiments/tune_elongate_wdist.py \
    --base-config elongate_n100 \
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
  --base-config elongate_n100 \
  --n-trials "${N_TRIALS}" \
  --tune-epochs "${TUNE_EPOCHS}" \
  --backend "${BACKEND}" \
  --out-base "${OUT_BASE}" \
  --storage "${STORAGE}" \
  --study-name "${STUDY_NAME}" \
  --write-best

echo "Done: ${OUT_BASE}/best_elongate_wdist_${BACKEND}.json"
