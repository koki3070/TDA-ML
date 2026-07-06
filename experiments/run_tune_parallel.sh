#!/usr/bin/env bash
# Parallel Optuna tuning launcher for elongate (W-Dist objective).
#
# NOTE: For re-tunes aligned with train_topo checkpoint selection + save_every=1,
# use experiments/run_retune_parallel.sh instead (writes to outputs/*_v2).
#
# Spawns N worker processes that share ONE SQLite study, so trials are sampled
# cooperatively (TPE sees all workers' results). The study stops once TOTAL
# completed trials reaches --n-trials (enforced per-worker via MaxTrialsCallback).
#
# Usage:
#   bash experiments/run_tune_parallel.sh [N_WORKERS] [N_TRIALS] [TUNE_EPOCHS] [BACKEND] [OUT_BASE]
# Defaults: 8 workers, 50 trials, 20 epochs, mahalanobis, outputs/tune_elongate.
#
# Examples:
#   bash experiments/run_tune_parallel.sh 8 50 20 mahalanobis
#   bash experiments/run_tune_parallel.sh 8 50 20 ellphi outputs/tune_elongate_ellphi

set -euo pipefail

N_WORKERS="${1:-8}"
N_TRIALS="${2:-50}"
TUNE_EPOCHS="${3:-20}"
BACKEND="${4:-mahalanobis}"
OUT_BASE="${5:-outputs/tune_elongate}"
STORAGE="sqlite:///${OUT_BASE}/study.db"
STUDY_NAME="elongate_wdist_4d_${BACKEND}"
THREADS_PER_WORKER=12

mkdir -p "${OUT_BASE}"

echo "Launching ${N_WORKERS} workers, study-wide budget=${N_TRIALS} trials, ${TUNE_EPOCHS}ep, backend=${BACKEND}"
echo "Storage: ${STORAGE}"

pids=()
for i in $(seq 0 $((N_WORKERS - 1))); do
  SEED=$((42 + i))  # different sampler seed per worker to decorrelate startup
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

echo "Waiting for ${N_WORKERS} workers to finish..."
for pid in "${pids[@]}"; do
  wait "${pid}" || echo "  worker PID ${pid} exited non-zero"
done

echo "All workers done. Writing best params..."
uv run python -u experiments/tune_elongate_wdist.py \
  --base-config elongate_n100 \
  --n-trials "${N_TRIALS}" \
  --tune-epochs "${TUNE_EPOCHS}" \
  --backend "${BACKEND}" \
  --out-base "${OUT_BASE}" \
  --storage "${STORAGE}" \
  --study-name "${STUDY_NAME}" \
  --write-best

echo "Done. See ${OUT_BASE}/best_elongate_wdist_${BACKEND}.json"
