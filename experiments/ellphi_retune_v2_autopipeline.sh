#!/usr/bin/env bash
# Autopipeline: (1) ellphi re-tune v2 → (2) update configs → (3) full120 30ep + eval
#
# Detached:
#   bash experiments/launch_detached_screen.sh ellphi_v2_pipeline \
#     outputs/tune_elongate_ellphi_v2/autopipeline.log \
#     experiments/ellphi_retune_v2_autopipeline.sh

set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

N_WORKERS="${N_WORKERS:-8}"
N_TRIALS="${N_TRIALS:-50}"
TUNE_EPOCHS="${TUNE_EPOCHS:-20}"
BACKEND="${BACKEND:-ellphi}"
OUT_BASE="${OUT_BASE:-outputs/tune_elongate_ellphi_v2}"
BEST_JSON="${OUT_BASE}/best_elongate_wdist_${BACKEND}.json"
POLL_SEC="${POLL_SEC:-120}"
LOG_DIR="${OUT_BASE}"

log() { echo "[$(date -Iseconds)] $*"; }

mkdir -p "${LOG_DIR}"

log "=== STEP 1/3: re-tune v2 (${N_WORKERS} workers, ${N_TRIALS} trials, ${TUNE_EPOCHS}ep) ==="
bash experiments/run_retune_parallel.sh "${N_WORKERS}" "${N_TRIALS}" "${TUNE_EPOCHS}" "${BACKEND}"

if [[ ! -f "${BEST_JSON}" ]]; then
  log "ERROR: missing ${BEST_JSON} after tune"
  exit 1
fi
log "tune complete: ${BEST_JSON}"

log "=== STEP 2/3: apply best params to configs ==="
uv run python experiments/apply_tune_best_to_configs.py --json "${BEST_JSON}"

log "=== STEP 3/3: full120 30ep train + evaluate ==="
TUNE_JSON="${BEST_JSON}" bash experiments/run_ellphi_tuned_v2_full120_30ep.sh

log "=== PIPELINE COMPLETE ==="
