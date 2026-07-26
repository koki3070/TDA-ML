#!/usr/bin/env bash
# Run both tuning objectives for power + local_pca + ellphi stack:
#   1. val topo W-Dist min  (hyperparams for geometric PD fit)
#   2. val DBSCAN MCC max   (hyperparams for clustering; maha DBSCAN distance)
#
# Usage:
#   bash experiments/run_tune_local_pca_power_objectives.sh [MODE] [N_WORKERS] [N_TRIALS] [TUNE_EPOCHS]
#
# MODE: wdist | mcc | both (default both). Runs sequentially when both.
#
# Detached (both studies, ~8–10h total with 4 workers each):
#   bash experiments/launch_detached_screen.sh tune_pwr_obj \
#     outputs/tune/pwr_objectives/launcher.log \
#     experiments/run_tune_local_pca_power_objectives.sh both 4 24 20

set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

MODE="${1:-both}"
N_WORKERS="${2:-4}"
N_TRIALS="${3:-24}"
TUNE_EPOCHS="${4:-20}"
BACKEND="ellphi"
DBSCAN_BACKEND="mahalanobis"

LOG_ROOT="${LOG_ROOT:-outputs/tune/pwr_objectives}"
mkdir -p "${LOG_ROOT}"

run_wdist() {
  echo "=== W-Dist objective tune ==="
  OUT_BASE="${OUT_BASE:-outputs/tune/pwr_wdist}" \
    bash experiments/run_tune_local_pca_power_wdist_parallel.sh \
    "${N_WORKERS}" "${N_TRIALS}" "${TUNE_EPOCHS}" "${BACKEND}" \
    2>&1 | tee "${LOG_ROOT}/wdist_launch.log"
}

run_mcc() {
  echo "=== MCC objective tune (DBSCAN=${DBSCAN_BACKEND}) ==="
  OUT_BASE="${OUT_BASE:-outputs/tune/pwr_mcc_dbscan_${DBSCAN_BACKEND}}" \
    bash experiments/run_tune_local_pca_power_mcc_parallel.sh \
    "${N_WORKERS}" "${N_TRIALS}" "${TUNE_EPOCHS}" "${BACKEND}" "${DBSCAN_BACKEND}" \
    2>&1 | tee "${LOG_ROOT}/mcc_launch.log"
}

case "${MODE}" in
  wdist) run_wdist ;;
  mcc) run_mcc ;;
  both)
    run_wdist
    run_mcc
    ;;
  *)
    echo "Unknown MODE=${MODE} (use wdist, mcc, or both)" >&2
    exit 1
    ;;
esac

echo "Objectives complete (mode=${MODE})."
echo "  W-Dist best: outputs/tune/pwr_wdist/best_elongate_wdist_${BACKEND}.json"
echo "  MCC best:    outputs/tune/pwr_mcc_dbscan_${DBSCAN_BACKEND}/best_elongate_mcc_${BACKEND}_dbscan_${DBSCAN_BACKEND}.json"
