#!/usr/bin/env bash
# 30ep proposed runs on paper seeds (fixed tune weights from seed-42 Optuna).
#
# Protocol: tune once on seed 42 → apply same w_*, lr to all 5 data seeds.
#
# Usage:
#   bash experiments/run_teacher_local_pca_power_30ep_multiseed.sh [MODE] [SEEDS...]
#
# MODE: wdist | mcc | both (default both)
# SEEDS: default 42 123 456 789 1024
#
# Detached:
#   bash experiments/launch_detached_screen.sh pwr30_ms \
#     outputs/supervised/0710_pwr30_multiseed/driver.log \
#     experiments/run_teacher_local_pca_power_30ep_multiseed.sh both

set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

MODE="${1:-both}"
shift || true
if [[ $# -gt 0 && "${1:-}" =~ ^[0-9]+$ ]]; then
  SEEDS=("$@")
else
  SEEDS=(42 123 456 789 1024)
fi

WDIST_JSON="${WDIST_JSON:-outputs/tune/0709_pwr_wdist/best_elongate_wdist_ellphi.json}"
MCC_JSON="${MCC_JSON:-outputs/tune/0709_pwr_mcc_dbscan_mahalanobis/best_elongate_mcc_ellphi_dbscan_mahalanobis.json}"
WDIST_OUT="${WDIST_OUT:-outputs/supervised/0710_pwr30_wdist}"
MCC_OUT="${MCC_OUT:-outputs/supervised/0710_pwr30_mcc_maha}"
LOG_ROOT="${LOG_ROOT:-outputs/supervised/0710_pwr30_multiseed}"
EPOCHS="${EPOCHS:-30}"
DBSCAN_BACKEND="${DBSCAN_BACKEND:-mahalanobis}"

mkdir -p "${LOG_ROOT}"

_metrics_done() {
  local out_base="$1"
  local seed="$2"
  local tag="$3"
  local f
  for f in "${out_base}"/pwr_s"${seed}"_*/logs/paper_metrics_test_"${tag}".json; do
    if [[ -f "${f}" ]]; then
      return 0
    fi
  done
  return 1
}

_run_seed() {
  local label="$1"
  local tune_json="$2"
  local out_base="$3"
  local tag="$4"
  local seed="$5"
  local log_file="${LOG_ROOT}/${label}_s${seed}.log"

  if _metrics_done "${out_base}" "${seed}" "${tag}"; then
    echo "[skip] ${label} seed=${seed} (paper_metrics_test already exists)"
    return 0
  fi

  echo "=== ${label} seed=${seed} ==="
  uv run python -u experiments/run_teacher_local_pca_power_30ep.py \
    --epochs "${EPOCHS}" \
    --seed "${seed}" \
    --out-base "${out_base}" \
    --tune-json "${tune_json}" \
    --tag "${tag}" \
    --dbscan-backend "${DBSCAN_BACKEND}" \
    2>&1 | tee -a "${log_file}"
}

_run_method() {
  local label="$1"
  local tune_json="$2"
  local out_base="$3"
  local tag="$4"
  mkdir -p "${out_base}"
  for seed in "${SEEDS[@]}"; do
    _run_seed "${label}" "${tune_json}" "${out_base}" "${tag}" "${seed}"
  done
}

case "${MODE}" in
  wdist)
    _run_method "wdist" "${WDIST_JSON}" "${WDIST_OUT}" "power_wdist_valtopo_paper_eval"
    ;;
  mcc)
    _run_method "mcc" "${MCC_JSON}" "${MCC_OUT}" "power_mcc_valtopo_paper_eval"
    ;;
  both)
    _run_method "wdist" "${WDIST_JSON}" "${WDIST_OUT}" "power_wdist_valtopo_paper_eval"
    _run_method "mcc" "${MCC_JSON}" "${MCC_OUT}" "power_mcc_valtopo_paper_eval"
    ;;
  *)
    echo "Unknown MODE=${MODE} (use wdist, mcc, or both)" >&2
    exit 1
    ;;
esac

echo "Aggregating..."
AGG_ARGS=(
  --wdist-out "${WDIST_OUT}"
  --mcc-out "${MCC_OUT}"
  --out-dir "${LOG_ROOT}"
  --seeds "${SEEDS[@]}"
)
case "${MODE}" in
  wdist) AGG_ARGS+=(--methods wdist) ;;
  mcc) AGG_ARGS+=(--methods mcc) ;;
esac
# Strict: missing seeds or aggregation failure must fail this driver (no || true).
uv run python -u experiments/aggregate_power_30ep_multiseed.py "${AGG_ARGS[@]}"

echo "Done. Logs: ${LOG_ROOT}/"
