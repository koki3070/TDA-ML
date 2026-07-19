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
# Detached (SSH/logout safe; machine reboot still stops the job):
#   N_WORKERS=4 THREADS_PER_WORKER=4 \
#   bash experiments/launch_detached_screen.sh pwr30_ms \
#     outputs/supervised/pwr30_multiseed/driver.log \
#     experiments/run_teacher_local_pca_power_30ep_multiseed.sh both
#
# Parallelism: N_WORKERS seeds per objective wave; THREADS_PER_WORKER per process
# (bench: raising OMP past 4 does not speed ellphi topo; parallel seeds do).

set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"
# Per-process threads (bench 2026-07-12: OMP 4/12/16 → identical ~1.5 s/batch; ellphi topo
# is sample-serial Python + C++). Do not raise unless re-benchmarked.
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS_PER_WORKER}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${THREADS_PER_WORKER}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${THREADS_PER_WORKER}}"
# Parallel seeds within each objective (wdist / mcc). Each worker uses THREADS_PER_WORKER.
N_WORKERS="${N_WORKERS:-4}"

MODE="${1:-both}"
shift || true
if [[ $# -gt 0 && "${1:-}" =~ ^[0-9]+$ ]]; then
  SEEDS=("$@")
else
  SEEDS=(42 123 456 789 1024)
fi

WDIST_JSON="${WDIST_JSON:-outputs/tune/pwr_wdist/best_elongate_wdist_ellphi.json}"
MCC_JSON="${MCC_JSON:-outputs/tune/pwr_mcc_dbscan_mahalanobis/best_elongate_mcc_ellphi_dbscan_mahalanobis.json}"
WDIST_OUT="${WDIST_OUT:-outputs/supervised/pwr30_wdist}"
MCC_OUT="${MCC_OUT:-outputs/supervised/pwr30_mcc_maha}"
LOG_ROOT="${LOG_ROOT:-outputs/supervised/pwr30_multiseed}"
EPOCHS="${EPOCHS:-30}"
DBSCAN_BACKEND="${DBSCAN_BACKEND:-mahalanobis}"
BASE_CONFIG="${BASE_CONFIG:-elongate_n100_no_cls_full120_teacher_local_pca}"
# Declared paper contract variant the tune JSONs must match (elongate | elongate_barrier).
ANISO_VARIANT="${ANISO_VARIANT:-elongate}"

mkdir -p "${LOG_ROOT}"

_metrics_done() {
  # Exit 0 = fresh skip; 1 = missing (run); other = stale/ambiguous hard-fail.
  local out_base="$1"
  local seed="$2"
  local tag="$3"
  local tune_json="$4"
  local rc=0
  uv run python experiments/power_30ep_freshness.py \
    --out-base "${out_base}" \
    --seed "${seed}" \
    --tag "${tag}" \
    --tune-json "${tune_json}" || rc=$?
  if [[ "${rc}" -eq 0 ]]; then
    return 0
  fi
  if [[ "${rc}" -eq 1 ]]; then
    return 1
  fi
  echo "error: freshness check failed for seed=${seed} (rc=${rc})" >&2
  exit "${rc}"
}

_run_seed() {
  local label="$1"
  local tune_json="$2"
  local out_base="$3"
  local tag="$4"
  local seed="$5"
  local log_file="${LOG_ROOT}/${label}_s${seed}.log"

  if _metrics_done "${out_base}" "${seed}" "${tag}" "${tune_json}"; then
    echo "[skip] ${label} seed=${seed} (fresh paper_metrics_test matches tune/revision)"
    return 0
  fi

  echo "=== ${label} seed=${seed} (threads=${THREADS_PER_WORKER}) ==="
  OMP_NUM_THREADS="${THREADS_PER_WORKER}" \
  MKL_NUM_THREADS="${THREADS_PER_WORKER}" \
  OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}" \
  uv run python -u experiments/run_teacher_local_pca_power_30ep.py \
    --base-config "${BASE_CONFIG}" \
    --epochs "${EPOCHS}" \
    --seed "${seed}" \
    --out-base "${out_base}" \
    --tune-json "${tune_json}" \
    --tag "${tag}" \
    --dbscan-backend "${DBSCAN_BACKEND}" \
    >> "${log_file}" 2>&1
}

_run_seed_batch() {
  local label="$1"
  local tune_json="$2"
  local out_base="$3"
  local tag="$4"
  shift 4
  local seeds=("$@")
  local pids=()
  local seed pid

  for seed in "${seeds[@]}"; do
    if _metrics_done "${out_base}" "${seed}" "${tag}" "${tune_json}"; then
      echo "[skip] ${label} seed=${seed} (fresh paper_metrics_test matches tune/revision)"
      continue
    fi
    _run_seed "${label}" "${tune_json}" "${out_base}" "${tag}" "${seed}" &
    pids+=($!)
    echo "  launched ${label} seed=${seed} PID=$!"
    sleep 1
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      echo "error: worker PID ${pid} failed" >&2
      exit 1
    fi
  done
}

_run_method() {
  local label="$1"
  local tune_json="$2"
  local out_base="$3"
  local tag="$4"
  mkdir -p "${out_base}"

  local pending=()
  local seed
  for seed in "${SEEDS[@]}"; do
    if _metrics_done "${out_base}" "${seed}" "${tag}" "${tune_json}"; then
      echo "[skip] ${label} seed=${seed} (fresh paper_metrics_test matches tune/revision)"
    else
      pending+=("${seed}")
    fi
  done

  if ((${#pending[@]} == 0)); then
    echo "[done] ${label}: all seeds complete"
    return 0
  fi

  echo "${label}: ${#pending[@]} seed(s) pending, N_WORKERS=${N_WORKERS}, threads=${THREADS_PER_WORKER}"
  local batch=()
  for seed in "${pending[@]}"; do
    batch+=("${seed}")
    if ((${#batch[@]} >= N_WORKERS)); then
      _run_seed_batch "${label}" "${tune_json}" "${out_base}" "${tag}" "${batch[@]}"
      batch=()
    fi
  done
  if ((${#batch[@]} > 0)); then
    _run_seed_batch "${label}" "${tune_json}" "${out_base}" "${tag}" "${batch[@]}"
  fi
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
  --wdist-tune-json "${WDIST_JSON}"
  --mcc-tune-json "${MCC_JSON}"
  --out-dir "${LOG_ROOT}"
  --seeds "${SEEDS[@]}"
  --aniso-variant "${ANISO_VARIANT}"
)
case "${MODE}" in
  wdist) AGG_ARGS+=(--methods wdist) ;;
  mcc) AGG_ARGS+=(--methods mcc) ;;
esac
# Strict: missing seeds or aggregation failure must fail this driver (no || true).
uv run python -u experiments/aggregate_power_30ep_multiseed.py "${AGG_ARGS[@]}"

echo "Done. Logs: ${LOG_ROOT}/"
