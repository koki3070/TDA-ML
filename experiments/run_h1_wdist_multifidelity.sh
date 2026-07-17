#!/usr/bin/env bash
# H1-only W-Dist tune:
#   stage 1: 16 proxy trials x 5 epochs
#   stage 2: top 4 proxy conditions x 20 epochs

set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

BASE_CONFIG="${BASE_CONFIG:-elongate_n100_no_cls_tune_local_pca_ellphi_power_h1}"
ROOT_OUT="${ROOT_OUT:-outputs/tune/0716_pwr_wdist_h1_multifidelity}"
STAGE1_OUT="${ROOT_OUT}/stage1_proxy5"
STAGE2_OUT="${ROOT_OUT}/stage2_top4_full20"
STAGE1_STUDY="h1_wdist_proxy5_16"
STAGE2_STUDY="h1_wdist_top4_full20"
STAGE1_STORAGE="sqlite:///${STAGE1_OUT}/study.db"
STAGE2_STORAGE="sqlite:///${STAGE2_OUT}/study.db"

record_plan_status() {
  local status="$1"
  local reason="$2"
  local plan="${ROOT_OUT}/MULTIFIDELITY_PLAN.json"
  [[ -f "${plan}" ]] || return 0
  uv run python - "${plan}" "${status}" "${reason}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["run_status"] = sys.argv[2]
payload["status_reason"] = sys.argv[3]
payload["status_recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

on_error() {
  local code=$?
  trap - ERR
  record_plan_status "failed" "pipeline command failed with exit code ${code}"
  exit "${code}"
}

on_interrupt() {
  trap - INT TERM
  record_plan_status "skipped" "pipeline interrupted intentionally"
  exit 130
}

trap on_error ERR
trap on_interrupt INT TERM

if [[ -e "${ROOT_OUT}" ]]; then
  echo "output already exists: ${ROOT_OUT}" >&2
  exit 1
fi

mkdir -p "${ROOT_OUT}"
cat > "${ROOT_OUT}/MULTIFIDELITY_PLAN.json" <<EOF
{
  "run_status": "running",
  "base_config": "${BASE_CONFIG}",
  "homology_dimensions": [1],
  "distance_backend": "ellphi",
  "max_points_policy": "full configured cloud",
  "stage1": {"trials": 16, "epochs": 5, "workers": 8},
  "stage2": {"selected_top_k": 4, "epochs": 20, "workers": 4},
  "fallbacks": []
}
EOF

BASE_CONFIG="${BASE_CONFIG}" \
OUT_BASE="${STAGE1_OUT}" \
STUDY_NAME="${STAGE1_STUDY}" \
THREADS_PER_WORKER="${STAGE1_THREADS:-12}" \
TRIALS_PER_WORKER=2 \
bash experiments/run_tune_local_pca_power_wdist_parallel.sh 8 16 5 ellphi

uv run python -u experiments/enqueue_top_optuna_trials.py \
  --source-storage "${STAGE1_STORAGE}" \
  --source-study "${STAGE1_STUDY}" \
  --target-storage "${STAGE2_STORAGE}" \
  --target-study "${STAGE2_STUDY}" \
  --top-k 4 \
  --out-dir "${STAGE2_OUT}"

BASE_CONFIG="${BASE_CONFIG}" \
OUT_BASE="${STAGE2_OUT}" \
STUDY_NAME="${STAGE2_STUDY}" \
THREADS_PER_WORKER="${STAGE2_THREADS:-24}" \
TRIALS_PER_WORKER=1 \
bash experiments/run_tune_local_pca_power_wdist_parallel.sh 4 4 20 ellphi

uv run python - "${ROOT_OUT}/MULTIFIDELITY_PLAN.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["run_status"] = "completed"
payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
payload["best_result"] = "stage2_top4_full20/best_elongate_wdist_ellphi.json"
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "Done: ${STAGE2_OUT}/best_elongate_wdist_ellphi.json"
