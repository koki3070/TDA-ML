#!/usr/bin/env bash
# Wait for 12ep ellphi calibration (seed 42), then launch 5-seed paper main run (fixed 12ep).
#
# Usage (background):
#   nohup ./experiments/ellphi_postfix_autopipeline.sh >> outputs/ellphi_postfix_calib/autopipeline.log 2>&1 &
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CALIB_OUT="${CALIB_OUT:-outputs/ellphi_postfix_calib}"
PILOT_LOG="${CALIB_OUT}/driver_12ep_seed42.log"
STATE_JSON="${CALIB_OUT}/autopipeline_state.json"
DECISION_JSON="${CALIB_OUT}/epoch_decision.json"
POLL_SEC="${POLL_SEC:-300}"
SEEDS=(42 123 456 789 1024)
MAIN_EPOCHS="${MAIN_EPOCHS:-12}"
MAIN_OUT="${MAIN_OUT:-outputs/paper_reproduce_1week_tuned}"
MAIN_CONFIG="${MAIN_CONFIG:-reproduce}"

log() { echo "[$(date -Iseconds)] $*"; }

write_state() {
  local phase="$1"
  local detail="${2:-}"
  python3 - "$phase" "$detail" "$STATE_JSON" <<'PY'
import json, sys
from datetime import datetime, timezone
phase, detail, path = sys.argv[1:4]
state = {}
try:
    with open(path) as f:
        state = json.load(f)
except FileNotFoundError:
    pass
state.update({
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "phase": phase,
    "detail": detail,
})
with open(path, "w") as f:
    json.dump(state, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
}

pilot_running() {
  pgrep -f "run_backend_multiseed.py.*out-base ${CALIB_OUT}" >/dev/null 2>&1 \
    || pgrep -f "run_backend_multiseed.py.*${CALIB_OUT}" >/dev/null 2>&1
}

pilot_done() {
  [[ -f "$PILOT_LOG" ]] && grep -q '^\[DONE\]' "$PILOT_LOG"
}

pilot_failed() {
  if pilot_running; then
    return 1
  fi
  if pilot_done; then
    return 1
  fi
  if [[ -f "$PILOT_LOG" ]] && grep -q 'Traceback' "$PILOT_LOG"; then
    return 0
  fi
  # No process, no DONE — treat as failure if log exists and has START
  if [[ -f "$PILOT_LOG" ]] && grep -q '^\[START\]' "$PILOT_LOG"; then
    return 0
  fi
  return 1
}

wait_for_pilot() {
  write_state "waiting_pilot" "polling every ${POLL_SEC}s"
  log "Waiting for 12ep calibration in ${CALIB_OUT} ..."
  while true; do
    if pilot_done; then
      log "Pilot completed ([DONE] in log)."
      write_state "pilot_completed"
      return 0
    fi
    if pilot_failed; then
      log "ERROR: Pilot exited without [DONE]. See ${PILOT_LOG}"
      write_state "pilot_failed" "check driver log"
      exit 1
    fi
    if pilot_running; then
      progress="$(python3 - "$PILOT_LOG" <<'PY' || true
import re, sys
t = open(sys.argv[1]).read()
m = list(re.finditer(r'Epoch (\d+):.*?(\d+)/71', t))
if m:
    print(f"epoch {m[-1].group(1)} batch {m[-1].group(2)}/71")
else:
    n = len(re.findall(r'Epoch \d+: Val MCC=', t))
    if n:
        print(f"completed_epochs={n}")
    else:
        print("epoch 1 starting")
PY
)"
      log "Pilot still running (${progress})"
      evaluate_pilot_if_done
    else
      log "Pilot process not found yet; waiting ..."
    fi
    sleep "$POLL_SEC"
  done
}

decide_epochs() {
  python3 - "$CALIB_OUT" "$DECISION_JSON" <<'PY'
import csv
import json
import sys
from pathlib import Path

calib = Path(sys.argv[1])
out_path = Path(sys.argv[2])

# Find metrics from progress_summary or latest run dir
metrics_path = None
progress = calib / "progress_summary.csv"
if progress.exists():
    rows = list(csv.DictReader(progress.open()))
    if rows:
        run_dir = Path(rows[-1]["run_dir"])
        candidate = run_dir / "logs" / "metrics.csv"
        if candidate.exists():
            metrics_path = candidate

if metrics_path is None:
    candidates = sorted(
        list(calib.glob("eph_s42_*/logs/metrics.csv"))
        + list(calib.glob("backend_ellphi_seed42_*/logs/metrics.csv"))
    )
    if candidates:
        metrics_path = candidates[-1]

if metrics_path is None or not metrics_path.exists():
    raise SystemExit("metrics.csv not found for pilot")

rows = list(csv.DictReader(metrics_path.open()))
if len(rows) < 3:
    raise SystemExit(f"Too few epochs in {metrics_path} ({len(rows)} rows)")

def mcc_at(epoch: int) -> float | None:
    for r in rows:
        if int(r["epoch"]) == epoch:
            return float(r["val_mcc"])
    return None

best_row = max(rows, key=lambda r: float(r["val_mcc"]))
best_ep = int(best_row["epoch"])
best_mcc = float(best_row["val_mcc"])

m10 = mcc_at(10)
m12 = mcc_at(12) if mcc_at(12) is not None else float(rows[-1]["val_mcc"])
last_ep = int(rows[-1]["epoch"])

gain_10_12 = None
if m10 is not None and m12 is not None:
    gain_10_12 = m12 - m10

use_20 = False
reason = []
if gain_10_12 is not None and gain_10_12 < 0.002 and best_ep <= 11:
    if m10 is not None and m10 >= 0.99 * best_mcc:
        use_20 = True
        reason.append("gain_10_12<0.002, best<=11, ep10>=99%best")
if last_ep < 10:
    use_20 = False
    reason = [f"only {last_ep} epochs logged; default 30"]

chosen = 20 if use_20 else 30
decision = {
    "chosen_epochs": chosen,
    "criteria": {
        "gain_10_12": gain_10_12,
        "best_epoch": best_ep,
        "best_val_mcc": best_mcc,
        "val_mcc_ep10": m10,
        "val_mcc_ep12_or_last": m12,
        "last_logged_epoch": last_ep,
    },
    "reason": reason or ["default 30 (still improving or criteria not met)"],
    "metrics_path": str(metrics_path),
}
out_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
print(chosen)
PY
}

launch_main() {
  local epochs="$1"
  local out_base="${MAIN_OUT}"
  mkdir -p "$out_base"

  if [[ -f "${out_base}/progress_summary.csv" ]]; then
  completed="$(python3 - "$out_base/progress_summary.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(sum(1 for r in rows if r.get("backend") == "ellphi"))
PY
)"
    if [[ "$completed" -ge 5 ]]; then
      log "Main run already has ${completed} ellphi rows in ${out_base}; skipping."
      write_state "main_already_complete" "$out_base"
      evaluate_all_runs "$out_base"
      return 0
    fi
  fi

  if pgrep -f "run_backend_multiseed.py.*out-base ${out_base}" >/dev/null 2>&1; then
    log "Main run already in progress under ${out_base}; skipping duplicate launch."
    write_state "main_running" "$out_base"
    return 0
  fi

  log "Launching main: ${epochs}ep × 5 seeds → ${out_base}"
  write_state "main_starting" "${epochs}ep ${out_base}"

  uv run python experiments/run_backend_multiseed.py \
    --base-config "$MAIN_CONFIG" \
    --epochs "$epochs" \
    --seeds "${SEEDS[@]}" \
    --backends ellphi \
    --out-base "$out_base" \
    2>&1 | tee -a "${out_base}/driver_main.log"

  write_state "main_completed" "$out_base"
  log "Main run finished. See ${out_base}/backend_stats.csv"
  evaluate_all_runs "$out_base"
}

evaluate_run_paper_protocol() {
  local run_dir="$1"
  local base_cfg="${2:-reproduce}"
  log "Paper DBSCAN eval: ${run_dir} (config=${base_cfg})"
  uv run python experiments/evaluate_paper_protocol.py \
    --run-dir "$run_dir" \
    --base-config "$base_cfg" \
    --split val
  uv run python experiments/evaluate_paper_protocol.py \
    --run-dir "$run_dir" \
    --base-config "$base_cfg" \
    --split test
}

evaluate_all_runs() {
  local out_base="$1"
  local base_cfg="${2:-$MAIN_CONFIG}"
  local progress="${out_base}/progress_summary.csv"
  [[ -f "$progress" ]] || return 0
  python3 - "$progress" <<'PY' | while read -r rd; do
import csv, sys
for row in csv.DictReader(open(sys.argv[1])):
    if row.get("backend") == "ellphi":
        print(row["run_dir"])
PY
    evaluate_run_paper_protocol "$rd" "$base_cfg"
  done
}

evaluate_pilot_if_done() {
  if ! pilot_done; then
    return 0
  fi
  evaluate_all_runs "$CALIB_OUT" "reproduce"
}

main() {
  log "=== ellphi postfix autopipeline start (repo=${REPO_ROOT}) ==="
  if ! pilot_done; then
    wait_for_pilot
  else
    log "Pilot already complete; skipping wait."
    write_state "pilot_completed" "pre-existing"
  fi

  evaluate_pilot_if_done

  epochs="$MAIN_EPOCHS"
  log "Paper main: fixed ${epochs}ep → ${MAIN_OUT} (config=${MAIN_CONFIG})"
  python3 - "$epochs" "$DECISION_JSON" <<'PY'
import json, sys
from datetime import datetime, timezone
epochs, path = sys.argv[1:3]
decision = {
    "chosen_epochs": int(epochs),
    "reason": ["fixed 12ep for paper reproduce_1week_tuned (override MAIN_EPOCHS to change)"],
    "decided_utc": datetime.now(timezone.utc).isoformat(),
}
with open(path, "w") as f:
    json.dump(decision, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
  write_state "epoch_decided" "${epochs}ep ${MAIN_OUT}"

  launch_main "$epochs"
  write_state "pipeline_finished" "${epochs}ep ${MAIN_OUT}"
  log "=== autopipeline finished ==="
}

main "$@"
