#!/usr/bin/env bash
# Detached screen launcher for long jobs (SSH logout / terminal close safe).
# Note: machine reboot or power-off still stops the job.
#
# Usage:
#   bash experiments/launch_detached_screen.sh SESSION_NAME LOG_FILE SCRIPT.sh [args...]
#
# Example:
#   bash experiments/launch_detached_screen.sh pwr30 \
#     outputs/supervised/pwr30/driver.log \
#     experiments/run_teacher_local_pca_power_30ep_multiseed.sh wdist

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 SESSION_NAME LOG_FILE SCRIPT [args...]" >&2
  exit 1
fi

SESSION="$1"
LOG="$2"
shift 2

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$(dirname "${ROOT}/${LOG}")"

if screen -ls | grep -q "[[:space:]]\+[0-9]*\.${SESSION}[[:space:]]"; then
  echo "screen session already exists: ${SESSION}" >&2
  screen -ls | grep "${SESSION}" || true
  exit 1
fi

export PATH="${HOME}/.local/bin:${PATH}"

# nohup inside screen: survives SIGHUP if the shell layer exits unexpectedly.
screen -dmS "${SESSION}" bash -lc "
  cd '${ROOT}'
  export PATH='${HOME}/.local/bin:'\${PATH}
  exec nohup $(printf '%q ' "$@") >> '${ROOT}/${LOG}' 2>&1
"

echo "started screen session: ${SESSION}"
echo "log: ${ROOT}/${LOG}"
echo "reattach: screen -r ${SESSION}"
echo "detach:   Ctrl+A then D"
