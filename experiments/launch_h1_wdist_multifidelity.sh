#!/usr/bin/env bash
# Launch the prepared H1 multifidelity tune as a detached process.

set -euo pipefail

cd "$(dirname "$0")/.."

ROOT_OUT="${ROOT_OUT:-outputs/tune/0716_pwr_wdist_h1_multifidelity}"
SESSION="${SESSION:-h1_wdist_multifidelity_0716}"
DRIVER_LOG="${ROOT_OUT}_driver.log"

if [[ -e "${ROOT_OUT}" || -e "${DRIVER_LOG}" ]]; then
  echo "output or driver log already exists: ${ROOT_OUT}" >&2
  exit 1
fi

exec bash experiments/launch_detached_screen.sh \
  "${SESSION}" \
  "${DRIVER_LOG}" \
  env "ROOT_OUT=${ROOT_OUT}" bash experiments/run_h1_wdist_multifidelity.sh
