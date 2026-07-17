#!/usr/bin/env bash
# Launch the H1 W-Dist multifidelity tune on tangent-direction outliers.
# Same 2-stage schedule (stage1 5ep proxy -> stage2 top-4 20ep) as the uniform
# H1 tune, but with outlier_mode=local_pca_tangent (anisotropic noise).

set -euo pipefail

cd "$(dirname "$0")/.."

BASE_CONFIG="${BASE_CONFIG:-elongate_n100_no_cls_tune_local_pca_ellphi_power_h1_tangent}"
ROOT_OUT="${ROOT_OUT:-outputs/tune/0717_pwr_wdist_h1_tangent_multifidelity}"
SESSION="${SESSION:-h1_wdist_tangent_multifidelity_0717}"
DRIVER_LOG="${ROOT_OUT}_driver.log"

if [[ -e "${ROOT_OUT}" || -e "${DRIVER_LOG}" ]]; then
  echo "output or driver log already exists: ${ROOT_OUT}" >&2
  exit 1
fi

exec bash experiments/launch_detached_screen.sh \
  "${SESSION}" \
  "${DRIVER_LOG}" \
  env "ROOT_OUT=${ROOT_OUT}" "BASE_CONFIG=${BASE_CONFIG}" \
  bash experiments/run_h1_wdist_multifidelity.sh
