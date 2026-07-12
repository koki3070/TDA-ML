#!/usr/bin/env bash
# Ensure ellphi_repo/ matches third_party/ellphi.ref.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF_FILE="${ROOT}/third_party/ellphi.ref"
REPO_DIR="${ROOT}/ellphi_repo"
URL="https://github.com/koki3070/ellphi.git"

if [[ ! -f "${REF_FILE}" ]]; then
  echo "error: missing ${REF_FILE}" >&2
  exit 1
fi

REF="$(
  grep -v '^[[:space:]]*#' "${REF_FILE}" | grep -v '^[[:space:]]*$' | head -1 | tr -d '[:space:]'
)"
if [[ -z "${REF}" ]]; then
  echo "error: empty ref in ${REF_FILE}" >&2
  exit 1
fi

checkout_ref() {
  cd "${REPO_DIR}"
  git fetch --depth 1 origin "${REF}"
  git checkout --detach "${REF}"
}

if [[ -f "${REPO_DIR}/pyproject.toml" || -f "${REPO_DIR}/setup.py" ]]; then
  current="$(cd "${REPO_DIR}" && git rev-parse HEAD)"
  if [[ "${current}" == "${REF}" ]]; then
    exit 0
  fi
  checkout_ref
  exit 0
fi

git clone "${URL}" "${REPO_DIR}"
checkout_ref
