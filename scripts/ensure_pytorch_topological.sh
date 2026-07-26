#!/usr/bin/env bash
# Ensure pytorch-topological/ matches third_party/pytorch_topological.ref.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF_FILE="${ROOT}/third_party/pytorch_topological.ref"
REPO_DIR="${ROOT}/pytorch-topological"
URL="https://github.com/aidos-lab/pytorch-topological.git"

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

if [[ -f "${REPO_DIR}/pyproject.toml" ]]; then
  current="$(cd "${REPO_DIR}" && git rev-parse HEAD)"
  if [[ "${current}" == "${REF}" ]]; then
    exit 0
  fi
  checkout_ref
  exit 0
fi

git clone "${URL}" "${REPO_DIR}"
checkout_ref
