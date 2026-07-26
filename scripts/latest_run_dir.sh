#!/usr/bin/env bash
# Resolve the newest run directory under OUT_BASE for run_paths slug(s).
#
# Usage:
#   scripts/latest_run_dir.sh OUT_BASE SLUG [LEGACY_PREFIX...]
#
# Example (backend_ellphi_seed42 → slug eph_s42):
#   scripts/latest_run_dir.sh outputs/foo eph_s42 backend_ellphi_seed42
set -euo pipefail

OUT_BASE="${1:?OUT_BASE required}"
SLUG="${2:?SLUG required}"
shift 2
LEGACY=("$@")

latest=""
for prefix in "$SLUG" "${LEGACY[@]}"; do
  shopt -s nullglob
  dirs=("${OUT_BASE}/${prefix}_"*)
  shopt -u nullglob
  if (("${#dirs[@]}")); then
    candidate="$(printf '%s\n' "${dirs[@]}" | sort | tail -1)"
    if [[ -z "$latest" || "$candidate" > "$latest" ]]; then
      latest="$candidate"
    fi
  fi
done

if [[ -z "$latest" ]]; then
  echo "error: no run directory under ${OUT_BASE} for slug=${SLUG} legacy=${LEGACY[*]:-}" >&2
  exit 1
fi
echo "$latest"
