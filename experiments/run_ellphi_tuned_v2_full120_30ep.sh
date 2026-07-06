#!/usr/bin/env bash
# 30ep full120 with v2 ellphi-tuned hyperparams + paper eval (train topo ckpt).
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

OUT_BASE="${OUT_BASE:-outputs/supervised/20260704_ellphi_tuned_v2_full120_30ep_screen}"
BASELINE_PARENT="${BASELINE_PARENT:-outputs/supervised/20260701/190250_elongate_n100_no_cls_full120_baseline_ellphi_30ep_screen}"
CONFIG="${CONFIG:-elongate_n100_no_cls_full120_ellphi_tuned}"
BASELINE_CONFIG="${BASELINE_CONFIG:-elongate_n100_no_cls_full120_baseline}"
TUNE_JSON="${TUNE_JSON:-outputs/tune_elongate_ellphi_v2/best_elongate_wdist_ellphi.json}"

mkdir -p "$OUT_BASE"
cat > "${OUT_BASE}/PURPOSE.md" <<EOF
# ellphi full120 30ep — Optuna v2 tuned hyperparams

Tune source: \`${TUNE_JSON}\` (train_topo best ckpt, save_every=1)
Compare baseline: \`${BASELINE_PARENT}\` (maha-tuned weights)

Post-train: \`experiments/evaluate_topo_checkpoint.py\`
EOF

echo "=== train $(date -Iseconds) ==="
uv run python experiments/run_backend_multiseed.py \
  --base-config "$CONFIG" \
  --backends ellphi \
  --seeds 42 \
  --epochs 30 \
  --out-base "$OUT_BASE"

RUN_DIR=$(ls -d "${OUT_BASE}"/backend_ellphi_seed42_* | tail -1)
BASELINE_DIR=$(ls -d "${BASELINE_PARENT}"/backend_ellphi_seed42_* | tail -1)

echo "=== evaluate $(date -Iseconds) ==="
uv run python experiments/evaluate_topo_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --base-config "$CONFIG" \
  --backend ellphi \
  --baseline-run-dir "$BASELINE_DIR" \
  --baseline-config "$BASELINE_CONFIG" \
  --tag topo_best \
  --baseline-tag topo_ep30 \
  --compare-json "${RUN_DIR}/logs/ellphi_tuned_v2_vs_baseline_test.json"

echo "=== done $(date -Iseconds) ==="
echo "run_dir=$RUN_DIR"
echo "compare=${RUN_DIR}/logs/ellphi_tuned_v2_vs_baseline_test.json"
