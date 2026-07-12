#!/usr/bin/env bash
# Train ellphi 30ep with tuned hyperparams, then evaluate vs maha-hparam baseline.
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

OUT_BASE="outputs/supervised/20260703_ellphi_tuned_full120_30ep_screen"
BASELINE_PARENT="outputs/supervised/20260701/190250_elongate_n100_no_cls_full120_baseline_ellphi_30ep_screen"
CONFIG="elongate_n100_no_cls_full120_ellphi_tuned"
BASELINE_CONFIG="elongate_n100_no_cls_full120_baseline"

mkdir -p "$OUT_BASE"
cat > "${OUT_BASE}/PURPOSE.md" <<'EOF'
# ellphi full120 30ep with Optuna-tuned hyperparams (ellphi backend)

Compare against `.../190250_..._baseline_ellphi_30ep_screen` (same config shape, maha-tuned weights).

Tune source: `outputs/tune_elongate_ellphi/best_elongate_wdist_ellphi.json`
- w_topo=0.2088, w_aniso=0.0843, w_size=0.4446, lr=0.000287

Post-train: `experiments/evaluate_topo_checkpoint.py` (train_topo_loss min checkpoint → test).
EOF

echo "=== train $(date -Iseconds) ==="
uv run python experiments/run_backend_multiseed.py \
  --base-config "$CONFIG" \
  --backends ellphi \
  --seeds 42 \
  --epochs 30 \
  --out-base "$OUT_BASE"

RUN_DIR=$(scripts/latest_run_dir.sh "$OUT_BASE" eph_s42 backend_ellphi_seed42)
BASELINE_DIR=$(scripts/latest_run_dir.sh "$BASELINE_PARENT" eph_s42 backend_ellphi_seed42)

echo "=== evaluate $(date -Iseconds) ==="
uv run python experiments/evaluate_topo_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --base-config "$CONFIG" \
  --backend ellphi \
  --baseline-run-dir "$BASELINE_DIR" \
  --baseline-config "$BASELINE_CONFIG" \
  --compare-json "${RUN_DIR}/logs/ellphi_tuned_vs_maha_hparams_test.json"

echo "=== done $(date -Iseconds) ==="
echo "run_dir=$RUN_DIR"
echo "compare=${RUN_DIR}/logs/ellphi_tuned_vs_maha_hparams_test.json"
