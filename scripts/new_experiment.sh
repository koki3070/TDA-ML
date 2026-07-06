#!/usr/bin/env bash
# 新しい数値実験フォルダを規約どおりに作成する。
#   outputs/<supervised|unsupervised>/<YYYYMMDD>/<HHMMSS>_<slug>/
# PURPOSE.md の雛形を生成し、run_backend_multiseed.py に渡す out-base のパスを表示する。
#
# 使い方:
#   scripts/new_experiment.sh [--unsupervised] <slug> ["1行の目的"]
# 例:
#   scripts/new_experiment.sh mahal_n100_noprob_30ep_seed42 "確率重みOFFの収束確認"
#   scripts/new_experiment.sh --unsupervised topo_prior_30ep "topo-prior のみ"
set -euo pipefail

mode_dir="supervised"
if [[ "${1:-}" == "--unsupervised" || "${1:-}" == "-u" ]]; then
  mode_dir="unsupervised"
  shift
fi

if [[ $# -lt 1 ]]; then
  echo "usage: $0 [--unsupervised] <slug> [\"one-line purpose\"]" >&2
  exit 1
fi

slug="$1"
purpose="${2:-（目的をここに記述）}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
date_dir="$(date +%Y%m%d)"
time_id="$(date +%H%M%S)"
exp_dir="${repo_root}/outputs/${mode_dir}/${date_dir}/${time_id}_${slug}"

mkdir -p "${exp_dir}"

cat > "${exp_dir}/PURPOSE.md" <<EOF
# ${slug}

- 日時: $(date '+%Y-%m-%d %H:%M JST') 開始
- git HEAD: $(git -C "${repo_root}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
- config: \`config_snapshot.yaml\`

## 目的・仮説

${purpose}

## 設定（要点）

- 

## 実行コマンド

\`\`\`bash
\`\`\`

## 結果

- 

## 解釈

- 
EOF

echo "created: ${exp_dir}"
echo ""
echo "次の手順:"
echo "  1) 使う config を snapshot:   cp configs/<your>.yaml ${exp_dir}/config_snapshot.yaml"
echo "  2) 実行時に out-base を指定:   --out-base ${exp_dir}"
echo "     （標準出力は ... | tee ${exp_dir}/run.log で保存）"
echo "  3) 実行後 PURPOSE.md の結果/解釈を埋める"
echo ""
echo "out-base path (copy):"
echo "${exp_dir}"
