#!/usr/bin/env bash
# NS16 Stage 6 — raw eval driver per NS16 checkpoint variant.
# Usage: bash scripts/ns16_run_raw_evals.sh <variant>
# variants: oversample_10x, oversample_20x, curriculum_continue
set -euo pipefail

VARIANT="${1:?usage: $0 <variant>}"
CKPT="project/models/gen_v5_ns16_${VARIANT}"
if [[ ! -d "$CKPT" ]]; then
  echo "missing checkpoint: $CKPT" >&2
  exit 1
fi

SETS=(
  ns14_nat_extra
  ns16_nat_iff_extra
  ns16_nat_div_mod_extra
  ns16_nat_order_extra
  ns16_nat_mixed_extra
  demo_v1
  nat_defs_medium
  nat_defs_large_v5
)

for S in "${SETS[@]}"; do
  TAG="gen_v5_ns16_${VARIANT}_raw_${S}"
  OUT="project/evolve/eval_runs/${TAG}"
  LOG="project/evolve/eval_runs/${TAG}.log"
  EXISTING=$(ls "${OUT}"/*/metrics.json 2>/dev/null | head -1 || true)
  if [[ -n "$EXISTING" ]]; then
    echo "[skip] $S — already done"
    continue
  fi
  echo "[run]  $S → $OUT"
  python3 eval_rollout_all.py \
    --theorem-set "$S" \
    --ckpt-dir "$CKPT" \
    --policy-type generative \
    --top-k 8 --max-steps 8 \
    --out-dir "$OUT" \
    > "$LOG" 2>&1
done

echo "[done] $VARIANT raw evals complete"
