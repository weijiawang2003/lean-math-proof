#!/usr/bin/env bash
# NS15 Stage 4 — sequential raw eval driver.
#
# Usage:
#   bash scripts/ns15_run_evals.sh <variant>
# where <variant> is one of:
#   combined_all, nat_oversample, balanced_namespace, curriculum
#
# Output: per-theorem-set metrics under
#   project/evolve/eval_runs/gen_v5_ns15_<variant>_raw_<set>/<eval-id>/
# Logs:
#   project/evolve/eval_runs/gen_v5_ns15_<variant>_raw_<set>.log
set -euo pipefail

VARIANT="${1:?usage: $0 <variant>}"
CKPT="project/models/gen_v5_ns15_${VARIANT}"
if [[ ! -d "$CKPT" ]]; then
  echo "missing checkpoint: $CKPT" >&2
  exit 1
fi

# Eval sets to run (smallest to largest so failures show up fast).
SETS=(
  ns14_nat_extra
  ns14_set_finset_extra
  ns14_mixed_easy
  ns14_mixed_medium
  demo_v1
  nat_defs_medium
  nat_defs_large_v5
)

for S in "${SETS[@]}"; do
  TAG="gen_v5_ns15_${VARIANT}_raw_${S}"
  OUT="project/evolve/eval_runs/${TAG}"
  LOG="project/evolve/eval_runs/${TAG}.log"
  EXISTING=$(ls "${OUT}"/*/metrics.json 2>/dev/null | head -1 || true)
  if [[ -n "$EXISTING" ]]; then
    echo "[skip] $S — metrics already present ($EXISTING)"
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

echo "[done] $VARIANT evals complete"
