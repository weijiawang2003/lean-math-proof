#!/usr/bin/env bash
# NS15 Stage 5 — routed-policy eval driver.
# Runs the NS15 routed policy on the eval matrix.
set -euo pipefail

ROUTE_CONFIG="project/evolve/routing/ns15_router.json"

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
  TAG="gen_v5_ns15_routed_raw_${S}"
  OUT="project/evolve/eval_runs/${TAG}"
  LOG="project/evolve/eval_runs/${TAG}.log"
  EXISTING=$(ls "${OUT}"/*/metrics.json 2>/dev/null | head -1 || true)
  if [[ -n "$EXISTING" ]]; then
    echo "[skip] $S — already done ($EXISTING)"
    continue
  fi
  echo "[run]  routed/$S → $OUT"
  python3 eval_rollout_all.py \
    --theorem-set "$S" \
    --policy-type routed_generative \
    --route-config "$ROUTE_CONFIG" \
    --top-k 8 --max-steps 8 \
    --out-dir "$OUT" \
    > "$LOG" 2>&1
done

echo "[done] ns15 routed evals complete"
