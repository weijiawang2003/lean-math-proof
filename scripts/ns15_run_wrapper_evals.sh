#!/usr/bin/env bash
# NS15 Stage 6 — wrapper-compatibility eval driver.
# Runs NS9 best genome + NS15 routed base on the core eval matrix.
set -euo pipefail

ROUTE_CONFIG="project/evolve/routing/ns15_router.json"
GENOME="project/evolve/best/ns9_best_genome.json"

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
  TAG="gen_v5_ns15_routed_wrapper_${S}"
  OUT="project/evolve/eval_runs/${TAG}"
  LOG="project/evolve/eval_runs/${TAG}.log"
  EXISTING=$(ls "${OUT}"/*/metrics.json 2>/dev/null | head -1 || true)
  if [[ -n "$EXISTING" ]]; then
    echo "[skip] $S — already done ($EXISTING)"
    continue
  fi
  echo "[run]  wrapper/$S → $OUT"
  python3 eval_rollout_all.py \
    --theorem-set "$S" \
    --policy-type hybrid_evolved \
    --route-config "$ROUTE_CONFIG" \
    --strategy-config "$GENOME" \
    --top-k 8 --max-steps 8 \
    --out-dir "$OUT" \
    > "$LOG" 2>&1
done

echo "[done] ns15 wrapper evals complete"
