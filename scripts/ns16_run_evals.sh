#!/usr/bin/env bash
# NS16 Stage 2 eval driver.
#
# Usage:
#   bash scripts/ns16_run_evals.sh <mode>
# where <mode> is one of:
#   raw       — raw NS15 routed
#   wrapper   — NS9 best genome + NS15 routed
set -euo pipefail

MODE="${1:?usage: $0 <raw|wrapper>}"
ROUTE_CONFIG="project/evolve/routing/ns15_router.json"
GENOME="project/evolve/best/ns9_best_genome.json"

SETS=(
  ns16_nat_iff_extra
  ns16_nat_div_mod_extra
  ns16_nat_order_extra
  ns16_nat_mixed_extra
)

case "$MODE" in
  raw)
    POLICY=routed_generative
    EXTRA=()
    TAG_PREFIX="ns16_ns15routed_raw"
    ;;
  wrapper)
    POLICY=hybrid_evolved
    EXTRA=(--strategy-config "$GENOME")
    TAG_PREFIX="ns16_ns15routed_wrapper"
    ;;
  *)
    echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

for S in "${SETS[@]}"; do
  TAG="${TAG_PREFIX}_${S}"
  OUT="project/evolve/eval_runs/${TAG}"
  LOG="project/evolve/eval_runs/${TAG}.log"
  EXISTING=$(ls "${OUT}"/*/metrics.json 2>/dev/null | head -1 || true)
  if [[ -n "$EXISTING" ]]; then
    echo "[skip] $S — already done ($EXISTING)"
    continue
  fi
  echo "[run]  $MODE/$S → $OUT"
  python3 eval_rollout_all.py \
    --theorem-set "$S" \
    --policy-type "$POLICY" \
    --route-config "$ROUTE_CONFIG" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    --top-k 8 --max-steps 8 \
    --out-dir "$OUT" \
    > "$LOG" 2>&1
done

echo "[done] $MODE evals complete"
