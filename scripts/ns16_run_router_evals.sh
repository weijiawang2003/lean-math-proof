#!/usr/bin/env bash
# NS16 Stage 6 — routed + wrapper eval driver using ns16_router.json.
set -euo pipefail

MODE="${1:?usage: $0 <routed|wrapper>}"
ROUTE_CONFIG="project/evolve/routing/ns16_router.json"
GENOME="project/evolve/best/ns9_best_genome.json"

SETS=(
  ns14_nat_extra
  ns14_set_finset_extra
  ns16_nat_iff_extra
  ns16_nat_div_mod_extra
  ns16_nat_order_extra
  ns16_nat_mixed_extra
  demo_v1
  nat_defs_medium
  nat_defs_large_v5
)

case "$MODE" in
  routed)
    POLICY=routed_generative
    EXTRA=()
    TAG_PREFIX="gen_v5_ns16_routed_raw"
    ;;
  wrapper)
    POLICY=hybrid_evolved
    EXTRA=(--strategy-config "$GENOME")
    TAG_PREFIX="gen_v5_ns16_routed_wrapper"
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
    echo "[skip] $S — already done"
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

echo "[done] $MODE eval complete"
