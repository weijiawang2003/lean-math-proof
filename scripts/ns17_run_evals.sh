#!/usr/bin/env bash
# NS17 Stage 5 — raw vs wrapper eval on NS17 family-mining surfaces.
# Usage: bash scripts/ns17_run_evals.sh <raw|wrapper>
set -euo pipefail

MODE="${1:?usage: $0 <raw|wrapper>}"
ROUTE_CONFIG="project/evolve/routing/ns15_router.json"
GENOME="project/evolve/best/ns9_best_genome.json"

SETS=(
  ns17_nat_remaining
  ns17_set_extra
  ns17_finset_extra
  ns17_list_multiset
)

case "$MODE" in
  raw)
    POLICY=routed_generative
    EXTRA=()
    TAG_PREFIX="ns17_ns15routed_raw"
    ;;
  wrapper)
    POLICY=hybrid_evolved
    EXTRA=(--strategy-config "$GENOME")
    TAG_PREFIX="ns17_ns15routed_wrapper"
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

echo "[done] $MODE evals complete"
