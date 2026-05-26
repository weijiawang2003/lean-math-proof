#!/usr/bin/env bash
# CX1 Stage 5 — eval one configuration on one CX1 theorem set.
#
# Usage:
#   bash scripts/cx1_run_eval.sh <mode> <variant_or_baseline> <set>
#
# mode = "raw"     — raw NS15 routed (no wrapper)
# mode = "ns9wrap" — NS9 best genome wrapper baseline
# mode = "variant" — experimental variant from
#                    project/evolve/experiments/ns19/ns19_<variant>.json
#                    (currently we only use finset_aesop_only)
set -euo pipefail

MODE="${1:?usage: $0 <mode> <variant_or_baseline> <set>}"
VBASE="${2:?usage: $0 <mode> <variant_or_baseline> <set>}"
SET="${3:?usage: $0 <mode> <variant_or_baseline> <set>}"
ROUTE_CONFIG="project/evolve/routing/ns15_router.json"

case "$MODE" in
  variant)
    GENOME="project/evolve/experiments/ns19/ns19_${VBASE}.json"
    TAG="cx1_${VBASE}_wrapper_${SET}"
    ;;
  ns9wrap)
    GENOME="project/evolve/best/ns9_best_genome.json"
    TAG="cx1_ns9wrap_${SET}"
    ;;
  raw)
    GENOME=""
    TAG="cx1_raw_${SET}"
    ;;
  *)
    echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

if [[ -n "$GENOME" && ! -f "$GENOME" ]]; then
  echo "missing genome: $GENOME" >&2
  exit 1
fi

OUT="project/evolve/eval_runs/${TAG}"
LOG="project/evolve/eval_runs/${TAG}.log"
EXISTING=$(ls "${OUT}"/*/metrics.json 2>/dev/null | head -1 || true)
if [[ -n "$EXISTING" ]]; then
  echo "[skip] $MODE/$VBASE/$SET already done ($EXISTING)"
  exit 0
fi

echo "[run]  $MODE/$VBASE/$SET → $OUT"
if [[ -z "$GENOME" ]]; then
  python3 eval_rollout_all.py \
    --theorem-set "$SET" \
    --policy-type routed_generative \
    --route-config "$ROUTE_CONFIG" \
    --top-k 8 --max-steps 8 \
    --out-dir "$OUT" \
    > "$LOG" 2>&1
else
  python3 eval_rollout_all.py \
    --theorem-set "$SET" \
    --policy-type hybrid_evolved \
    --route-config "$ROUTE_CONFIG" \
    --strategy-config "$GENOME" \
    --top-k 8 --max-steps 8 \
    --out-dir "$OUT" \
    > "$LOG" 2>&1
fi
echo "[done] $MODE/$VBASE/$SET"
