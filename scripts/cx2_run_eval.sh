#!/usr/bin/env bash
# CX2 Stage 3 — eval one configuration on one CX2 Int theorem set.
#
# Usage:
#   bash scripts/cx2_run_eval.sh <mode> <variant_or_baseline> <set>
#
# mode = "raw"        — raw NS15 routed (no wrapper)
# mode = "ns9wrap"    — NS9 best genome wrapper baseline
# mode = "variant"    — experimental wrapper from
#                       project/evolve/experiments/cx2/<variant>.json
set -euo pipefail

MODE="${1:?usage: $0 <mode> <variant> <set>}"
VBASE="${2:?usage: $0 <mode> <variant> <set>}"
SET="${3:?usage: $0 <mode> <variant> <set>}"
ROUTE_CONFIG="project/evolve/routing/ns15_router.json"

case "$MODE" in
  variant)
    GENOME="project/evolve/experiments/cx2/${VBASE}.json"
    TAG="cx2_${VBASE}_${SET}"
    ;;
  ns9wrap)
    GENOME="project/evolve/best/ns9_best_genome.json"
    TAG="cx2_ns9wrap_${SET}"
    ;;
  raw)
    GENOME=""
    TAG="cx2_raw_${SET}"
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
mkdir -p "$OUT"

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
