#!/usr/bin/env bash
# NS19 — eval one configuration on one theorem set.
#
# Usage:
#   bash scripts/ns19_run_eval.sh <mode> <variant_or_baseline> <set>
#
# mode = "variant" — runs the NS19 experimental wrapper variant
#                    located at project/evolve/experiments/ns19/ns19_<variant>.json
# mode = "ns9wrap" — runs the NS9 best genome wrapper baseline
# mode = "raw"     — runs raw NS15 routed (no wrapper)
#
# All modes use project/evolve/routing/ns15_router.json for routing.
set -euo pipefail

MODE="${1:?usage: $0 <mode> <variant_or_baseline> <set>}"
VBASE="${2:?usage: $0 <mode> <variant_or_baseline> <set>}"
SET="${3:?usage: $0 <mode> <variant_or_baseline> <set>}"
ROUTE_CONFIG="project/evolve/routing/ns15_router.json"

case "$MODE" in
  variant)
    GENOME="project/evolve/experiments/ns19/ns19_${VBASE}.json"
    TAG="ns19_${VBASE}_wrapper_${SET}"
    ;;
  ns9wrap)
    GENOME="project/evolve/best/ns9_best_genome.json"
    TAG="ns19_ns9wrap_${SET}"
    ;;
  raw)
    GENOME=""
    TAG="ns19_raw_${SET}"
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
  # Raw NS15 routed — same setup as the NS15 raw baseline.
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
