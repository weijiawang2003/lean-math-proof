#!/usr/bin/env bash
# CX3 — eval one configuration on one Bool/Option theorem set.
# Mirrors scripts/ns24_run_eval.sh; only the TAG prefix differs (cx3_*).
#
# Usage:
#   bash scripts/cx3_run_eval.sh <mode> <router_basename> <set>
#
# mode = "raw_routed"  — multi-route raw policy (Bool/Option fall through
#                        to the router default checkpoint)
# mode = "wrap_routed" — NS9 best genome wrapper + given router
set -euo pipefail

MODE="${1:?usage: $0 <mode> <router> <set>}"
VBASE="${2:?usage: $0 <mode> <router> <set>}"
SET="${3:?usage: $0 <mode> <router> <set>}"

case "$MODE" in
  raw_routed)
    ROUTE_CONFIG="project/evolve/routing/${VBASE}.json"
    GENOME=""
    TAG="cx3_rawrouted_${VBASE}_${SET}"
    ;;
  wrap_routed)
    ROUTE_CONFIG="project/evolve/routing/${VBASE}.json"
    GENOME="project/evolve/best/ns9_best_genome.json"
    TAG="cx3_wraprouted_${VBASE}_${SET}"
    ;;
  *)
    echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

if [[ ! -f "$ROUTE_CONFIG" ]]; then
  echo "missing router: $ROUTE_CONFIG" >&2; exit 1
fi
if [[ -n "$GENOME" && ! -f "$GENOME" ]]; then
  echo "missing genome: $GENOME" >&2; exit 1
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
case "$MODE" in
  raw_routed)
    python3 eval_rollout_all.py \
      --theorem-set "$SET" \
      --policy-type routed_generative \
      --route-config "$ROUTE_CONFIG" \
      --top-k 8 --max-steps 8 \
      --out-dir "$OUT" \
      > "$LOG" 2>&1
    ;;
  wrap_routed)
    python3 eval_rollout_all.py \
      --theorem-set "$SET" \
      --policy-type hybrid_evolved \
      --route-config "$ROUTE_CONFIG" \
      --strategy-config "$GENOME" \
      --top-k 8 --max-steps 8 \
      --out-dir "$OUT" \
      > "$LOG" 2>&1
    ;;
esac
echo "[done] $MODE/$VBASE/$SET"
