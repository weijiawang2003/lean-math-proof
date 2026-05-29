#!/usr/bin/env bash
# NS24 — eval one configuration on one theorem set. Mirrors
# scripts/ns22_run_eval.sh; only the TAG prefix differs (ns24_*).
#
# Usage:
#   bash scripts/ns24_run_eval.sh <mode> <variant> <set>
#
# mode = "raw_ckpt"    — single-checkpoint raw policy (variant = ckpt name
#                        under project/models/)
# mode = "raw_routed"  — multi-route raw policy (variant = router basename
#                        under project/evolve/routing/)
# mode = "wrap_routed" — NS9 best genome wrapper + given router (variant =
#                        router basename)
set -euo pipefail

MODE="${1:?usage: $0 <mode> <variant> <set>}"
VBASE="${2:?usage: $0 <mode> <variant> <set>}"
SET="${3:?usage: $0 <mode> <variant> <set>}"

case "$MODE" in
  raw_ckpt)
    CKPT="project/models/${VBASE}"
    TAG="ns24_rawckpt_${VBASE}_${SET}"
    GENOME=""
    ROUTE_CONFIG=""
    SINGLE_CKPT="$CKPT"
    ;;
  raw_routed)
    ROUTE_CONFIG="project/evolve/routing/${VBASE}.json"
    TAG="ns24_rawrouted_${VBASE}_${SET}"
    GENOME=""
    SINGLE_CKPT=""
    ;;
  wrap_routed)
    ROUTE_CONFIG="project/evolve/routing/${VBASE}.json"
    GENOME="project/evolve/best/ns9_best_genome.json"
    TAG="ns24_wraprouted_${VBASE}_${SET}"
    SINGLE_CKPT=""
    ;;
  *)
    echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

if [[ -n "$ROUTE_CONFIG" && ! -f "$ROUTE_CONFIG" ]]; then
  echo "missing router: $ROUTE_CONFIG" >&2; exit 1
fi
if [[ -n "$GENOME" && ! -f "$GENOME" ]]; then
  echo "missing genome: $GENOME" >&2; exit 1
fi
if [[ -n "$SINGLE_CKPT" && ! -d "$SINGLE_CKPT" ]]; then
  echo "missing ckpt dir: $SINGLE_CKPT" >&2; exit 1
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
  raw_ckpt)
    python3 eval_rollout_all.py \
      --theorem-set "$SET" \
      --policy-type generative \
      --ckpt-dir "$SINGLE_CKPT" \
      --top-k 8 --max-steps 8 \
      --out-dir "$OUT" \
      > "$LOG" 2>&1
    ;;
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
