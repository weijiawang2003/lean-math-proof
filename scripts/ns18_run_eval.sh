#!/usr/bin/env bash
# NS18 — eval one experimental wrapper variant on one theorem set.
# Usage: bash scripts/ns18_run_eval.sh <variant> <set>
set -euo pipefail

VARIANT="${1:?usage: $0 <variant> <set>}"
SET="${2:?usage: $0 <variant> <set>}"
ROUTE_CONFIG="project/evolve/routing/ns15_router.json"
GENOME="project/evolve/experiments/ns18/ns18_${VARIANT}.json"

if [[ ! -f "$GENOME" ]]; then
  echo "missing genome: $GENOME" >&2
  exit 1
fi

TAG="ns18_${VARIANT}_wrapper_${SET}"
OUT="project/evolve/eval_runs/${TAG}"
LOG="project/evolve/eval_runs/${TAG}.log"
EXISTING=$(ls "${OUT}"/*/metrics.json 2>/dev/null | head -1 || true)
if [[ -n "$EXISTING" ]]; then
  echo "[skip] $VARIANT/$SET already done ($EXISTING)"
  exit 0
fi

echo "[run]  $VARIANT/$SET → $OUT"
python3 eval_rollout_all.py \
  --theorem-set "$SET" \
  --policy-type hybrid_evolved \
  --route-config "$ROUTE_CONFIG" \
  --strategy-config "$GENOME" \
  --top-k 8 --max-steps 8 \
  --out-dir "$OUT" \
  > "$LOG" 2>&1
echo "[done] $VARIANT/$SET"
