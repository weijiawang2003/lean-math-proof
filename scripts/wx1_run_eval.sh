#!/usr/bin/env bash
# WX1 — eval one configuration on one theorem set, against the NS24 router.
#
# Usage:
#   bash scripts/wx1_run_eval.sh <tag> <genome|NONE> <set>
#
#   tag    : short label for the eval-run dir (e.g. raw / ns9 / wx1 / wx1b)
#   genome : path to a strategy-config JSON, or NONE for routed-raw (no wrapper)
#   set    : theorem-set name
#
# Bool/Option fall through to the NS24 router default checkpoint
# (gen_v5_ns12_balanced); the WX1 option-cases skeletons are namespace-gated
# inside the wrapper so Nat/Set/Finset are unaffected.
set -euo pipefail

TAG="${1:?usage: $0 <tag> <genome|NONE> <set>}"
GENOME="${2:?usage: $0 <tag> <genome|NONE> <set>}"
SET="${3:?usage: $0 <tag> <genome|NONE> <set>}"

ROUTER="project/evolve/routing/ns24_router.json"
OUT="project/evolve/eval_runs/wx1_${TAG}_${SET}"
LOG="project/evolve/eval_runs/wx1_${TAG}_${SET}.log"

if [[ ! -f "$ROUTER" ]]; then echo "missing router: $ROUTER" >&2; exit 1; fi

EXISTING=$(ls "${OUT}"/*/metrics.json 2>/dev/null | head -1 || true)
if [[ -n "$EXISTING" ]]; then
  echo "[skip] $TAG/$SET already done ($EXISTING)"; exit 0
fi
mkdir -p "$OUT"

echo "[run]  $TAG/$SET (genome=$GENOME) → $OUT"
if [[ "$GENOME" == "NONE" ]]; then
  python3 eval_rollout_all.py \
    --theorem-set "$SET" \
    --policy-type routed_generative \
    --route-config "$ROUTER" \
    --top-k 8 --max-steps 8 \
    --out-dir "$OUT" > "$LOG" 2>&1
else
  if [[ ! -f "$GENOME" ]]; then echo "missing genome: $GENOME" >&2; exit 1; fi
  python3 eval_rollout_all.py \
    --theorem-set "$SET" \
    --policy-type hybrid_evolved \
    --route-config "$ROUTER" \
    --strategy-config "$GENOME" \
    --top-k 8 --max-steps 8 \
    --out-dir "$OUT" > "$LOG" 2>&1
fi
echo "[done] $TAG/$SET"
