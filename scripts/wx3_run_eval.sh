#!/usr/bin/env bash
# WX3 — eval one configuration on one theorem set against the NS24 router.
#
# Usage:
#   bash scripts/wx3_run_eval.sh <tag> <genome|NONE> <set>
#
#   tag    : short label for the eval-run dir (raw / ns9 / ind / ext / comb)
#   genome : path to a strategy-config JSON, or NONE for routed-raw (no wrapper)
#   set    : theorem-set name
#
# Configs used by WX3 Stage 6:
#   raw  : NONE                                                  (routed_generative)
#   ns9  : project/evolve/best/ns9_best_genome.json             (NS9 best genome)
#   ind  : project/evolve/experiments/wx3/wx3_multiset_induction_safe.json
#   ext  : project/evolve/experiments/wx3/wx3_multiset_ext_safe.json
#   comb : project/evolve/experiments/wx3/wx3_multiset_combined_safe.json
#
# Multiset symbolic actions are namespace-gated to `Multiset.` inside the
# wrapper, so non-Multiset sets are unaffected.
set -euo pipefail

TAG="${1:?usage: $0 <tag> <genome|NONE> <set>}"
GENOME="${2:?usage: $0 <tag> <genome|NONE> <set>}"
SET="${3:?usage: $0 <tag> <genome|NONE> <set>}"

ROUTER="project/evolve/routing/ns24_router.json"
OUT="project/evolve/eval_runs/wx3_${TAG}_${SET}"
LOG="project/evolve/eval_runs/wx3_${TAG}_${SET}.log"

if [[ ! -f "$ROUTER" ]]; then echo "missing router: $ROUTER" >&2; exit 1; fi

EXISTING=$(ls "${OUT}"/*/metrics.json 2>/dev/null | head -1 || true)
if [[ -n "$EXISTING" ]]; then
  echo "[skip] $TAG/$SET already done ($EXISTING)"; exit 0
fi
mkdir -p "$OUT"

echo "[run]  $TAG/$SET (genome=$GENOME) -> $OUT"
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
