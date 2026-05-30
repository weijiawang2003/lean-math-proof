#!/usr/bin/env bash
# AX3 — eval one configuration on one theorem set against the NS24 router.
#   Usage: bash scripts/ax3_run_eval.sh <tag> <genome|NONE> <set>
# Configs:
#   raw    : NONE                                              (routed_generative)
#   ns9    : project/evolve/best/ns9_best_genome.json
#   wx3ind : project/evolve/experiments/wx3/wx3_multiset_induction_safe.json
#   ax3pred: project/evolve/experiments/ax3/ax3_multiset_symbolic_predictor.json
set -euo pipefail
TAG="${1:?usage: $0 <tag> <genome|NONE> <set>}"
GENOME="${2:?}"; SET="${3:?}"
ROUTER="project/evolve/routing/ns24_router.json"
OUT="project/evolve/eval_runs/ax3_${TAG}_${SET}"
LOG="project/evolve/eval_runs/ax3_${TAG}_${SET}.log"
[[ -f "$ROUTER" ]] || { echo "missing router" >&2; exit 1; }
EXISTING=$(ls "${OUT}"/*/metrics.json 2>/dev/null | head -1 || true)
if [[ -n "$EXISTING" ]]; then echo "[skip] $TAG/$SET ($EXISTING)"; exit 0; fi
mkdir -p "$OUT"
echo "[run]  $TAG/$SET (genome=$GENOME) -> $OUT"
if [[ "$GENOME" == "NONE" ]]; then
  python3 eval_rollout_all.py --theorem-set "$SET" --policy-type routed_generative \
    --route-config "$ROUTER" --top-k 8 --max-steps 8 --out-dir "$OUT" > "$LOG" 2>&1
else
  [[ -f "$GENOME" ]] || { echo "missing genome: $GENOME" >&2; exit 1; }
  python3 eval_rollout_all.py --theorem-set "$SET" --policy-type hybrid_evolved \
    --route-config "$ROUTER" --strategy-config "$GENOME" \
    --top-k 8 --max-steps 8 --out-dir "$OUT" > "$LOG" 2>&1
fi
echo "[done] $TAG/$SET"
