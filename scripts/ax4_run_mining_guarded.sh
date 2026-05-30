#!/usr/bin/env bash
# AX4 Stage 3 — mine the AX4 frontier theorem sets under the NS24 router with
# three configs (raw / NS9 / WX3 oracle induction wrapper), each guarded by a
# per-run timeout so a single Lean REPL load-hang cannot stall the pipeline.
# Plain bash (no xargs); PAR concurrent jobs; idempotent (skips cells whose
# metrics.json already exists). wx3ind is scheduled FIRST per set so the
# label-bearing data (+ state_pp traces) lands even if later cells are killed.
#   Usage: bash scripts/ax4_run_mining_guarded.sh [PAR] [TLIMIT_SECONDS]
set -uo pipefail
cd "$(dirname "$0")/.."
PAR="${1:-3}"; TLIMIT="${2:-2400}"
ROUTER=project/evolve/routing/ns24_router.json
NS9=project/evolve/best/ns9_best_genome.json
WX3=project/evolve/experiments/wx3/wx3_multiset_induction_safe.json

SETS=(
  ax4_multiset_induction_high_confidence
  ax4_multiset_cross_surface
  ax4_multiset_induction_heldout
  ax4_multiset_induction_medium_confidence
  ax4_multiset_induction_hard
  ax4_multiset_negative_control
)

run_cell() {
  local TAG="$1" GEN="$2" SET="$3"
  local OUT="project/evolve/eval_runs/ax4_${TAG}_${SET}" LOG
  LOG="${OUT}.log"
  if ls "$OUT"/eval-*/metrics.json >/dev/null 2>&1; then echo "[skip] $TAG/$SET"; return 0; fi
  mkdir -p "$OUT"
  echo "[run]  $TAG/$SET -> $OUT"
  if [ "$GEN" = "NONE" ]; then
    python3 scripts/run_with_timeout.py "$TLIMIT" python3 eval_rollout_all.py --theorem-set "$SET" \
      --policy-type routed_generative --route-config "$ROUTER" \
      --top-k 8 --max-steps 8 --out-dir "$OUT" > "$LOG" 2>&1
  else
    python3 scripts/run_with_timeout.py "$TLIMIT" python3 eval_rollout_all.py --theorem-set "$SET" \
      --policy-type hybrid_evolved --route-config "$ROUTER" \
      --strategy-config "$GEN" --top-k 8 --max-steps 8 --out-dir "$OUT" > "$LOG" 2>&1
  fi
  echo "[done $TAG/$SET rc=$?]"
}

i=0
for SET in "${SETS[@]}"; do
  # wx3ind first (label-bearing), then ns9, then raw
  for SPEC in "wx3ind $WX3" "ns9 $NS9" "raw NONE"; do
    set -- $SPEC
    run_cell "$1" "$2" "$SET" &
    i=$((i+1))
    if [ $((i % PAR)) -eq 0 ]; then wait -n 2>/dev/null || wait; fi
  done
done
wait
echo "[ax4-mining-done]"
