#!/usr/bin/env bash
# AX3 — run remaining matrix cells, each under a per-run timeout guard so a
# single Lean REPL hang cannot stall the pipeline. Plain bash (no xargs);
# PAR concurrent jobs. Idempotent (skips cells with metrics.json).
set -uo pipefail
cd "$(dirname "$0")/.."
PAR="${1:-3}"; TLIMIT="${2:-1400}"
ROUTER=project/evolve/routing/ns24_router.json
SETS=(ax3_multiset_induction_mine ax3_multiset_mixed_heldout ax3_multiset_negative_control)

run_cell() {
  local TAG="$1" GEN="$2" SET="$3"
  local OUT="project/evolve/eval_runs/ax3_${TAG}_${SET}" LOG
  LOG="${OUT}.log"
  if ls "$OUT"/eval-*/metrics.json >/dev/null 2>&1; then echo "[skip] $TAG/$SET"; return 0; fi
  mkdir -p "$OUT"
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
  for SPEC in "raw NONE" "ns9 project/evolve/best/ns9_best_genome.json" \
              "wx3ind project/evolve/experiments/wx3/wx3_multiset_induction_safe.json"; do
    set -- $SPEC
    run_cell "$1" "$2" "$SET" &
    i=$((i+1))
    if [ $((i % PAR)) -eq 0 ]; then wait -n 2>/dev/null || wait; fi
  done
done
wait
echo "[ax3-remaining-done]"
