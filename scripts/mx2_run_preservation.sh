#!/usr/bin/env bash
# MX2 Stage 6 — live preservation runs for the best Set-aesop config.
#
# Runs production (A) vs the chosen MX2 config (the broad mx2_set_aesop_safe by
# default) LIVE on preservation sets that the Set gate could touch or that define
# the canonical floors. Off-gate Nat/large floors get 0 aesop emissions (proven
# statically) and are not re-run unless cheap. Idempotent; trace dirs ignored.
set -uo pipefail
cd "$(dirname "$0")/.."

ROUTER="project/evolve/routing/ns24_router.json"
PROD="project/evolve/experiments/wx3/wx3_multiset_induction_safe.json"
MX2="project/evolve/experiments/mx2/mx2_set_aesop_safe.json"
TPT=900
# Live regression check on the main gated Set surface. Off-gate Nat/demo/large
# floors get 0 aesop emissions (proven statically by mx2_preservation_matrix.py)
# and are not re-run; ns17_set_extra is the representative live gated check.
SETS=(ns17_set_extra)

run() {  # tag set cfg
  local tag="$1" set="$2" cfg="$3"
  local out="project/evolve/eval_runs/mx2_pres_${tag}_${set}"
  if ls "${out}"/eval-*/traces.jsonl >/dev/null 2>&1; then
    echo "[skip] pres ${tag}/${set}"; return 0; fi
  mkdir -p "$out"
  echo "[run ] pres ${tag}/${set}"
  python3 scripts/run_with_timeout.py "$TPT" python3 eval_rollout_all.py \
    --theorem-set "$set" --policy-type hybrid_evolved --route-config "$ROUTER" \
    --strategy-config "$cfg" --top-k 8 --max-steps 8 --out-dir "$out" \
    > "${out}/run.log" 2>&1
  echo "[done] pres ${tag}/${set} (exit $?)"
}

for s in "${SETS[@]}"; do
  run A "$s" "$PROD"
  run E "$s" "$MX2"
done
echo "MX2 PRESERVATION RUNS COMPLETE"
