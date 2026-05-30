#!/usr/bin/env bash
# MX1 Stage 7 — live preservation regression runs.
#
# Runs variant B (production WX3) and E (MX1 combined symbolic) LIVE on the
# GATED preservation sets where the new Set/Finset actions actually emit, so a
# regression (production win lost) would show up. Off-gate Nat/demo/large floors
# get 0 emissions (proven statically by mx1_preservation_matrix.py) and are not
# re-run here. Idempotent; trace dirs are git-ignored.
set -uo pipefail
cd "$(dirname "$0")/.."

ROUTER="project/evolve/routing/ns24_router.json"
WX3="project/evolve/experiments/wx3/wx3_multiset_induction_safe.json"
MX1E="project/evolve/experiments/mx1/mx1_combined_symbolic_frontier_safe.json"
TPT=900
SETS=(ns17_set_extra ns17_finset_extra)

run() {  # tag set cfg
  local tag="$1" set="$2" cfg="$3"
  local out="project/evolve/eval_runs/mx1_pres_${tag}_${set}"
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
  run B "$s" "$WX3"
  run E "$s" "$MX1E"
done
echo "MX1 PRESERVATION RUNS COMPLETE"
