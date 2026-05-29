#!/usr/bin/env bash
# MX2 Stage 4 — live eval of the Set-aesop fallback configs.
#
# Variants over the MX2 Set theorem sets against the NS24 router, LIVE:
#   A prod   : production wrapper (WX3 oracle config; no Set aesop)
#   B broad  : mx2_set_aesop_safe        (aesop gated to `Set.`)
#   C narrow : mx2_set_finite_aesop_safe (aesop gated to `Set.Finite.`/`Set.toFinset`)
# Idempotent; trace dirs git-ignored.
set -uo pipefail
cd "$(dirname "$0")/.."

ROUTER="project/evolve/routing/ns24_router.json"
PROD="project/evolve/experiments/wx3/wx3_multiset_induction_safe.json"
BROAD="project/evolve/experiments/mx2/mx2_set_aesop_safe.json"
NARROW="project/evolve/experiments/mx2/mx2_set_finite_aesop_safe.json"
TPT=900

SETS=(mx2_set_aesop_known mx2_set_finite_frontier mx2_set_aesop_frontier \
      mx2_set_negative_control mx2_mixed_preservation_control)

run() {  # tag set cfg
  local tag="$1" set="$2" cfg="$3"
  local out="project/evolve/eval_runs/mx2_${tag}_${set}"
  if ls "${out}"/eval-*/traces.jsonl >/dev/null 2>&1; then
    echo "[skip] ${tag}/${set}"; return 0; fi
  mkdir -p "$out"
  echo "[run ] ${tag}/${set} (cfg=${cfg})"
  python3 scripts/run_with_timeout.py "$TPT" python3 eval_rollout_all.py \
    --theorem-set "$set" --policy-type hybrid_evolved --route-config "$ROUTER" \
    --strategy-config "$cfg" --top-k 8 --max-steps 8 --out-dir "$out" \
    > "${out}/run.log" 2>&1
  echo "[done] ${tag}/${set} (exit $?)"
}

# Narrow config (C) gates aesop to Set.Finite./Set.toFinset, so it can only
# differ from production on sets containing those names; elsewhere C == A. Run C
# only where it can fire to save runtime.
C_SETS=(mx2_set_aesop_known mx2_set_finite_frontier)
in_cset() { local x="$1"; for c in "${C_SETS[@]}"; do [ "$c" = "$x" ] && return 0; done; return 1; }

for s in "${SETS[@]}"; do
  run A_prod   "$s" "$PROD"
  run B_broad  "$s" "$BROAD"
  if in_cset "$s"; then run C_narrow "$s" "$NARROW"; fi
done
echo "MX2 EVAL DRIVER COMPLETE"
