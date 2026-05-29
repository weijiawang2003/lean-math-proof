#!/usr/bin/env bash
# MX1 Stage 3 — live LeanDojo mining driver.
#
# Runs the mining variants over the MX1 frontier sets against the NS24 router,
# LIVE (real Dojo sessions), writing per-run trace dirs under eval_runs/ (all
# git-ignored). Idempotent: a run with existing traces is skipped.
#
# Variants:
#   A raw   : routed_generative, no wrapper          (NS24 router raw baseline)
#   B prod  : hybrid_evolved + WX3 oracle config      (production wrapper:
#             Multiset symbolic + NS9 retrieval/templates; NS9 elsewhere)
#   E sym   : hybrid_evolved + MX1 combined symbolic   (B + new Set/Finset
#             ext/cases symbolic actions — the extended frontier wrapper)
#   D seq   : hybrid_evolved + SX1 combined sequence   (depth-2 trace generator;
#             Multiset/List only) — run on multiset+list sets only.
#
# Variant C (AX4 predictor) is computed OFFLINE from B's Multiset traces by the
# aggregator (the predictor only suppresses NULL-scored Multiset emissions), so
# it needs no live run.
set -uo pipefail
cd "$(dirname "$0")/.."

ROUTER="project/evolve/routing/ns24_router.json"
WX3="project/evolve/experiments/wx3/wx3_multiset_induction_safe.json"
MX1E="project/evolve/experiments/mx1/mx1_combined_symbolic_frontier_safe.json"
SX1D="project/evolve/experiments/sx1/sx1_combined_sequence_safe.json"
TPT=900   # per-(variant,set) wall-clock cap (s); ~2s/theorem => generous

ALL_SETS=(mx1_multiset_frontier mx1_finset_symbolic_frontier mx1_list_frontier \
          mx1_set_ext_frontier mx1_mixed_symbolic_frontier)
SEQ_SETS=(mx1_multiset_frontier mx1_list_frontier)

run() {  # tag set policy strategy_or_NONE
  local tag="$1" set="$2" pol="$3" cfg="$4"
  local out="project/evolve/eval_runs/mx1_${tag}_${set}"
  if ls "${out}"/eval-*/traces.jsonl >/dev/null 2>&1; then
    echo "[skip] ${tag}/${set} (traces exist)"; return 0
  fi
  mkdir -p "$out"
  echo "[run ] ${tag}/${set} (pol=${pol} cfg=${cfg})"
  if [[ "$cfg" == "NONE" ]]; then
    python3 scripts/run_with_timeout.py "$TPT" python3 eval_rollout_all.py \
      --theorem-set "$set" --policy-type "$pol" --route-config "$ROUTER" \
      --top-k 8 --max-steps 8 --out-dir "$out" > "${out}/run.log" 2>&1
  else
    python3 scripts/run_with_timeout.py "$TPT" python3 eval_rollout_all.py \
      --theorem-set "$set" --policy-type "$pol" --route-config "$ROUTER" \
      --strategy-config "$cfg" --top-k 8 --max-steps 8 --out-dir "$out" \
      > "${out}/run.log" 2>&1
  fi
  echo "[done] ${tag}/${set} (exit $?)"
}

for s in "${ALL_SETS[@]}"; do
  run A_raw  "$s" routed_generative NONE
  run B_prod "$s" hybrid_evolved   "$WX3"
  run E_sym  "$s" hybrid_evolved   "$MX1E"
done
for s in "${SEQ_SETS[@]}"; do
  run D_seq  "$s" hybrid_evolved   "$SX1D"
done
echo "MX1 LIVE MINING DRIVER COMPLETE"
