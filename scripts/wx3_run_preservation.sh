#!/usr/bin/env bash
# WX3 Stage 7 — preservation matrix runs.
#
# WX3 configs = ns9_best_genome.json + a `Multiset.`-gated symbolic block.
# On any non-Multiset theorem every Multiset action's namespace gate blocks
# emission, so the ranked list is identical to NS9 (verified: WX3 base ==
# ns9_best_genome.json byte-for-byte). We still confirm empirically on the
# cheaper preservation sets that WX3-comb == NS9 (wins + zero Multiset
# emissions). The heavy sets (nat_defs_large_v5, ns14_set_finset_extra) are
# preserved by the same ranked-list-identity argument and not re-run.
set -uo pipefail
cd "$(dirname "$0")/.."

SETS=(demo_v1 nat_defs_medium ns17_set_extra ns17_finset_extra)
NS9="project/evolve/best/ns9_best_genome.json"
COMB="project/evolve/experiments/wx3/wx3_multiset_combined_safe.json"

for SET in "${SETS[@]}"; do
  bash scripts/wx3_run_eval.sh ns9 "$NS9" "$SET" || echo "[FAIL] ns9/$SET"
  bash scripts/wx3_run_eval.sh comb "$COMB" "$SET" || echo "[FAIL] comb/$SET"
done
echo "[preservation-done]"
