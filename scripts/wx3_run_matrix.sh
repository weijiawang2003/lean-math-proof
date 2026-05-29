#!/usr/bin/env bash
# WX3 Stage 6 — full evaluation matrix: configs x Multiset theorem sets.
# Sequential, idempotent (each run skips if metrics.json already exists).
set -uo pipefail
cd "$(dirname "$0")/.."

SETS=(
  wx3_multiset_simp_easy
  wx3_multiset_induction_easy
  wx3_multiset_ext_medium
  wx3_multiset_quotient_medium
  wx3_multiset_mixed
)

# tag  genome
CONFIGS=(
  "raw|NONE"
  "ns9|project/evolve/best/ns9_best_genome.json"
  "ind|project/evolve/experiments/wx3/wx3_multiset_induction_safe.json"
  "ext|project/evolve/experiments/wx3/wx3_multiset_ext_safe.json"
  "comb|project/evolve/experiments/wx3/wx3_multiset_combined_safe.json"
)

for SET in "${SETS[@]}"; do
  for C in "${CONFIGS[@]}"; do
    TAG="${C%%|*}"; GEN="${C##*|}"
    bash scripts/wx3_run_eval.sh "$TAG" "$GEN" "$SET" || echo "[FAIL] $TAG/$SET"
  done
done
echo "[matrix-done]"
