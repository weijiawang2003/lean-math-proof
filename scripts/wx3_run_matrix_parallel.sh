#!/usr/bin/env bash
# WX3 Stage 6 — parallel evaluation matrix (configs x Multiset sets).
# Idempotent per-run (wx3_run_eval.sh skips if metrics.json exists).
# Concurrency via xargs -P; each job loads its own model (default route =
# gen_v5_ns12_balanced for Multiset). Default 4-way on a 12-core box.
set -uo pipefail
cd "$(dirname "$0")/.."
PAR="${1:-4}"

SETS=(
  wx3_multiset_simp_easy
  wx3_multiset_induction_easy
  wx3_multiset_ext_medium
  wx3_multiset_quotient_medium
  wx3_multiset_mixed
)
CONFIGS=(
  "raw|NONE"
  "ns9|project/evolve/best/ns9_best_genome.json"
  "ind|project/evolve/experiments/wx3/wx3_multiset_induction_safe.json"
  "ext|project/evolve/experiments/wx3/wx3_multiset_ext_safe.json"
  "comb|project/evolve/experiments/wx3/wx3_multiset_combined_safe.json"
)

JOBS=()
for SET in "${SETS[@]}"; do
  for C in "${CONFIGS[@]}"; do
    JOBS+=("${C%%|*} ${C##*|} ${SET}")
  done
done

printf '%s\n' "${JOBS[@]}" | xargs -P "$PAR" -I {} bash -c \
  'set -- {}; bash scripts/wx3_run_eval.sh "$1" "$2" "$3" || echo "[FAIL] $1/$3"'
echo "[matrix-parallel-done]"
