#!/usr/bin/env bash
# AX3 Stage 3 — parallel eval matrix: raw/ns9/wx3ind x 4 AX3 sets.
# Idempotent per run; xargs -P concurrency (default 4).
set -uo pipefail
cd "$(dirname "$0")/.."
PAR="${1:-4}"
SETS=(ax3_multiset_induction_mine ax3_multiset_induction_heldout
      ax3_multiset_mixed_heldout ax3_multiset_negative_control)
CONFIGS=(
  "raw|NONE"
  "ns9|project/evolve/best/ns9_best_genome.json"
  "wx3ind|project/evolve/experiments/wx3/wx3_multiset_induction_safe.json"
)
JOBS=()
for SET in "${SETS[@]}"; do
  for C in "${CONFIGS[@]}"; do JOBS+=("${C%%|*} ${C##*|} ${SET}"); done
done
printf '%s\n' "${JOBS[@]}" | xargs -P "$PAR" -I {} bash -c \
  'set -- {}; bash scripts/ax3_run_eval.sh "$1" "$2" "$3" || echo "[FAIL] $1/$3"'
echo "[ax3-matrix-done]"
