#!/usr/bin/env bash
# Overnight A+D experiment.
#
# What this runs:
#   Phase 1 — Deterministic beam-k=8 anchor for each of three checkpoints.
#             (gen_v5 t5-small, gen_ckpt_v6_premise t5-small + premise,
#              gen_v6 t5-base.) gen_v6 has never been evaluated before.
#   Phase 2 — Sampling decoding (temp=0.8, top-p=0.95, k=8) with 5 seeds
#             per checkpoint.  Gives variance bars on the small/premise/base
#             scaling story.
#   Phase 3 — Retriever quality probe over project_state.json proven theorems.
#   Phase 4 — Aggregate everything into SUMMARY.md for morning review.
#
# Each individual eval is independent and isolated — one failure does not
# stop the rest.  Logs go to <ROOT>/run.log.
#
# Usage (from repo root):
#   bash experiments/overnight.sh            # full run (~6-9 hours)
#   bash experiments/overnight.sh --smoke    # tiny dry-run on 2 theorems
#
# After it finishes:
#   cat experiments/overnight_<TIMESTAMP>/SUMMARY.md

set -uo pipefail
# Note: deliberately NOT using `set -e` — we want failed runs to be
# recorded but not kill the orchestrator.

cd "$(dirname "$0")/.."   # repo root regardless of cwd

DATE=$(date +%Y%m%d_%H%M%S)
ROOT="experiments/overnight_${DATE}"
mkdir -p "$ROOT"

# Tee all stdout/stderr to a log file in the run directory.
exec > >(tee -a "$ROOT/run.log") 2>&1

THEOREM_SET="curriculum_all"
TOP_K=8
MAX_STEPS=8
TEMPERATURE=0.8
SEEDS=(42 123 456 789 1024)
SMOKE=0

if [[ "${1:-}" == "--smoke" ]]; then
  echo "[smoke] Running tiny smoke test"
  echo "[smoke]   - 1 theorem (set_small)"
  echo "[smoke]   - all 3 checkpoints, beam + 1 seed each"
  echo "[smoke]   - confirms every load path before committing to overnight"
  SMOKE=1
  THEOREM_SET="set_small"  # 1 theorem
  SEEDS=(42)
fi

echo "============================================================"
echo "  OVERNIGHT A+D RUN"
echo "============================================================"
echo "  Started   : $(date)"
echo "  Root dir  : $ROOT"
echo "  Theorems  : $THEOREM_SET"
echo "  Top-k     : $TOP_K"
echo "  Max steps : $MAX_STEPS"
echo "  Sample T  : $TEMPERATURE"
echo "  Seeds     : ${SEEDS[*]}"
echo "============================================================"

# Checkpoints: tag -> (ckpt-dir, policy-type).
# We avoid `declare -A` because macOS still ships bash 3.2 by default,
# which does not support associative arrays. The case lookup below works
# in any bash >= 3.0.
TAGS="v5 premise base"

ckpt_for() {
  case "$1" in
    v5)      echo "project/models/gen_v5" ;;
    premise) echo "project/gen_ckpt_v6_premise" ;;
    base)    echo "project/models/gen_v6" ;;
    *)       echo "" ;;
  esac
}

policy_for() {
  case "$1" in
    v5|base) echo "generative" ;;
    premise) echo "premise_augmented" ;;
    *)       echo "" ;;
  esac
}

run_eval() {
  local label="$1"
  local out_dir="$2"
  shift 2
  echo ""
  echo "------------------------------------------------------------"
  echo "  $label"
  echo "  out: $out_dir"
  echo "------------------------------------------------------------"
  mkdir -p "$out_dir"
  if python eval_rollout_all.py "$@" --out-dir "$out_dir"; then
    echo "[ok] $label"
  else
    echo "[FAIL] $label (continuing)" >&2
  fi
}

# ------------------------------------------------------------------
# Phase 1: deterministic beam baseline for each checkpoint
# ------------------------------------------------------------------
for tag in $TAGS; do
  ckpt=$(ckpt_for "$tag")
  policy=$(policy_for "$tag")
  out="$ROOT/${tag}_beam"
  run_eval \
    "Phase 1: $tag (beam, deterministic anchor)" \
    "$out" \
    --theorem-set "$THEOREM_SET" \
    --ckpt-dir "$ckpt" \
    --policy-type "$policy" \
    --top-k "$TOP_K" \
    --max-steps "$MAX_STEPS" \
    --decode-mode beam
done

# ------------------------------------------------------------------
# Phase 2: 5 sampling seeds × 3 checkpoints
# ------------------------------------------------------------------
for tag in $TAGS; do
  ckpt=$(ckpt_for "$tag")
  policy=$(policy_for "$tag")
  for seed in "${SEEDS[@]}"; do
    out="$ROOT/${tag}_sample_seed${seed}"
    run_eval \
      "Phase 2: $tag sample seed=$seed" \
      "$out" \
      --theorem-set "$THEOREM_SET" \
      --ckpt-dir "$ckpt" \
      --policy-type "$policy" \
      --top-k "$TOP_K" \
      --max-steps "$MAX_STEPS" \
      --decode-mode sample \
      --temperature "$TEMPERATURE" \
      --seed "$seed"
  done
done

# ------------------------------------------------------------------
# Phase 3: retriever-quality probe (D)
# ------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  Phase 3: retriever quality probe"
echo "------------------------------------------------------------"
if python experiments/retriever_probe.py \
    --state project/project_state.json \
    --traces project/all_traces.jsonl \
    --premise-index project/premise_index.json \
    --out "$ROOT/retriever_probe.json"; then
  echo "[ok] retriever probe"
else
  echo "[FAIL] retriever probe (continuing)" >&2
fi

# ------------------------------------------------------------------
# Phase 4: morning summary
# ------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  Phase 4: summary"
echo "------------------------------------------------------------"
if python experiments/summarize_overnight.py \
    --root "$ROOT" \
    --out "$ROOT/SUMMARY.md"; then
  echo "[ok] summary written"
else
  echo "[FAIL] summary (check the run dirs manually)" >&2
fi

echo ""
echo "============================================================"
echo "  DONE  $(date)"
echo "  See: $ROOT/SUMMARY.md"
echo "============================================================"
