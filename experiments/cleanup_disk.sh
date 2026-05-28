#!/usr/bin/env bash
# Disk cleanup for dojo_sandbox.
#
# Defaults to DRY-RUN.  Shows exactly what would be deleted and the size of
# each target, so you can review before pulling the trigger.  Pass
# --yes-actually-delete to do the rm.
#
# What this deletes:
#   1. HuggingFace per-epoch intermediate checkpoints in every trained model
#      directory.  These are training-time bookkeeping; the final model is the
#      top-level model.safetensors and is preserved.
#   2. Old experimental classifier checkpoints under curriculum_runs/.
#   3. The 1-epoch smoke checkpoint (gen_v7_base_on_v5data_smoke).
#   4. Stray tmp*.jsonl files in project/.
#
# What this DOES NOT touch:
#   - gen_v7_base_on_v5data/  (currently being written by an active training
#     run; deleting would crash it).
#   - The final flat checkpoint files (model.safetensors, tokenizer*, configs)
#     in any gen_v*/ directory.
#   - all_traces.jsonl, project_state.json, seq2seq_data_v*.jsonl.
#   - run.log files, metrics.json files, traces.jsonl files.
#
# Usage:
#   bash experiments/cleanup_disk.sh                       # dry-run (default)
#   bash experiments/cleanup_disk.sh --yes-actually-delete # real

set -uo pipefail
cd "$(dirname "$0")/.."

DELETE=0
if [[ "${1:-}" == "--yes-actually-delete" ]]; then
  DELETE=1
fi

# Targets — each line is a glob.  Quoted to avoid premature expansion;
# we expand inside the loop with `compgen`.
TARGETS=(
  "project/models/gen_v1/checkpoint-*"
  "project/models/gen_v2/checkpoint-*"
  "project/models/gen_v3/checkpoint-*"
  "project/models/gen_v4/checkpoint-*"
  "project/models/gen_v5/checkpoint-*"
  "project/models/gen_v6/checkpoint-*"
  "project/gen_ckpt_v6_premise/checkpoint-*"
  "clf_ckpt/checkpoint-*"
  "curriculum_runs/tier1_ckpt"
  "curriculum_runs/merged_ckpt"
  "project/models/gen_v7_base_on_v5data_smoke"
  "project/tmp*.jsonl"
)

if [[ $DELETE -eq 1 ]]; then
  banner="ACTUALLY DELETING"
else
  banner="DRY RUN — nothing will be removed"
fi

echo "============================================================"
echo "  DOJO_SANDBOX DISK CLEANUP — $banner"
echo "============================================================"

# Refuse if the active capacity-isolation training is still writing.
GUARD="project/models/gen_v7_base_on_v5data"
if [[ -d "$GUARD" ]]; then
  age=$(( $(date +%s) - $(stat -f %m "$GUARD" 2>/dev/null || stat -c %Y "$GUARD") ))
  if (( age < 600 )); then
    echo ""
    echo "  WARNING: $GUARD was modified within the last 10 minutes."
    echo "           If a training run is in progress, deleting nothing"
    echo "           there is fine — this script never touches it.  Just"
    echo "           making sure you know."
  fi
fi

total_freed=0
for pattern in "${TARGETS[@]}"; do
  matches=()
  for m in $pattern; do
    [[ -e "$m" ]] && matches+=("$m")
  done
  if [[ ${#matches[@]} -eq 0 ]]; then
    continue
  fi
  for m in "${matches[@]}"; do
    size_h=$(du -sh -- "$m" 2>/dev/null | awk '{print $1}')
    size_kb=$(du -sk -- "$m" 2>/dev/null | awk '{print $1}')
    if [[ -n "$size_kb" ]]; then
      total_freed=$(( total_freed + size_kb ))
    fi
    if [[ $DELETE -eq 1 ]]; then
      printf "  rm -rf  %-8s  %s\n" "$size_h" "$m"
      rm -rf -- "$m"
    else
      printf "  WOULD   %-8s  %s\n" "$size_h" "$m"
    fi
  done
done

human_total=$(awk -v kb="$total_freed" 'BEGIN {
  units = "KB MB GB TB"; split(units, u, " ")
  i = 1; n = kb
  while (n >= 1024 && i < 4) { n /= 1024; i++ }
  printf "%.1f %s", n, u[i]
}')

echo ""
echo "------------------------------------------------------------"
if [[ $DELETE -eq 1 ]]; then
  echo "  Freed approximately: $human_total"
else
  echo "  Would free approximately: $human_total"
  echo ""
  echo "  Re-run with --yes-actually-delete to do it for real."
fi
echo "============================================================"

# Optional: extra hints for caches outside the project dir
if [[ $DELETE -eq 0 ]]; then
  echo ""
  echo "  Extra space sometimes hides outside the project dir.  Check yourself:"
  echo "    du -sh ~/.cache/huggingface ~/.cache/lean_dojo 2>/dev/null"
  echo "  Both are safe to delete; HuggingFace re-downloads on demand,"
  echo "  LeanDojo rebuilds its mathlib trace cache when next needed."
fi
