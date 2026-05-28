#!/usr/bin/env bash
# Capacity-isolation experiment.
#
# Question: is t5-base's 22/30 vs t5-small's 25/30 a CAPACITY problem
# (model too big and under-trained) or a DATA SHIFT problem (idiom-frequency
# dilution in v6 training pool)?
#
# Test: train t5-base on EXACTLY v5's training pool (project/seq2seq_data_v5.jsonl,
# 5577 examples, same 93-tactic vocabulary) with everything else identical to
# v5's training (15 epochs, lr=5e-5, batch=8, seed=42).  Then eval on
# curriculum_all and compare.
#
# Interpretation:
#   gen_v7 ≥ 25/30  → capacity helps; v6's regression was data-shift only.
#                     Phase 2 hypothesis is rescued by retraining on v5 pool.
#   gen_v7 in [22, 24]  → capacity gives marginal lift over v5; data shift
#                         was net-bad but capacity didn't fully compensate.
#   gen_v7 < 22/30  → capacity actively hurts on this curriculum.  The
#                     small-model-with-frequent-idioms is genuinely the
#                     better operating point.
#
# Either answer is a clean finding that the project couldn't have made before.
#
# Usage:
#   bash experiments/capacity_isolation.sh           # full train + eval
#   bash experiments/capacity_isolation.sh --smoke   # 1 epoch, sanity only

set -uo pipefail
cd "$(dirname "$0")/.."

DATE=$(date +%Y%m%d_%H%M%S)
ROOT="experiments/capacity_isolation_${DATE}"
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/run.log") 2>&1

DATA="project/seq2seq_data_v5.jsonl"     # v5's exact training pool
MODEL_NAME="t5-base"
OUT_DIR="project/models/gen_v7_base_on_v5data"
EPOCHS=15
SEED=42
LR=5e-5
BATCH=8
VAL_SPLIT=0.1

if [[ "${1:-}" == "--smoke" ]]; then
  echo "[smoke] 1-epoch training run, then eval on set_small (1 theorem)"
  EPOCHS=1
  OUT_DIR="${OUT_DIR}_smoke"
  EVAL_SET="set_small"
else
  EVAL_SET="curriculum_all"
fi

echo "============================================================"
echo "  CAPACITY ISOLATION EXPERIMENT"
echo "============================================================"
echo "  Started   : $(date)"
echo "  Root dir  : $ROOT"
echo "  Data      : $DATA  (v5's exact pool)"
echo "  Model     : $MODEL_NAME"
echo "  Epochs    : $EPOCHS"
echo "  Output    : $OUT_DIR"
echo "  Eval set  : $EVAL_SET"
echo "============================================================"

# Refuse to silently overwrite an existing checkpoint
if [[ -d "$OUT_DIR" && "${1:-}" != "--smoke" ]]; then
  echo "[ERROR] $OUT_DIR already exists.  Move or remove it before retrying."
  echo "        (Smoke runs use a separate _smoke suffix, so this only fires"
  echo "         on the real run.)"
  exit 1
fi

# ------------------------------------------------------------------
# Phase 1: train t5-base on v5 data
# ------------------------------------------------------------------
echo ""
echo "--- Phase 1: train ---"
if python train_tactic_generator.py \
    --data "$DATA" \
    --model "$MODEL_NAME" \
    --output-dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --lr "$LR" \
    --seed "$SEED" \
    --val-split "$VAL_SPLIT"; then
  echo "[ok] training complete  → $OUT_DIR"
else
  echo "[FAIL] training did not complete.  Aborting eval." >&2
  exit 2
fi

# ------------------------------------------------------------------
# Phase 2: eval on curriculum_all (or set_small for smoke)
# ------------------------------------------------------------------
echo ""
echo "--- Phase 2: eval ---"
EVAL_OUT="$ROOT/eval"
if python eval_rollout_all.py \
    --theorem-set "$EVAL_SET" \
    --ckpt-dir "$OUT_DIR" \
    --policy-type generative \
    --top-k 8 \
    --max-steps 8 \
    --decode-mode beam \
    --out-dir "$EVAL_OUT"; then
  echo "[ok] eval complete"
else
  echo "[FAIL] eval did not complete." >&2
  exit 3
fi

# ------------------------------------------------------------------
# Phase 3: small report
# ------------------------------------------------------------------
echo ""
echo "--- Phase 3: report ---"
python3 << PYEOF
import glob, json
paths = sorted(glob.glob("$EVAL_OUT/eval-*/metrics.json"))
if not paths:
    print("[report] No metrics found, cannot summarize.")
    exit(0)
m = json.loads(open(paths[0]).read())
proved = m.get("proved", "?")
avail  = m.get("available", "?")
print()
print("============================================================")
print("  CAPACITY ISOLATION — RESULT")
print("============================================================")
print(f"  gen_v7 (t5-base on v5 data, $EPOCHS epochs): {proved}/{avail}")
print()
print("  Reference:")
print(f"    gen_v5  (t5-small on v5 data):          25/30")
print(f"    gen_v6_premise (t5-small + premise):    19/30")
print(f"    gen_v6  (t5-base on v6 data):           22/30")
print()
if isinstance(proved, int):
    if proved >= 25:
        print("  → Capacity helps. v6's regression was data-shift only.")
        print("    Phase 2 hypothesis is rescued; the right move is to")
        print("    retrain on idiom-frequency-preserving data pools.")
    elif proved >= 22:
        print("  → Capacity gives marginal lift over v5 but doesn't fully")
        print("    compensate for the v6 dataset's idiom dilution.")
    else:
        print("  → Capacity actively hurts on this curriculum.  The small-")
        print("    model-with-frequent-idioms is the better operating point")
        print("    for this benchmark, regardless of training data.")
print("============================================================")
PYEOF

echo ""
echo "Done $(date).  Artifacts under $ROOT/"
