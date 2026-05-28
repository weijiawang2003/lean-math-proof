#!/usr/bin/env bash
# Overnight v9 sweep — ~6 hours of compute.
#
# Tests two ways to break the idiom-dilution mechanism that bit gen_v8:
#   - Path A: fine-tune from gen_v7 on frontier-only proofs (gen_v9_ft).
#             Hypothesis: starting from a model that already knows the
#             curriculum and only training on the new patterns avoids
#             ejecting old idioms.
#   - Path B: full retrain t5-base on v5 data + 50× upsampled frontier
#             (gen_v9_full).  Hypothesis: enough upsampling crosses the
#             dilution-floor that bit gen_v8 (which had 2.1% frontier).
#
# Then variance bars (5 sampling seeds) on the four headline checkpoints
# (gen_v5, gen_v7, gen_v9_ft, gen_v9_full) to confirm whatever beam result
# emerges is not a single-seed artifact.
#
# Phases (estimated wall-clock on M4 Pro):
#   1. search_v6 on frontier_v1                                  ~15 min
#   2. build augmented training pool (50x upsampling)            ~5 min
#   3a. fine-tune gen_v7 → gen_v9_ft (5 epochs, frontier only)   ~30 min
#   3b. full retrain → gen_v9_full (15 epochs, v5 + 50x frontier) ~3 hours
#   4. eval gen_v9_ft and gen_v9_full (beam k=8)                 ~20 min
#   5. variance bars: 4 ckpts × 5 seeds                          ~2 hours
#   6. auto-summary
#
# Total: ~5.5-6 hours.
#
# Crash-safe: each phase tolerates failures.  If Phase 3a fails we still
# attempt 3b; if 3b fails we still run variance on what we have.
#
# Usage:
#   caffeinate -i bash experiments/overnight_v9_sweep.sh
#
# Resume in the morning:
#   cat experiments/overnight_v9_<TIMESTAMP>/SUMMARY.md

set -uo pipefail
cd "$(dirname "$0")/.."

DATE=$(date +%Y%m%d_%H%M%S)
ROOT="experiments/overnight_v9_${DATE}"
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/run.log") 2>&1

THEOREM_SET="curriculum_all"
TOP_K=8
MAX_STEPS=8
TEMPERATURE=0.8
SEEDS=(42 123 456 789 1024)

echo "============================================================"
echo "  OVERNIGHT V9 SWEEP"
echo "============================================================"
echo "  Started   : $(date)"
echo "  Root dir  : $ROOT"
echo "============================================================"

# Disk + safety preflight
if [[ ! -f train_tactic_generator.py ]]; then
  echo "[FATAL] not in repo root" >&2; exit 1
fi
if ! grep -q "save_total_limit=2" train_tactic_generator.py; then
  echo "[FATAL] save_total_limit=2 missing from train_tactic_generator.py" >&2
  echo "        previous disk-fill failure mode is back; refusing to launch" >&2
  exit 1
fi
echo "[ok] save_total_limit=2 confirmed in trainer"

# ============================================================
# Phase 1: search_v6 on frontier
# ============================================================
echo ""
echo "------------------------------------------------------------"
echo "  Phase 1: search_v6 on frontier_v1"
echo "------------------------------------------------------------"
SEARCH_OUT="$ROOT/search_v6"
mkdir -p "$SEARCH_OUT"
if python search_generate_traces.py \
    --theorem-set frontier_v1 \
    --beam-width 32 \
    --max-depth 8 \
    --action-space search_v6 \
    --out "$SEARCH_OUT/traces.jsonl" \
    --out-dir "$SEARCH_OUT"; then
  echo "[ok] search_v6 complete"
else
  echo "[FAIL] search_v6 failed (continuing — will detect empty traces)" >&2
fi

n_traces=$(wc -l < "$SEARCH_OUT/traces.jsonl" 2>/dev/null | tr -d ' ' || echo 0)
echo "Search produced $n_traces transitions."

# ============================================================
# Phase 2: build augmented training pool
# ============================================================
echo ""
echo "------------------------------------------------------------"
echo "  Phase 2: build augmented pool (50x upsampling)"
echo "------------------------------------------------------------"
DATA_OUT="$ROOT/data"
mkdir -p "$DATA_OUT"

python build_seq2seq_dataset.py \
  --in "$SEARCH_OUT/traces.jsonl" \
  --out "$DATA_OUT/frontier_seq2seq.jsonl" \
  --min-goal-drop 1 \
  || echo "[WARN] build_seq2seq_dataset returned non-zero" >&2

n_frontier=$(wc -l < "$DATA_OUT/frontier_seq2seq.jsonl" 2>/dev/null | tr -d ' ' || echo 0)
echo "Frontier seq2seq pairs: $n_frontier"

if [[ "$n_frontier" -eq 0 ]]; then
  echo "[FATAL] no frontier pairs produced; cannot retrain.  Skipping to Phase 5." >&2
  SKIP_RETRAIN=1
else
  SKIP_RETRAIN=0
  # 50x upsample
  python3 -c "
from pathlib import Path
src = Path('$DATA_OUT/frontier_seq2seq.jsonl').read_text()
Path('$DATA_OUT/frontier_seq2seq_50x.jsonl').write_text(src * 50)
"
  cat project/seq2seq_data_v5.jsonl "$DATA_OUT/frontier_seq2seq_50x.jsonl" \
    > "$DATA_OUT/v5_plus_frontier_50x.jsonl"

  echo ""
  echo "  Pool sizes:"
  echo "    v5 baseline:       $(wc -l < project/seq2seq_data_v5.jsonl | tr -d ' ') examples"
  echo "    50x frontier:      $(wc -l < "$DATA_OUT/frontier_seq2seq_50x.jsonl" | tr -d ' ') examples"
  echo "    combined:          $(wc -l < "$DATA_OUT/v5_plus_frontier_50x.jsonl" | tr -d ' ') examples"
fi

# ============================================================
# Phase 3a: fine-tune gen_v7 → gen_v9_ft (frontier-only)
# ============================================================
GEN_V9_FT="project/models/gen_v9_ft"
if [[ "$SKIP_RETRAIN" -eq 0 ]]; then
  echo ""
  echo "------------------------------------------------------------"
  echo "  Phase 3a: fine-tune gen_v7 → gen_v9_ft (frontier-only)"
  echo "------------------------------------------------------------"
  rm -rf "$GEN_V9_FT"
  if python train_tactic_generator.py \
      --data "$DATA_OUT/frontier_seq2seq_50x.jsonl" \
      --model project/models/gen_v7_base_on_v5data \
      --output-dir "$GEN_V9_FT" \
      --epochs 5 \
      --batch-size 8 \
      --lr 2e-5 \
      --seed 42 \
      --val-split 0.1; then
    echo "[ok] gen_v9_ft trained"
  else
    echo "[FAIL] gen_v9_ft training failed" >&2
  fi
else
  echo "[SKIP] Phase 3a — no frontier data"
fi

# ============================================================
# Phase 3b: full retrain t5-base → gen_v9_full
# ============================================================
GEN_V9_FULL="project/models/gen_v9_full"
if [[ "$SKIP_RETRAIN" -eq 0 ]]; then
  echo ""
  echo "------------------------------------------------------------"
  echo "  Phase 3b: full retrain t5-base → gen_v9_full"
  echo "------------------------------------------------------------"
  rm -rf "$GEN_V9_FULL"
  if python train_tactic_generator.py \
      --data "$DATA_OUT/v5_plus_frontier_50x.jsonl" \
      --model t5-base \
      --output-dir "$GEN_V9_FULL" \
      --epochs 15 \
      --batch-size 8 \
      --lr 5e-5 \
      --seed 42 \
      --val-split 0.1; then
    echo "[ok] gen_v9_full trained"
  else
    echo "[FAIL] gen_v9_full training failed" >&2
  fi
else
  echo "[SKIP] Phase 3b — no frontier data"
fi

# ============================================================
# Phase 4: eval gen_v9_ft and gen_v9_full (beam)
# ============================================================
for tag in v9_ft v9_full; do
  ckpt="project/models/gen_${tag}"
  out="$ROOT/eval_${tag}_beam"
  if [[ -d "$ckpt" ]]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "  Phase 4: eval gen_${tag} (beam)"
    echo "------------------------------------------------------------"
    if python eval_rollout_all.py \
        --theorem-set "$THEOREM_SET" \
        --ckpt-dir "$ckpt" \
        --policy-type generative \
        --top-k "$TOP_K" --max-steps "$MAX_STEPS" \
        --decode-mode beam \
        --out-dir "$out"; then
      echo "[ok] eval $tag complete"
    else
      echo "[FAIL] eval $tag" >&2
    fi
  else
    echo "[SKIP] eval $tag — checkpoint $ckpt not found"
  fi
done

# ============================================================
# Phase 5: variance bars (sampling, 5 seeds × 4 checkpoints)
# ============================================================
echo ""
echo "------------------------------------------------------------"
echo "  Phase 5: variance bars (sampling)"
echo "------------------------------------------------------------"

ckpt_for() {
  case "$1" in
    v5)      echo "project/models/gen_v5" ;;
    v7)      echo "project/models/gen_v7_base_on_v5data" ;;
    v9_ft)   echo "project/models/gen_v9_ft" ;;
    v9_full) echo "project/models/gen_v9_full" ;;
    *)       echo "" ;;
  esac
}

for tag in v5 v7 v9_ft v9_full; do
  ckpt=$(ckpt_for "$tag")
  if [[ ! -d "$ckpt" ]]; then
    echo "[SKIP] variance for $tag — $ckpt not found"
    continue
  fi
  for seed in "${SEEDS[@]}"; do
    out="$ROOT/sample_${tag}_seed${seed}"
    echo ""
    echo "  variance: $tag seed=$seed"
    if python eval_rollout_all.py \
        --theorem-set "$THEOREM_SET" \
        --ckpt-dir "$ckpt" \
        --policy-type generative \
        --top-k "$TOP_K" --max-steps "$MAX_STEPS" \
        --decode-mode sample \
        --temperature "$TEMPERATURE" --seed "$seed" \
        --out-dir "$out"; then
      echo "  [ok] $tag seed=$seed"
    else
      echo "  [FAIL] $tag seed=$seed (continuing)" >&2
    fi
  done
done

# ============================================================
# Phase 6: summary
# ============================================================
echo ""
echo "------------------------------------------------------------"
echo "  Phase 6: generating SUMMARY.md"
echo "------------------------------------------------------------"

python3 << PYEOF > "$ROOT/SUMMARY.md"
import json, glob, math
from pathlib import Path

ROOT = Path("$ROOT")

def load_metrics(p):
    matches = sorted(p.glob("eval-*/metrics.json"))
    if not matches: return None
    try:
        return json.loads(matches[0].read_text())
    except Exception:
        return None

def stats(vals):
    if not vals: return float('nan'), float('nan')
    m = sum(vals) / len(vals)
    if len(vals) < 2: return m, 0.0
    var = sum((v-m)**2 for v in vals) / (len(vals)-1)
    return m, math.sqrt(var)

print("# Overnight v9 Sweep — Summary")
print()
print(f"Run dir: \`{ROOT}\`")
print()

# ---- Phase 1: Search ----
search_metrics = (ROOT / "search_v6" / "metrics.json")
n_traces = 0
search_path = ROOT / "search_v6" / "traces.jsonl"
if search_path.exists():
    n_traces = sum(1 for _ in search_path.open())
print("## Phase 1 — search_v6 on frontier_v1")
print()
print(f"- Transitions logged: {n_traces}")
n_frontier_pairs_path = ROOT / "data" / "frontier_seq2seq.jsonl"
if n_frontier_pairs_path.exists():
    n_pairs = sum(1 for _ in n_frontier_pairs_path.open())
    print(f"- Seq2seq pairs after filtering: {n_pairs}")
    print(f"- After 50x upsampling: {n_pairs * 50}")
print()

# ---- Phase 4: Beam evals ----
print("## Phase 4 — Beam evals on curriculum_all")
print()
print("| Checkpoint | Score | Notes |")
print("|---|---|---|")
print("| gen_v5 (anchor)   | 25/30 | t5-small, v5 data — historical SOTA |")
print("| gen_v7 (anchor)   | 24/30 | t5-base, v5 data — capacity isolation |")
for tag in ['v9_ft', 'v9_full']:
    m = load_metrics(ROOT / f"eval_{tag}_beam")
    if m:
        proved = m.get('proved', '?')
        avail = m.get('available', '?')
        notes = ""
        if isinstance(proved, int) and isinstance(avail, int):
            if proved >= 27:
                notes = "**STRONG SUCCESS — frontier loop works**"
            elif proved > 24:
                notes = "lifts above gen_v7 — partial success"
            elif proved == 24:
                notes = "no movement vs gen_v7"
            else:
                notes = "regression vs gen_v7"
        print(f"| gen_{tag}        | {proved}/{avail} | {notes} |")
    else:
        print(f"| gen_{tag}        | (not produced) | training or eval failed |")
print()

# ---- Phase 5: Variance bars ----
print("## Phase 5 — Variance bars (sampling, temp=0.8, top-p=0.95, k=8)")
print()
print("| Checkpoint | Mean ± std | Range | N seeds |")
print("|---|---|---|---|")
for tag in ['v5', 'v7', 'v9_ft', 'v9_full']:
    samples = []
    for seed in [42, 123, 456, 789, 1024]:
        m = load_metrics(ROOT / f"sample_{tag}_seed{seed}")
        if m:
            p = m.get('proved')
            if isinstance(p, int):
                samples.append(p)
    if samples:
        mean, std = stats([float(x) for x in samples])
        rng = f"{min(samples)}-{max(samples)}"
        print(f"| gen_{tag} | {mean:.1f} ± {std:.1f} | {rng} | {len(samples)} |")
    else:
        print(f"| gen_{tag} | — | — | 0 |")
print()

# ---- Hypotheses ----
print("## Hypotheses tested")
print()
print("**H1 (fine-tune avoids dilution):** Fine-tuning gen_v7 on frontier-only")
print("data should preserve gen_v7's curriculum solves while adding the new")
print("ones. → check gen_v9_ft beam score: ≥25/30 supports H1.")
print()
print("**H2 (50x upsampling crosses dilution floor):** Full retrain with")
print("frontier upsampled to ~9% of pool should override v5 priors enough")
print("to keep frontier proofs in beam. → check gen_v9_full beam score:")
print("≥27/30 supports H2.")
print()
print("**H3 (variance bars):** The 24 vs 25 gap between gen_v7 and gen_v5")
print("should be small under sampling noise.  If it's swamped by std,")
print("the entire capacity-isolation finding needs caveating.")
print()

# ---- Decision tree ----
print("## What to do next (decision tree)")
print()
print("- **gen_v9_ft ≥ 27/30:** fine-tuning is the right pattern;")
print("  iterate the loop — search next batch of frontier theorems and fine-tune again.")
print("- **gen_v9_full ≥ 27/30:** upsampling-with-full-retrain works;")
print("  more expensive but cleanly comparable to v7.")
print("- **Both at 24-25:** dilution mechanism is even more robust than thought;")
print("  next move is the strategic-policy ablation, not more data manipulation.")
print("- **Net regression:** something specific went wrong in one of the runs;")
print("  inspect run.log + per-theorem diffs.")
print()
print("Per-theorem disposition matrix and idiom-frequency comparison should be")
print("done as a morning analysis pass via Claude Code.")
PYEOF

echo ""
echo "============================================================"
echo "  DONE  $(date)"
echo "  See: $ROOT/SUMMARY.md"
echo "============================================================"
