# Step B — Retrain with Frontier Proofs (gen_v8)

> Claude Code brief. Read this whole document before doing anything.
> **Prerequisite:** Step A must have completed successfully. Read its
> findings doc at
> `experiments/search_frontier_v5_<TIMESTAMP>/FINDINGS.md` (the most
> recent one) before starting. If Step A did NOT find ≥3 frontier proofs,
> stop — the premise of this brief is broken.

## What we know

Step A confirmed (or refuted) that beam-search with `search_v5` action
space finds the frontier proofs. If you're here, it confirmed at least
3 of them. The next question is whether folding those proofs into the
training pool and retraining lifts the *model's* curriculum score on
rollout — beam finding proofs doesn't help the model unless the model
sees them in training.

## Goal

Train a new t5-base checkpoint (`gen_v8`) on the same data as `gen_v7`
plus the frontier proofs from Step A. Eval on `curriculum_all`. Compare
to gen_v7's 24/30. The expected lift is +3 to +5 theorems if the model
absorbs the new idioms via fine-tuning.

This is the first experiment in the project where new search-found
proofs feed back into training — a small, controlled instance of the
expert-iteration loop.

## Concrete plan

### Phase 1 — locate Step A's traces

Find the most recent `experiments/search_frontier_v5_*/` directory. The
search results live at `<that_dir>/traces.jsonl` (a TransitionRecord
JSONL produced by `search_generate_traces.py`).

Sanity-check the file:
- Count rows: `wc -l <traces_path>`
- Filter to `proof_finished: true` rows. These are the closing tactics
  for the frontier proofs Step A found.
- Verify there are state→tactic pairs for each frontier theorem Step A
  reported as solved.

If the file is empty or doesn't contain the expected proofs, stop and
report — something went wrong between Step A's findings and the trace
file.

### Phase 2 — build the augmented seq2seq dataset

The training pipeline uses seq2seq pairs (`state_pp` → `tactic`), built
by `build_seq2seq_dataset.py` from raw transition traces. The existing
`gen_v5`/`gen_v7` data pool is `project/seq2seq_data_v5.jsonl`.

Build a new dataset that includes both pools:

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
ROOT="experiments/gen_v8_$STAMP"
mkdir -p "$ROOT"

# Convert Step A's traces to seq2seq pairs
python build_seq2seq_dataset.py \
  --in <STEP_A_TRACES_PATH> \
  --out "$ROOT/frontier_seq2seq.jsonl" \
  --min-goal-drop 1

# Concatenate v5 pool + frontier pool (additive — original pool unchanged)
cat project/seq2seq_data_v5.jsonl "$ROOT/frontier_seq2seq.jsonl" > "$ROOT/seq2seq_v5_plus_frontier.jsonl"

# Sanity check
echo "v5 pool:       $(wc -l < project/seq2seq_data_v5.jsonl) examples"
echo "frontier add:  $(wc -l < "$ROOT/frontier_seq2seq.jsonl") examples"
echo "combined:      $(wc -l < "$ROOT/seq2seq_v5_plus_frontier.jsonl") examples"
```

The frontier addition will be small — likely 5-15 new pairs, since each
proof is 1-2 tactics. That's fine; the goal isn't bulk, it's exposing
the model to the *kind* of tactic (`Iff.rfl`, `rfl`) that closes
definitional goals.

### Phase 3 — train gen_v8

```bash
caffeinate -i python train_tactic_generator.py \
  --data "$ROOT/seq2seq_v5_plus_frontier.jsonl" \
  --model t5-base \
  --output-dir project/models/gen_v8 \
  --epochs 15 \
  --batch-size 8 \
  --lr 5e-5 \
  --seed 42 \
  --val-split 0.1
```

Identical hyperparameters to `gen_v7` so the ONLY difference between
gen_v7 and gen_v8 is the inclusion of the frontier proofs in the pool.
Disk: should stay under 10 GB total because `save_total_limit=2` is in
place.

Expected runtime: ~3 hours on M4 Pro. Monitor the first epoch to
confirm checkpoint-628 saves cleanly (~12 min in) — that's the
disk-fill failure mode. After that, leave it.

### Phase 4 — eval gen_v8

```bash
python eval_rollout_all.py \
  --theorem-set curriculum_all \
  --ckpt-dir project/models/gen_v8 \
  --policy-type generative \
  --top-k 8 --max-steps 8 \
  --decode-mode beam \
  --out-dir "$ROOT/eval"
```

### Phase 5 — write findings

Produce `experiments/gen_v8_<TIMESTAMP>/FINDINGS.md` with:

- Headline: gen_v8 score / 30, vs gen_v7's 24/30 baseline.
- Per-theorem disposition diff: which theorems v7 missed are now
  solved by v8? Which (if any) does v8 lose vs v7? Specifically check
  the four frontier theorems that Step A solved — did v8 actually use
  the new idioms (`Iff.rfl`, `rfl`) on rollout?
- Tactic emission analysis: for each frontier theorem, what did v8
  emit at top-1, top-2, etc.? Did `Iff.rfl` make it into the model's
  beam output?
- One-paragraph synthesis:
  - **+3 or more on the frontier:** expert-iteration loop works at
    this scale. Recommend repeating with the next batch of unsolved
    theorems.
  - **+1 to +2 on the frontier:** partial transfer. Some idioms made
    it; others didn't. Investigate which.
  - **0 on the frontier (no movement):** the new pairs were too few
    or the model didn't learn from them. Recommend upsampling them
    in the training data (e.g., 10× duplication) and retraining.
  - **Net regression (gen_v8 < gen_v7):** something went wrong.
    Likely the small number of new pairs perturbed training
    unhelpfully. Investigate.

## Constraints

You may:
- Read any file in the repo.
- Run `build_seq2seq_dataset.py`, `train_tactic_generator.py`,
  `eval_rollout_all.py` with the parameters above.
- Write files under `experiments/gen_v8_*/` and to
  `project/models/gen_v8/` (new directory; nothing existing lives
  there).
- Concatenate JSONL files into a new combined file (do NOT modify
  `project/seq2seq_data_v5.jsonl` itself).

You should NOT:
- Modify `train_tactic_generator.py`, `build_seq2seq_dataset.py`,
  `eval_rollout_all.py`, or any other canonical script.
- Overwrite `project/models/gen_v7_base_on_v5data` or any other
  existing checkpoint.
- Modify `project/seq2seq_data_v5.jsonl` in place — only create new
  combined files in the experiment directory.
- Modify `project/project_state.json` or `project/all_traces.jsonl`.
- Edit MEMORY.md or anything under `.claude/`.

## Acceptance criteria

- **gen_v8 ≥ 27/30:** strong success. Frontier-proof-feedback works
  on this pipeline at this scale. Project's first concrete
  expert-iteration result.
- **gen_v8 in 25-26/30:** partial success. Some frontier theorems
  transferred, others didn't. Worth analyzing.
- **gen_v8 = 24/30 (no change):** the frontier pairs were too few
  to move training. Recommend upsampling.
- **gen_v8 < 24/30:** regression. Something specific went wrong.
  Investigate.

In all cases, write the findings doc with concrete numbers and the
per-theorem disposition matrix.

## Disk safety

- Confirm `save_total_limit=2` is set in `train_tactic_generator.py`
  before launching (search the file for "save_total_limit"). If it's
  not there, STOP and surface the discrepancy — the prior disk-fill
  bite happened because this was missing. Do not proceed without it.
- Keep `df -h /` checked at the end of training; if free space is
  under 30 GB, flag it but don't intervene.

## Style

- Concrete > ceremonial.
- Quote actual numbers and tactics.
- If gen_v8 doesn't move from 24/30, say so plainly and propose the
  next debug step (likely upsampling the frontier pairs).

When done, print the path to FINDINGS.md and a 3-line headline.
