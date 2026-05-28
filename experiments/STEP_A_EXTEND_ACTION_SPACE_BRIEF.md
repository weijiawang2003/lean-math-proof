# Step A — Extend Action Space and Re-Search the Frontier

> Claude Code brief. Read this whole document before doing anything. Builds
> on the prior search-frontier finding at
> `experiments/search_frontier_20260502_170824/FINDINGS.md` — read that
> first if not already.

## What we know

The prior beam-search-on-the-frontier task found 1/5 proofs. The four
unsolved evaluable Finset theorems all have proofs that are literally
`Iff.rfl` (or `rfl`) in Lean 4 — they are definitional unfoldings, e.g.
`Finset.mem_insert : a ∈ insert b s ↔ a = b ∨ a ∈ s` is true *by
definition* of `insert`. Beam search didn't find them because the
`search_v4` action space doesn't contain `Iff.rfl` or `rfl` as candidates.

## Goal

Add a new action space `search_v5` that extends `search_v4` with the
missing definitional-proof tactics. Re-run beam search on `frontier_v1`.
Confirm whether the four evaluable Finset theorems get solved.

This is a falsifiable check on the diagnosis. If beam finds them with the
extended action space, the diagnosis is right and Step B (retraining)
becomes worth the 3-hour cost. If beam still doesn't find them, the
diagnosis is incomplete and we need to investigate further before
committing to retraining.

## Concrete plan

### Phase 1 — define `search_v5`

Edit `actions.py`. Find the existing `search_v4` action space definition
(it'll be a list literal in some `ACTION_SPACES` registry or similar).
**Do not modify `search_v4`** — existing classifier checkpoints (`clf_ckpt`,
older runs) may depend on its exact contents. Add a new entry,
`search_v5`, that contains everything in `search_v4` plus these four
candidates at the end:

```
"rfl",
"Iff.rfl",
"exact Iff.rfl",
"exact rfl",
```

Make sure `search_v5` is registered in whatever lookup function
`get_action_space()` uses, so `--action-space search_v5` resolves
correctly.

### Phase 2 — re-run search

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
ROOT="experiments/search_frontier_v5_$STAMP"
mkdir -p "$ROOT"
python search_generate_traces.py \
  --theorem-set frontier_v1 \
  --beam-width 32 \
  --max-depth 8 \
  --action-space search_v5 \
  --out "$ROOT/traces.jsonl" \
  --out-dir "$ROOT"
```

Expected runtime ~5-15 min — much faster than the prior search because
the right tactic is at depth 1 if the diagnosis is correct.

### Phase 3 — categorize results

For each of the 5 theorems in `frontier_v1`, classify the outcome from
`metrics.json` and `traces.jsonl`:

- **Solved** (record the proof — usually 1-2 tactics).
- **Still stuck** (record what beam tried and why it failed).
- **Unavailable** (`Finset.insert_comm` was unavailable last time — same
  result expected, just note it).

If beam finds 4/5 (the 4 evaluable Finset theorems plus `Nat.mul_add_mod'`
which carries over from the prior `search_v4` run via the same action set),
the diagnosis is confirmed.

### Phase 4 — write findings

Produce `experiments/search_frontier_v5_<TIMESTAMP>/FINDINGS.md` with:

- Per-theorem result (table: theorem, solved?, proof if yes, depth).
- The `Iff.rfl` / `rfl` tactic count: how often did beam reach for these
  tactics, and on which theorems did they close the goal?
- Confirmation status: does this match the prediction (4/4 evaluable
  solved)? If yes, Step B is unblocked. If no, dig in.
- If the result is mixed (e.g., 2 of 4 Finset solved), characterize the
  remaining failures.

## Constraints

You may:
- Edit `actions.py` to add `search_v5` (additive only — do not modify
  existing action spaces).
- Read any file in the repo.
- Run `search_generate_traces.py` with the parameters above.
- Write new files under `experiments/search_frontier_v5_*/`.

You should NOT:
- Modify `search_v4` or any other existing action space.
- Modify `search_generate_traces.py`, `env.py`, or any training script.
- Re-train any model.
- Modify `project/project_state.json`, `project/all_traces.jsonl`, or any
  `seq2seq_data_v*.jsonl`.
- Add the search-found traces to the training pool yourself — that's
  Step B's job.

## Acceptance criteria

- **4/4 evaluable solved (Iff.rfl / rfl found at depth 1-2):** diagnosis
  confirmed, Step B unblocked.
- **2-3/4 solved:** partial confirmation; investigate the holdouts before
  proceeding to Step B.
- **0-1/4 solved:** diagnosis wrong; do not proceed to Step B. Write up
  the failure mode and recommend the next investigation.

## Style

- Concrete > ceremonial. Quote the actual proofs found.
- If the diagnosis was right, say so plainly. If it was wrong, also say
  so plainly.
- Skip apologies and hedging.

When done, print the path to FINDINGS.md and a 3-line summary of what
beam found. Stop there. Step B is a separate brief that the user will
launch only after reviewing your output.
