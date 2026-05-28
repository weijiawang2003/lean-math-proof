# Search the Frontier — Brief for Claude Code

> You are Claude Code, running in `~/dev/dojo_sandbox`. Read this whole brief
> before doing anything. Two prior briefs have been executed and produced
> findings docs that constrain this one — read them first if you have not:
>
>   `experiments/capacity_isolation_20260502_002705/FINDINGS.md`
>   `experiments/finset_wall_20260502_164911/FINDINGS.md`

## Why this task exists

Two prior investigations ruled out two hypotheses for why the same five
theorems are unsolved by every checkpoint in `curriculum_all`
(`Finset.mem_insert`, `Finset.mem_singleton`, `Finset.disjoint_insert_right`,
`Finset.insert_comm`, `Nat.mul_add_mod'`):

- **Capacity is not the cause** (`gen_v7` t5-base on v5 data still misses
  all five).
- **Retrieval is not the cause** (patching `STATIC_PREMISES["Finset"]`
  didn't move the score; the model emits premise names from memorization
  not from the inference prefix).

The remaining hypothesis is **training-data composition**. `Finset.mem_insert`
appears 160× in the training pool as a *tactic argument* (`simp [Finset.mem_insert]`)
but never as a *goal being proved*. The model has learned what the lemma
does, never how to prove it. To fix the wall, the training pool needs
proofs *of* these theorems, not just uses of them.

The cheapest test of this hypothesis: ask whether **beam search alone**
can find proofs for the five frontier theorems. If yes, those proofs become
seed traces for a future training round and the wall is fixable. If no,
the wall requires something stronger — strategic policy, distillation
from a frontier model, or human curation. Either answer redirects the
project's next move.

## Goal

For each of the five frontier theorems, run a wide beam search and report:
- Did beam find a proof? If yes, what is it?
- If no, what tactics did beam try, and what was the failure mode?

This is a diagnostic task. Do not retrain anything. Do not modify the
training pool. Just measure whether search finds proofs.

## Concrete plan

### Phase 1 — define the frontier theorem set

Add a new entry to `tasks.py`'s `THEOREM_SETS` dict. This is purely
additive and safe:

```python
"frontier_v1": [
    TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.mem_insert"),
    TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.mem_singleton"),
    TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.disjoint_insert_right"),
    TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.insert_comm"),
    TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean",     full_name="Nat.mul_add_mod'"),
],
```

This is the only edit to a "canonical" file allowed by this brief.

### Phase 2 — run beam search

```bash
python search_generate_traces.py \
  --theorem-set frontier_v1 \
  --beam-width 32 \
  --max-depth 8 \
  --action-space search_v4 \
  --out experiments/search_frontier_$(date +%Y%m%d_%H%M%S)/traces.jsonl \
  --out-dir experiments/search_frontier_$(date +%Y%m%d_%H%M%S)
```

(Use one timestamp, not two — bash will evaluate `$(date)` twice and you'll
get mismatched paths. Set a shell variable first:
`STAMP=$(date +%Y%m%d_%H%M%S); ROOT="experiments/search_frontier_$STAMP"`.)

Beam-32 × depth-8 × 5 theorems is a meaningful search budget. Expected
runtime on this Mac: probably 10-40 minutes depending on how many tactics
each theorem's beam expands and how often Dojo errors. If any single
theorem hangs for >10 min on its own, kill the run and report it.

### Phase 3 — categorize per-theorem outcomes

Read the run's `metrics.json` and `traces.jsonl`. For each of the 5
theorems, classify into one of:

- **Solved** — beam found a finishing tactic. Record the proof
  (sequence of tactics, depth at which it closed).
- **Plausibly solvable, not found at this budget** — beam made progress
  (goal count dropped at intermediate steps) but didn't close. Worth
  retrying with wider beam or richer action space.
- **Stuck** — beam errored at every action from the initial state, or
  never reduced the goal count. Indicates the action space lacks the
  right primitive for this theorem.
- **Unavailable** — Dojo couldn't initialize this theorem (LeanDojo
  artifact missing). Note and move on; not a real result either way.

For each "stuck" theorem, look at the actual transitions logged: what
tactics were attempted, what error messages did they produce? Sometimes
the failure mode is informative (e.g., "all `simp` variants timed out"
vs. "no tactic ever produced `rfl` or `decide`").

### Phase 4 — write findings

Produce `experiments/search_frontier_<TIMESTAMP>/FINDINGS.md` with:

- Per-theorem classification (table).
- For each solved theorem, the actual proof.
- For each unsolved theorem, the failure-mode classification and a brief
  interpretation.
- A one-paragraph synthesis: how many of the five did beam crack? What
  does this say about the next step?

Decision tree for the synthesis:
- **All 5 solved.** Then the wall is purely a training-pool gap. Next
  move: bake these traces into the next training pool and retrain.
- **2-4 solved.** The wall has multiple layers — some theorems are
  search-tractable, others need a stronger tool. Recommend incorporating
  the solved ones into training and using strategic policy for the rest.
- **0-1 solved.** Beam search is not sufficient for the frontier. The
  next move is either strategic policy (still unablated), distillation
  from Claude/GPT, or human-curated proofs.

Do not retrain or modify training pools yourself. The brief ends at
"recommend the next experiment."

## Constraints

You may:
- Add ONE new theorem set entry to `tasks.py` (additive only).
- Read any file in the repo.
- Run `search_generate_traces.py` with the parameters above.
- Write new files under `experiments/search_frontier_*/`.

You should NOT:
- Modify `search_generate_traces.py`, `env.py`, `core_types.py`, or any
  other canonical script beyond the one additive `tasks.py` entry.
- Re-train any model.
- Run any rollout that takes more than ~45 minutes total.
- Modify `project/project_state.json` or `project/all_traces.jsonl`.
- Add the search-found traces to the training pool yourself — that's a
  separate experiment.
- Edit `MEMORY.md` or anything under `.claude/`.

If beam search hangs hard on one theorem and you have to skip it,
record exactly which one and the symptom; don't power through silently.

## Acceptance criteria

This task is a clear success regardless of how many proofs beam finds:

- If beam finds 4+ proofs → the wall is search-tractable; redirect to
  training-pool curation.
- If beam finds 1-3 → mixed wall; partial path forward documented.
- If beam finds 0 → the wall requires stronger tooling; rules out the
  cheap fix and redirects to strategic policy / distillation.

The only failure mode is *not running* — any concrete answer here moves
the project forward.

## Style

- Concrete > ceremonial. Quote the actual tactics beam tried.
- If you find a proof, say so plainly. Don't bury it in qualifications.
- If beam fails informatively (e.g., the theorem requires `induction`
  which isn't in `search_v4`), say that and recommend the action-space fix.
- Skip apologies, skip "as a language model" framing.

When you finish, print the path to `FINDINGS.md` and a 3-line summary of
what beam found. That's it.
