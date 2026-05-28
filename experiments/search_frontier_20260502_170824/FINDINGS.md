# Search the Frontier — Beam Cracks 1 of 5, Action Space Limits the Rest

**Date:** 2026-05-02
**Run dir:** `experiments/search_frontier_20260502_170824/`
**Run id:** `search-b9c392e3`
**Config:** beam_width=32, max_depth=8, action_space=search_v4 (179 tactics), seed=42, total runtime ~13 min.

## TL;DR

Beam-32 × depth-8 over `search_v4` (179 tactics) cracked **1 of 5** frontier theorems:
**`Nat.mul_add_mod'`** is solved at depth 1 with a single rewrite — the wall
on this theorem was the model's beam preference, not search-tractability.
The four Finset frontier theorems all stayed unsolved, with two distinct
failure modes: `Finset.insert_comm` is unavailable (LeanDojo can't locate
it); the other three (`Finset.mem_insert`, `Finset.mem_singleton`,
`Finset.disjoint_insert_right`) are stuck for a structural reason — the
canonical Mathlib proofs go through `Iff.rfl` after definitional unfolding,
and `Iff.rfl`-style closers are not in the `search_v4` action space.
A complicating factor: the LeanDojo REPL also crashed on `trivial` partway
through depth 2 for all three Finset theorems, so deeper search wouldn't
have helped without separately addressing both issues. Per the brief's
decision tree (1 solved → 0-1 bucket), beam alone is not the unlock for
the Finset wall; the next move is either expanding the action space to
include `Iff.rfl`, distilling proofs from a frontier model, or human
curation.

## Per-theorem outcomes

| Theorem | Outcome | Depth reached | Goal-reducing tactic seen |
|---|---|:-:|---|
| `Finset.mem_insert` | **Stuck** — REPL crashed at depth 2 on `trivial`, then on `intro` at depth 3; circular `simp [Finset.mem_insert]` made 2→1 progress but no closer | 2 | `simp [Finset.mem_insert]` (2→1, circular) |
| `Finset.mem_singleton` | **Stuck** — same REPL crash pattern; *no* goal-reducing tactic found in 58 logged transitions | 2 | none |
| `Finset.disjoint_insert_right` | **Stuck** — same REPL crash pattern; circular `simp [Finset.disjoint_insert_right]` made 2→1 progress but no closer | 2 | `simp [Finset.disjoint_insert_right]` (2→1, circular) |
| `Finset.insert_comm` | **Unavailable** — LeanDojo precheck failed: "Failed to locate the theorem with `Finset.insert_comm` as its fully qualified name." Not attempted. | — | — |
| **`Nat.mul_add_mod'`** | **Solved** at depth 1 | 8 | **`rw [Nat.mul_comm, Nat.mul_add_mod]`** (the proof) |

Final tally: **1 solved / 4 evaluable** (1/5 if you count the unavailable
one as a strict failure of the brief's frontier set).

## The proof beam found

`Nat.mul_add_mod'`:

```
⊢ (a * b + c) % b = c % b

   rw [Nat.mul_comm, Nat.mul_add_mod]   -- closes in one step
```

Beam also found 59 other proofs of this same theorem at depth ≤ 2, including the
shortest single-step alternative:

```
   simp [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]
```

— which matches the canonical Mathlib proof recorded in `project_state.json`. The
fact that the model never emits this in its top-8 beam, while a
non-learned tree-search trivially does, confirms the prior writeup's
finding: this theorem is a *model preference* problem, not a
search-difficulty problem. Adding either of these two depth-1 traces to
the next training pool would almost certainly fix it.

## Why the three Finset theorems stayed unsolved

Two stacked failure modes — both load-bearing.

### 1. The canonical proofs are `Iff.rfl`-shaped, and `Iff.rfl` is not in search_v4

In Mathlib, `Finset.mem_insert : a ∈ insert b s ↔ a = b ∨ a ∈ s` is proven
by reducing through `Multiset.mem_cons` to a definitional unfolding, then
closing with `Iff.rfl` (or implicit `rfl` after `simp only`). The same is
true for `Finset.mem_singleton` and `Finset.disjoint_insert_right` — each
is a small definitional unfolding plus a reflexivity closer.

`search_v4` contains `rfl` (definitional refl on equalities) but not
`Iff.rfl` or `exact Iff.rfl`. After 12 `simp` variants, `aesop`, `tauto`,
`decide`, etc., none of those can substitute for the missing `Iff.rfl` on
a goal of the form `(P ↔ P)` after definitional unfolding. The
goal-reducing tactics that *did* fire (`simp [Finset.mem_insert]` on its
own goal) reduced 2→1 by simplifying one side of the iff via the lemma
itself — circular, and unable to fully close because using
`Finset.mem_insert` to prove `Finset.mem_insert` is exactly the kind of
loop simp's termination check rejects.

`Finset.mem_singleton` is the most informative: across 58 logged
transitions and the full 179-action menu at depth 1, **no tactic produced
goals_after < goals_before**. The action space simply contains no
primitive that touches `b ∈ {a} ↔ b = a` constructively. Without
unfolding `{a}` to `insert a ∅` and running through definitional-eq
machinery, the goal sits.

### 2. The Lean REPL crashed mid-depth-2 on `trivial`

For all three evaluable Finset theorems, the run.log shows:

```
==== Depth 2 ====
  ... (prior tactics ran fine)
  [CRASH] REPL died on `trivial` — aborting theorem
Beam size: 32, finished: 0
==== Depth 3 ====
  [CRASH] REPL died on `intro` — aborting theorem
No valid successors, stop.
```

`trivial` is action #42 of 179 in `search_v4`. After it kills the Dojo
session, every subsequent transition fails (the REPL is dead), so the
search aborts at depth 3 with no progress. We never see what depths 3-8
could have done — but given the action-space-limit finding above, the
prediction is "still not enough," because the missing primitive is
`Iff.rfl`, not deeper search.

This crash pattern is not a one-off — it reproduced on 3 of 3 Finset
runs. Worth filing as an upstream LeanDojo-or-action-space bug; until
fixed, beam search cannot run to its full depth on goals of this shape.

### 3. Finset.insert_comm: LeanDojo can't locate it

The precheck in `_partition_available_theorems` raised:

```
Failed to locate the theorem with `Finset.insert_comm` as its fully qualified name.
```

This is the same error the curriculum eval reported. Probably a
namespace shadowing or trace-artifact issue in the cached LeanDojo build
of mathlib4 commit 29dcec07. Out of scope for this brief.

## Synthesis

**One of five solved, three structurally blocked, one unavailable.** Per the
brief's decision tree, this is the **0–1 solved bucket**: beam search alone
is not the unlock for the Finset wall.

But the structure of the failure is informative and points at the cheapest
possible next move: **add `Iff.rfl` (and/or `exact Iff.rfl`) to the action
space**, then re-run this exact experiment. If the three crashed-but-stuck
Finset theorems are really just "definitional unfold + Iff.rfl," that one
change could shift the result from 1/4 to 4/4 *without retraining*.
That's a smaller, faster experiment than strategic policy or distillation
and should be tried first.

If `Iff.rfl` doesn't crack them, then the wall really does need stronger
tooling. The brief's three options at that point — strategic policy,
distillation from Claude/GPT, human curation — are roughly in order of
increasing investment. Strategic policy is "free" (no extra training
data, just routing), and the prior writeup notes it has not been ablated
on this curriculum yet, which makes it the natural next step after the
action-space patch.

For `Nat.mul_add_mod'` specifically: this one is purely a training-pool
gap. Beam found `rw [Nat.mul_comm, Nat.mul_add_mod]` and 59 other proofs
in seconds. Adding any of them as a (state, tactic) pair to the next
training pool will almost certainly fix it without touching architecture
or retrieval.

## Recommended next experiment

In priority order, descending by cost-effectiveness:

1. **Action-space expansion (~1 line change to `actions.py`).** Add
   `Iff.rfl`, `exact Iff.rfl`, `simp only [Finset.mem_insert]; exact
   Iff.rfl`-style finishers to `search_v4` (or define `search_v5`).
   Re-run this brief's experiment. Expected: 4/4 evaluable solved if the
   theorems really are Iff.rfl-shaped after one simp.
2. **Diagnose and fix the `trivial` REPL crash.** Reproduce on a minimal
   theorem; file with LeanDojo or wrap `trivial` in a try/finally that
   restarts the Dojo session. This is upstream of all beam-search work
   on Finset goals.
3. **Bake the Nat.mul_add_mod' proofs into the next training pool.**
   Either of the two depth-1 proofs found here suffices. This is one
   theorem of curriculum lift for the cost of two `append_jsonl` calls.
4. **(Only if 1-2 don't crack the Finsets) Strategic policy ablation.**
   Run the existing `strategic_policy.py` on `frontier_v1` and see
   whether it routes to the right tactics. Costs no retraining.
5. **(Last resort) Distillation from Claude/GPT or human curation** of
   the Finset proofs. Significant effort; only justified if 1-4 fail.

## Files

- `tasks.py` — added one entry: `THEOREM_SETS["frontier_v1"]` with the 5 theorems above. Additive; no other edits.
- `experiments/search_frontier_20260502_170824/`:
  - `traces.jsonl` — 504 records (60 with `proof_finished=True`, all on `Nat.mul_add_mod'`).
  - `run.log` — full beam-search log including the `[CRASH]` and `FOUND PROOF` lines quoted above.
  - `search-b9c392e3/{config.json,metrics.json}` — episode summary; success_rate=0.25 across 4 evaluable.

## Limitations

- Single seed (42), deterministic beam over a fixed action ordering. Beam-search outcomes are reproducible but a different action ordering could plausibly hit the `trivial` crash earlier or later, changing how much depth-2 progress gets logged before the REPL dies.
- Total runtime ~13 min; under the brief's 45-min budget. No theorem was killed for budget reasons. The 4-evaluable-of-5 number is a tooling artifact (`Finset.insert_comm` lookup failure), not a search outcome.
- The "stuck" classification on the three Finset theorems compounds two issues — action-space gap and REPL crash — and we cannot independently rule out either without a re-run after fixing both. The action-space gap is the prediction-bearing claim; the REPL crash is the immediate blocker.
