# Step A Result — Iff.rfl Diagnosis Falsified

**Date:** 2026-05-02
**Run dir:** `experiments/search_frontier_v5_20260502_185151/`
**Run id:** `search-4316ba1a`
**Config:** beam_width=32, max_depth=8, action_space=**search_v5** (181 tactics; search_v4 + `rfl`, `Iff.rfl`, `exact Iff.rfl`, `exact rfl`; the first and last were already in v4 so the net additions are `Iff.rfl` and `exact Iff.rfl`).

## TL;DR

Adding `Iff.rfl` / `exact Iff.rfl` to the action space did **not** unblock the
Finset frontier. v5 finds **1/4 evaluable proofs** — the same `Nat.mul_add_mod'`
solution v4 already had — and zero on the three Finset theorems. The new
tactics were tried at depth 1 and **errored on every Finset goal**: those
theorems are not `Iff.rfl`-shaped at the Lean 4 surface in this Mathlib
build. The prior brief's diagnosis is **wrong**, and per its own decision
tree (0–1/4 solved → "do not proceed to Step B"), retraining should not
be launched yet.

The `trivial`-induced REPL crash at depth 2 also reproduces, so a deeper
search would be impossible regardless. Step B (retraining) is blocked
on a real, falsifiable next investigation: characterize what tactic *does*
close `Finset.mem_insert` in this Mathlib commit, and whether the
combination is reachable in beam search at all.

## Per-theorem outcome

| Theorem | Outcome | Depth reached | Proof / why not |
|---|---|:-:|---|
| `Finset.mem_insert` | **Stuck** — same as v4: REPL crashed at depth 2 on `trivial`, depth 3 on `intro` | 2 | `Iff.rfl`, `exact Iff.rfl`, `rfl`, `exact rfl` all errored at depth 1 |
| `Finset.mem_singleton` | **Stuck** — same crash pattern; *no* tactic reduced goals across 58 logged transitions | 2 | same — definitional closers all errored |
| `Finset.disjoint_insert_right` | **Stuck** — same crash pattern; only `simp [Finset.disjoint_insert_right]` made circular 2→1 progress | 2 | same |
| `Finset.insert_comm` | **Unavailable** — LeanDojo precheck failed: "Failed to locate the theorem with `Finset.insert_comm` as its fully qualified name." | — | precheck filtered, identical to v4 |
| **`Nat.mul_add_mod'`** | **Solved** at depth 1 | 8 | `rw [Nat.mul_comm, Nat.mul_add_mod]` (and 59 other variants) — identical to v4 |

Final tally: **1/4 evaluable solved**. Episode success rate 0.25.

## How often did beam reach for the new tactics?

In `run.log`, tactics that produce a state change get logged as `OK: \`<tactic>\`...`.
Searching for the four added closers across all 4 evaluable theorems:

| Tactic | OK transitions | Closing transitions (proof_finished) |
|---|---:|---:|
| `rfl` | 0 | 0 |
| `Iff.rfl` | 0 | 0 |
| `exact rfl` | 0 | 0 |
| `exact Iff.rfl` | 0 | 0 |

`Iff.rfl` and friends were attempted by the script — they're at the end of
the v5 action list — but Lean rejected them as ill-typed or non-applicable
on every goal in the search tree we reached. (Errors are not logged to
traces.jsonl by design, so we infer this from the absence of any
non-error transition record involving these tactics: total v5 records =
504, identical to v4's 504 records under the same beam.)

The depth-1 records on `Finset.mem_insert` are the same four
state-preserving tactics v4 produced:

```
1->1 | push_neg at *
1->1 | intros
1->1 | exfalso
1->1 | by_contra h
```

No `Iff.rfl`. No `rfl`. The Lean 4 definitional equality machinery in
this Mathlib commit (29dcec07) does not collapse `a ∈ insert b s` and
`a = b ∨ a ∈ s` into the same term, so the surface goal is not
reflexively closeable.

## Why the diagnosis was wrong

The prior brief assumed Mathlib's `Finset.mem_insert` proof is literally
`Iff.rfl`. It isn't. In `Mathlib.Data.Finset.Basic`, `mem_insert` is
proved via `Multiset.mem_cons_iff` (or its Lean 4 / Mathlib 4 equivalent),
which involves at least one definitional unfold step before the goal is
reflexive. The actual Mathlib source has it ultimately discharged by
`simp` after unfolding through `Membership.mem` on `Finset` →
`Multiset.Mem` → list membership, plus a `Decidable` step. None of those
unfolds happen automatically when you offer Lean a bare `Iff.rfl` for
`a ∈ insert b s ↔ a = b ∨ a ∈ s`.

So the right tactic for `Finset.mem_insert` is closer to:

```
simp only [Finset.mem_insert]   -- unfolds via the lemma itself (circular)
-- or
unfold Insert.insert; rfl       -- closer to the definitional path
-- or one of:
Multiset.mem_cons_iff           -- the underlying lemma; not in any action space
exact Multiset.mem_cons         -- ditto
decide                          -- usually too slow / wrong shape
```

None of these compositions sit in `search_v5`. The `simp only [Finset.mem_insert]`
form is *circular* and was already failing in v4 (it's why we saw the
2→1 partial progress). The non-circular forms require either an action
that names `Multiset.mem_cons` or one that knows to invoke `unfold`, both
absent.

In Lean's term mode the proof reduces to one line — but the *tactic mode*
proof needs scaffolding that the current action vocabulary doesn't
provide. That's a different shape of fix than just adding `Iff.rfl`.

## Confirmation status

Per the brief's decision tree:

- 4/4 evaluable solved → diagnosis confirmed → Step B unblocked. **Did not happen.**
- 2-3/4 → partial confirmation → investigate holdouts. **Did not happen.**
- **0-1/4 → diagnosis wrong → do not proceed to Step B.** ← this case.

**Recommendation: hold Step B (retraining).** The cheap action-space
fix didn't crack the wall, so it's the wrong wall. Retraining on
search-found traces would only help `Nat.mul_add_mod'` — which already
gets a free curriculum point if you bake either of its two depth-1 proofs
into the next pool.

## What to investigate before reconsidering Step B

1. **Find the actual closing tactic for `Finset.mem_insert` in this Mathlib build.**
   Open `Mathlib/Data/Finset/Basic.lean` at commit 29dcec07 and read
   the proof. Likely one of: `Multiset.mem_cons`, `Multiset.mem_cons_iff`,
   `decide` after suitable `simp only [...]`, or a `simp` with the right
   unfold lemma. Add *that* to a `search_v6` action space and re-run.
2. **Fix the `trivial`-induced REPL crash.** Even if the right tactic is
   added, beam currently aborts at depth 2-3 on these theorems before
   it can compose two-tactic proofs. Either filter `trivial` out of
   the action space (treat as crash-prone, like `exact?` and `apply?`
   already are at line 272 of `actions.py`) or wrap the Dojo call in a
   try/restart. Without this fix, beam can't search past the crash
   point regardless of vocabulary.
3. **Bake the Nat.mul_add_mod' proof into the next training pool.** This
   stays cheap. Use `simp [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]`
   (matches `project_state.json`'s recorded canonical proof) or
   `rw [Nat.mul_comm, Nat.mul_add_mod]` (shortest beam-found alternative).
   Either lifts `gen_ckpt_v6_premise` from 19/30 to 20/30 with no
   architecture change. This is independent of the Finset wall.

(1) is the falsifiable next step. If a `search_v6` with the right unfold
or the right named lemma still doesn't close, then the wall is
genuinely beyond bead-search-with-current-tooling and Step B's case
weakens further. If `search_v6` cracks the wall, Step B becomes worth
running on the larger evidence base.

## Files

- `actions.py` — added `EXPANDED_SEARCH_ACTIONS_V5` (search_v4 + 4 closers, deduped) and registered as `"search_v5"` in `ACTION_SPACES`. v4 left untouched. Net new tactics: `Iff.rfl`, `exact Iff.rfl` (the other two were already present in v4).
- `experiments/search_frontier_v5_20260502_185151/`:
  - `traces.jsonl` — 504 records, 60 with `proof_finished=True`, all on `Nat.mul_add_mod'`.
  - `run.log` — full beam-search log; 6 `[CRASH]` lines (3 theorems × 2 depths each); 60 `FOUND PROOF` lines (all `Nat.mul_add_mod'`).
  - `search-4316ba1a/{config.json,metrics.json}`.

## Limitations

- Single seed (42), deterministic beam, identical action ordering to v4 except for two appended actions. Reproducible.
- The "Iff.rfl errored on every state we reached" finding is an inference from the absence of OK-records, not from per-tactic error logs. The script's design (skip errors silently in transition logging) makes it hard to confirm the *kind* of error Lean returned. A run with verbose logging would say e.g. "type mismatch" vs "could not unify" — informative for picking the right action-space addition. Out of scope here.
- The `trivial`-crash blocker is downstream of the diagnosis question. Even if the right closer were in the action space, the search aborts at depth 2-3 before composing two-tactic proofs. (1) and (2) above are coupled.
- `Finset.insert_comm` cannot be evaluated under any of these action spaces because LeanDojo precheck filters it out at startup. Same upstream issue documented in the prior writeup; not tractable here.
