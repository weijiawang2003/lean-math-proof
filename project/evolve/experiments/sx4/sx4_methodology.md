# SX4 — Sequence Attribution Methodology

## Why SX4 exists

The RC3 candidate (`RC2 ⊕ SX3_SET_ITE_AESOP`, sequence `simp [Set.ite] <;> aesop`) was initially
credited with **+5 wins** (4 deferred + 1 fresh) by the custom SX3 runner, then **rejected**
(`REJECT_NO_LITERAL_DELTA`) under literal-production validation: literal RC2 and literal RC3 both
solve 17/30, credited delta **0**. Every one of the 5 "wins" was already solved by literal RC2's
best-first search via the 2-step path `simp [Set.ite]` (advances) → `aesop` (closes).

The over-credit was a **methodology bug**, not a measurement glitch. SX4 fixes the methodology and
makes the fix reusable + regression-tested.

## The false-credit problem

For a candidate sequence

    A <;> B

the **naive** attribution the SX3 runner used was:

1. `A <;> B` succeeds, **and**
2. `A` alone (from the initial goal) fails, **and**
3. `B` alone (from the initial goal) fails

→ conclude the sequence is a genuine new win.

This is **insufficient**. It only inspects **depth-1** controls applied to the *initial* goal. It
never asks the decisive question: **does the literal production search already reach the
`A`-advanced state and then apply `B`?**

A best-first search with `max_steps > 1` routinely does exactly this: it applies `A` at step *i*
(advancing the goal but not closing it), then at step *i+1* applies `B` from the advanced state and
closes. The two search steps **are** `A <;> B`, decomposed. `B` "alone from the initial goal"
failing is irrelevant — production never applies `B` to the initial goal; it applies `B` to the
`A`-advanced state, which is what the grouped sequence also does.

Concretely for `simp [Set.ite] <;> aesop` on `Set.ite_inter`:

- SX3 control `aesop` (on the initial goal) → **fails** ⇒ SX3 inferred the sequence was essential.
- Literal RC2 trace: step 1 `simp [Set.ite]` → `TacticState` (1 goal → 1 goal, advanced); step 2
  `aesop` from that advanced state → `ProofFinished`. ⇒ production **already** does `A`-then-`B`.

So the sequence is **PRODUCTION_SUBSUMED**, credit = false.

## Correct attribution requires

A sequence `A <;> B` is credited as a genuine delta **only if**:

1. **Literal production baseline does not already solve the theorem** (`baseline_finished == false`).
2. The **candidate** (production ⊕ the sequence) **does** solve it (`candidate_finished == true`).
3. **Depth-1 controls fail**: neither `A` alone nor `B` alone (from the initial goal) closes.
4. **The literal production trace does not already reach an equivalent `A`-advanced state and apply
   `B`** (no equivalent `A`→`B` continuation already present in the baseline search).
5. The win is **not merely a restatement of production search behavior** (it is fresh over literal
   production, generic — not a copy of the library source proof, and survives off-gate / floors /
   determinism / fresh-holdout).

If (1) is false → **PRODUCTION_SUBSUMED**. This single check would have caught the RC3 bug.

## Classification definitions

| class | meaning | credit |
|---|---|---|
| `TRUE_SEQUENCE_DELTA` | candidate sequence solves a theorem literal production does **not** solve, and depth-1 controls fail, and no equivalent production `A`→`B` continuation exists | **true** |
| `PRODUCTION_SUBSUMED` | literal production already solves it using the same/equivalent intermediate-state continuation (`A`-state then `B`) | false |
| `DEPTH1_DUPLICATE` | `A` alone or `B` alone (from the initial goal) solves | false |
| `ROUTING_DUPLICATE` | production search already emits an equivalent tactic family (e.g. bare `aesop` / `simp_all`) that closes it | false |
| `TRACE_INSUFFICIENT` | logs do not expose enough state to distinguish subsumption from genuine delta | false (default) |
| `FAILED_SEQUENCE` | candidate does not solve the theorem | false |
| `NEEDS_REVIEW` | none of the above cleanly applies | false |

**Default-to-no-credit:** absence of evidence is not evidence of a delta. `TRACE_INSUFFICIENT` and
`NEEDS_REVIEW` never carry credit.

## The one-line rule

> **Never credit a depth-k sequence based only on depth-(k-1) controls. Always compare against a
> literal production run with the same `max_steps`/`top_k`, and inspect its trace for an equivalent
> intermediate-state continuation.**
