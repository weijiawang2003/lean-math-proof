# Deep dive — `Nat.add_mod_eq_ite` (one of v5's 7 remaining failures)

This document documents one of the unsolved theorems from v5 in
deeper detail than the headline failure classification. It exists so
a v6 contributor can pick up exactly where this v5 left off, without
re-deriving the analysis.

## Theorem statement

```
theorem Nat.add_mod_eq_ite :
  (m + n) % k =
    if k ≤ m % k + n % k then m % k + n % k - k else m % k + n % k
```

## Why it doesn't close under v5-27-w4-master (31/38)

Step-1 trace (in chronological order, first attempts that produce a
TacticState — see `v5-wave4-…/eval/v5-27-w4-master/eval-*/traces.jsonl`):

```
step=1  tac=split_ifs <;> omega                               result=LeanError
        err="omega could not prove the goal: a possible counterexample
        may satisfy the constraints ..."
step=1  tac=split_ifs <;> simp_all <;> omega                  result=LeanError
        err="simp_all made no progress / omega could not prove the goal"
step=1  tac=simp [Nat.add_mod, Nat.mod_eq_of_lt]              result=TacticState (advance)
```

The first two tactics — both involving `split_ifs` — DO split the
if-then-else correctly, producing two subgoals:

  - if-branch: `(m + n) % k = m % k + n % k - k`
  - else-branch: `(m + n) % k = m % k + n % k`

After split_ifs, the `<;>` propagates `omega` (or `simp_all <;> omega`)
to both branches. omega fails because the goal involves `%`, which it
cannot directly reason about. `simp_all` fails because the goal is
already simplified to a form `simp_all` cannot reduce further.

After split_ifs fails, the wrapper moves on. The next non-erroring
tactic is `simp [Nat.add_mod, Nat.mod_eq_of_lt]` (family layer), which
"advances" the state — but only by rewriting one side of the equation
into the same `(m + n) % k = ...` form it started in (with a slightly
different surface representation). The proof never closes.

## What would actually work

In standard mathlib style, the human proof of this lemma is something
like:

```lean
theorem Nat.add_mod_eq_ite :
    (m + n) % k =
      if k ≤ m % k + n % k then m % k + n % k - k else m % k + n % k := by
  rw [Nat.add_mod]
  split_ifs with h
  · -- branch: k ≤ m%k + n%k. Goal: (m%k + n%k) % k = m%k + n%k - k
    rw [Nat.mod_eq_sub_of_lt (Nat.lt_two_mul_self h)]  -- or similar
    -- Hand-prove using the relationship between mod and subtraction.
  · -- branch: ¬ k ≤ m%k + n%k. Goal: (m%k + n%k) % k = m%k + n%k
    rw [Nat.mod_eq_of_lt (Nat.not_le.mp h)]
```

The key insight: split_ifs splits CORRECTLY, but each branch needs a
DIFFERENT closing tactic — one needs `Nat.mod_eq_sub_of_lt` (or a
chain of sub/le lemmas), the other needs `Nat.mod_eq_of_lt`.

A symmetric `<;>` distribution can't do this.

## v6 implication

This is the canonical case for the **asymmetric branch skeleton**.
A v6 skeleton declaration should look like:

```python
SKEL_ite_asymmetric = Skeleton(
    name="ite_asymmetric_eq",
    shape="le",   # the classifier returns "le" because ≤ is inside the if
    template="""
        rw [{outer_rewrite}]
        split_ifs with h
        · {if_branch}
        · {else_branch}
    """,
    slots={
        "outer_rewrite": ["Nat.add_mod"],
        "if_branch": [
            "rw [Nat.mod_eq_sub_of_lt h']; omega",
            "exact Nat.sub_mod_eq ...",
            "simp_all; omega",
        ],
        "else_branch": [
            "rw [Nat.mod_eq_of_lt (Nat.not_le.mp h)]",
            "simp [Nat.mod_eq_of_lt, Nat.not_le.mp h]",
        ],
    },
)
```

This skeleton has THREE separate slots, and the mutator can rewrite
each independently. The v5 priority_templates slot has only ONE
inner tactic that propagates via `<;>` — fundamentally too coarse for
asymmetric branches.

## Lower-hanging-fruit alternative

A v5.1 hack that doesn't need v6 architecture: a longer-form priority
template that hard-codes the asymmetric branches:

```
rw [Nat.add_mod]; split_ifs with h; · rw [Nat.mod_eq_sub_of_lt ...]; omega; · rw [Nat.mod_eq_of_lt (Nat.not_le.mp h)]
```

This isn't elegant, but it's the kind of thing the LLM mutator
proposed in `v5_alphaevolve_architecture.md` could generate. It's
also the kind of thing that, once proved manually, the trace
contains and a future `gen_v5+1` fine-tune could learn (per
`v5_trace_to_training_plan.md`).

## Other 7-unsolved theorems likely needing similar analysis

Apply the same approach to:

  - `Nat.eq_one_of_mul_eq_one_left` — case decomposition on `m, n`.
  - `Nat.div_le_div_right` — likely needs the right Mathlib chain
    (`Nat.div_mul_le_self`, `Nat.le_div_iff_mul_le`).
  - `Nat.dvd_iff_div_mul_eq` — asymmetric iff with dvd witness on
    one side and div arithmetic on the other.

The other three (`Nat.AM_GM`, `Nat.sqrt_lt`, `Nat.pow_lt_pow_iff_left`)
are environment-limited and need either a Lean tactic that doesn't
exist in this env (`nlinarith`) or a Mathlib lemma rename / addition.
Those four are out of scope for v5 / v6 architectural changes.
