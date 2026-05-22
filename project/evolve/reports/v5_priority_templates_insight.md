# Why `Nat.div_lt_one_iff` doesn't close — and the priority_templates fix

## The trace

In `v5-01-div-hyp-pos`, the v45 `{hyp_pos}` div templates were restored
to the constructor variant's div family. The template
`rw [Nat.div_lt_iff_lt_mul hb, Nat.mul_one]` should close
`Nat.div_lt_one_iff` (`a / b < 1 ↔ a < b` with `hb : 0 < b`) in a single
step. The trace says it didn't, and the reason is structural.

At step 1 on `Nat.div_lt_one_iff`:

```
step=1 origin=generative_topk   tactic=simp [Nat.one_mul]   kind=TacticState   ng:1→1
```

`simp [Nat.one_mul]` advanced state. It didn't close the goal — it
just rewrote it into a less-useful form (apparently simp normalized
`a < 1` into `a = 0`, removing the `< 1` pattern that
`Nat.div_lt_iff_lt_mul` needs to match).

At step 2 the rewrite then fails on every form:

```
step=2 origin=family_tactic     tactic=rw [Nat.div_lt_iff_lt_mul hb, Nat.mul_one]
                                kind=LeanError
                                err=rewrite failed, did not find instance of the pattern in the target expr
```

The goal that the rewrite needs is no longer there.

## A second-order consequence: term_builder didn't fire either

In `v5-04-term-iff-basic`, the iff term skeletons were configured for
goal_shape="iff", which `Nat.div_lt_one_iff` matches. The trace confirms
the wrapper *did* emit `exact ⟨fun h => by omega, fun h => by omega⟩`
in its ranked list. But the trace shows **zero term_builder attempts on
div_lt_one_iff**.

Why? Because the wrapper has *first-non-erroring-wins* semantics. At
step 1, generative_topk's `simp [Nat.one_mul]` is tried first, advances
state (doesn't error), and the wrapper accepts it. The downstream
ranked list — including term_builder — is never examined at this step.

So term_builder is emitted but unreachable behind the generative model's
output. The same is true of family templates and retrieved-premise
rewrites: they sit in the ranked list, but if the model produces a
non-erroring tactic earlier in the list, they are silently bypassed.

This is the same structural ordering bug seen on `Nat.div_lt_one_iff`,
generalized: **the wrapper's ranked list is a search-priority list, but
the search uses only the first non-erroring entry. The rest of the list
is wasted compute.**

`priority_templates` moves selected templates to the front of the list
so they get the *first* attempt rather than being bypassed.

## Why this happens

The wrapper emits tactics in fixed order:

```
generative_topk → family_tactic → retrieved_premise → term_builder → fallback_tactic
```

`generative_topk` runs first because the v3 design assumed the model
"knows best". For most goals that's true. But for goals where the model
emits a syntactically-valid simp that *advances* the state into a
less-useful form (rather than erroring out), the family/template/term
layers downstream never get to fire on the original goal.

This is a **structural ordering bug** in the wrapper. The genome can
configure *what* templates exist and in what relative order they go,
but not whether they go before or after the model.

## The fix

Add a new `priority_templates` block, shape-keyed like
`term_builder_templates`, that emits BEFORE generative_topk. Use it
surgically: only configure priority templates for goal shapes where
you have strong family knowledge that you don't want the model to
override.

In the wrapper:

```python
all_entries = priority_entries + base_entries + extra_entries
```

In the genome:

```json
{
  "priority_templates": {
    "iff": [
      "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
      "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.mul_one]"
    ]
  },
  "priority_template_budget": 4
}
```

When the current state classifies as `iff` and a `{hyp_pos}` hypothesis
is in scope, these tactics emit FIRST. If one closes the goal, we never
even call the base policy at this step. If they all error, the wrapper
falls through to generative_topk → family → … as before.

## Why this matters

This is one of the central findings of the v5 exploration. It is also a
mini-AlphaEvolve discovery: the architecture limitation was invisible at
the design level (the wrapper looked fine on paper) and only became
visible when an autonomous loop made the same mistake on the same
theorem twelve times in a row.

Two architectural lessons:

1. **The mutator can't fix structural bugs.** v3 → v4 spent a quarter
   of its mutation budget reordering fallback strings. None of those
   reorderings could express "this template should run before the
   model" — the slot didn't exist. New slots are *outer-tier* mutations
   (per the architecture proposal in `v5_alphaevolve_architecture.md`),
   and they unblock entire classes of theorems that no inner-tier
   mutation can reach.

2. **Empirical structural bugs justify schema changes more than
   theoretical worry.** Adding `priority_templates` is a strictly
   wider genome — old genomes (with empty `priority_templates`) get
   identical behavior. The genome surface grew by one knob, and that
   knob unlocks a previously-blocked direction.

The v5 second-pass loop (`autonomous_research_followup.py`) tests
this fix.

## Result: confirmed.

Variant `v5-12-prio-div-hyp` ships only this minimal addition to the
v4.7 seed:

```json
{
  "priority_templates": {
    "iff": [
      "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
      "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]",
      "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.mul_one]",
      "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.one_mul]",
      "simp [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]"
    ],
    "lt": [ /* same templates */ ]
  },
  "priority_template_budget": 4
}
```

The trace on `Nat.div_lt_one_iff` shows:

```
step=1  LeanError      origin=tactic_template  fam=priority:iff
        tac=rw [Nat.div_lt_iff_lt_mul hb, Nat.mul_one]
step=1  ProofFinished  origin=tactic_template  fam=priority:iff
        tac=rw [Nat.div_lt_iff_lt_mul hb, Nat.one_mul]   ← closes
```

The first variant (`Nat.mul_one`) fails (the rewrite produces
`a < 1 * b`, so `Nat.one_mul` is needed; not `Nat.mul_one`). The
second variant fires and closes the goal at step 1.

`v5-12-prio-div-hyp` is the **first** v5 candidate to beat the v4.7
26/38 plateau. Newly proved: `Nat.div_lt_one_iff`. New total: **27/38**.

This single-theorem win is structurally important even if numerically
small. It shows:

  - a new genome slot (`priority_templates`), introduced precisely to
    address a structural finding made by the autonomous loop;
  - measurably moves the needle on a previously-unproved theorem;
  - composes with the rest of the v3 → v4 wrapper (the v45 div templates,
    the constructor split, the retrieved-premise rewrites) without
    breaking any of them — the `Nat.div_lt_iff_lt_mul'` win is
    preserved.

Future variants will test whether the same slot unlocks other
template-tractable theorems: `Nat.mul_eq_left/right`,
`Nat.sqrt_lt`, possibly `Nat.add_mod_eq_ite`.

## Update: priority_templates closes two more theorems

Variant `v5-15-prio-mul-specific` ships a single new template per
mul direction:

```
exact ⟨fun h => Nat.eq_of_mul_eq_mul_left
            (Nat.pos_of_ne_zero {hyp_ne_zero})
            (h.trans (Nat.mul_one _).symm),
        fun h => by simp [h]⟩
exact ⟨fun h => Nat.eq_of_mul_eq_mul_right
            (Nat.pos_of_ne_zero {hyp_ne_zero})
            (h.trans (Nat.one_mul _).symm),
        fun h => by simp [h]⟩
```

These are the exact two-direction term-mode proofs of mul cancellation.
The trace shows:

  - **`Nat.mul_eq_left`** (`a*b = a ↔ b = 1`, `ha : a ≠ 0`):
    forward direction closed by `Nat.eq_of_mul_eq_mul_left`. **+1**.
  - **`Nat.mul_eq_right`** (`a*b = b ↔ a = 1`, `hb : b ≠ 0`):
    forward direction closed by `Nat.eq_of_mul_eq_mul_right`. **+1**.

This brings the v5 cumulative new-wins to three:

  1. `Nat.div_lt_one_iff` — v5-12, priority div_hyp_pos
  2. `Nat.mul_eq_left`    — v5-15, priority mul-specific
  3. `Nat.mul_eq_right`   — v5-15, priority mul-specific

Variants like `v5-18-prio-kitchen` and `v5-22-deny-derailers` should
combine all three (their priority lists contain both the div and mul
templates) and we expect **29 / 38** on those.

This confirms `priority_templates` is the structural fix the v5
research was looking for. With it, the v3 → v4 26/38 plateau is
broken by at least three new theorems, each requiring an asymmetric
or hypothesis-aware proof skeleton that the prior architecture
could not surface.
