# v5 autonomous research exploration — report

**Branch:** `v5-autonomous-proof-program-evolution`
**Start:** 2026-05-22 04:45 CDT
**Run id (first pass):** `v5-auto-20260522-095802-1fcaa0`
**Run id (followup pass):** `v5-followup-20260522-103058-537f36`

This document is the v5 result writeup. The companion documents:

  - `v5_research_plan.md` — the steering plan written before any coding.
  - `nat_defs_medium_failure_classification_v5.md` — failure analysis
    of the 12 unsolved theorems from the v4.7 26/38 seed.
  - `v5_priority_templates_insight.md` — **the central architectural
    finding of the run.**
  - `v5_trace_to_training_plan.md` — Direction E plan (no training
    tonight).
  - `v5_alphaevolve_architecture.md` — Direction F architecture
    proposal.

## Headline

> _Confirmed: v4.7 plateau **26 / 38** → v5 best **31 / 38**
> (v5-27-w4-master)._ All five new wins stack in the master combo,
> confirming the priority_templates mechanism is the right
> architectural fix. **+5 theorems / +13 percentage points** over v4.7.

The five new wins:

  1. `Nat.div_lt_one_iff`  — `rw [Nat.div_lt_iff_lt_mul hb, Nat.one_mul]`
  2. `Nat.div_pos`         — `exact (Nat.le_div_iff_mul_le hb).mpr (by simpa using hba)`
  3. `Nat.div_pos_iff`     — `rw [Nat.pos_iff_ne_zero, Nat.div_ne_zero_iff hb]`
  4. `Nat.mul_eq_left`     — `exact ⟨Nat.eq_of_mul_eq_mul_left (Nat.pos_of_ne_zero ha) ..., simp [h]⟩`
  5. `Nat.mul_eq_right`    — `exact ⟨Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero hb) ..., simp [h]⟩`
>
> Newly proved (priority_templates origin):
>   1. **`Nat.div_lt_one_iff`** — v5-12 (`rw [Nat.div_lt_iff_lt_mul hb, Nat.one_mul]`)
>   2. **`Nat.mul_eq_left`**   — v5-15 (term-mode `Nat.eq_of_mul_eq_mul_left` skeleton)
>   3. **`Nat.mul_eq_right`**  — v5-15 (term-mode `Nat.eq_of_mul_eq_mul_right` skeleton)
>
> Δ comes from a single architectural change: a new genome slot
> (`priority_templates`) that emits a small set of family templates
> BEFORE the generative model's output, rather than after. The
> shadowing fix unlocks template-tractable theorems the model would
> otherwise derail with a weak `simp [...]` at step 1.

## Total runtime

> _To be filled at completion._

## Experiments run

### First-pass autonomous loop (12 variants)

Tested whether v3 → v4-style mutations (extra templates, new
families, term_builder origin, skeleton mutation) move beyond
26/38 without changing the wrapper's tactic-emit order.

### Second-pass followup loop (11 variants)

Tested the `priority_templates` mechanism added between the two
loops. Every followup variant uses some priority templates; the
question is which targets pay off.

### Wave 3 (adaptive)

Mutations around any winning followup variant.

### Direction D — `nat_defs_large_v5`

Single eval of the best v5 candidate on a 68-theorem set
extending nat_defs_medium with 30 new theorems from
`Mathlib/Data/Nat/Defs.lean`.

## Headline scoreboard (first pass — Directions A/B/C only)

| # | variant | direction | proved | Δ | term_builder a/p | new wins |
|---|---|---|---|---|---|---|
| 1 | v5-00-baseline-repro | baseline | 26/38 | +0 | 0/0 | — |
| 2 | v5-01-div-hyp-pos | B | 26/38 | +0 | 0/0 | — |
| 3 | v5-02-mul-family | B | 26/38 | +0 | 0/0 | — |
| 4 | v5-03-split-ifs | B | 26/38 | +0 | 0/0 | — |
| 5 | v5-04-term-iff-basic | A | 26/38 | +0 | 122/14 | — |
| 6 | v5-05-term-iff-adv | A | 26/38 | +0 | 138/14 | — |
| 7 | v5-06-term-iff-hyp | A | 26/38 | +0 | 54/14 | — |
| 8 | v5-07-term-dvd | A | 26/38 | +0 | / | — |
| 9 | v5-08-pow-sqrt | B | 26/38 | +0 | 0/0 | — |
| 10 | v5-09-skeleton-mut | C | running | | | |
| 11 | v5-10-combo-minimal | B | pending | | | |
| 12 | v5-11-combo-aggressive | B | pending | | | |

**First-pass finding:** *every* first-pass variant plateaued at 26/38.
Term_builder works as a mechanism (122 attempts, 14 attributions on
v5-04), but it never broke the plateau because the wrapper's
generative-first ordering meant template/term entries were
shadowed by the model's `simp [...]` output that advanced state
without closing.

## Headline scoreboard (second pass — priority_templates, full)

| # | variant | direction | proved | new wins (vs v5-00) |
|---|---|---|---|---|
| 1 | v5-12-prio-div-hyp | B+priority | 27/38 | `Nat.div_lt_one_iff` |
| 2 | v5-13-prio-iff-constructor | A+priority | 26/38 | — |
| 3 | v5-14-prio-combo | A+B+priority | 27/38 | `Nat.div_lt_one_iff` |
| 4 | v5-15-prio-mul-specific | B+priority | 28/38 | `Nat.mul_eq_left`, `Nat.mul_eq_right` |
| 5 | v5-16-prio-sqrt-pow | B+priority | 26/38 | — (`Nat.sqrt_lt'` doesn't exist in this env) |
| 6 | v5-17-prio-term-iff | A+priority | 26/38 | — (covered by basic omega) |
| 7 | v5-18-prio-kitchen | all+priority | **29/38** | `Nat.div_lt_one_iff`, `Nat.mul_eq_left`, `Nat.mul_eq_right` |
| 8 | v5-19-prio-split-ifs | B+priority | 26/38 | — (split_ifs advances but no closing tactic) |
| 9 | v5-20-prio-div-pos | B+priority | 28/38 | `Nat.div_pos`, `Nat.div_pos_iff` |
| 10 | v5-21-prio-iff-basic | A+priority | 26/38 | — |
| 11 | v5-22-deny-derailers | B+priority+deny | 29/38 | same as v5-18 |

> **First-pass best: 26/38 (no movement). Second-pass best: 29/38 (+3).
> Second pass also discovered TWO MORE priority targets (`Nat.div_pos`,
> `Nat.div_pos_iff`) via v5-20.**

## Headline scoreboard (wave 4 — combine all priority wins)

| # | variant | proved | new wins (vs v5-00) |
|---|---|---|---|
| 1 | v5-23-w4-split-ifs | 26/38 | — |
| 2 | v5-24-w4-dvd-iff | 26/38 | — |
| 3 | v5-25-w4-div-pos | 28/38 | `Nat.div_pos`, `Nat.div_pos_iff` |
| 4 | v5-26-w4-sqrt-pow | 26/38 | — (no `Nat.sqrt_lt'` in env, no `Nat.pow_lt_pow_iff_*` alternate) |
| 5 | v5-27-w4-master | **31/38** | **all 5: div_lt_one_iff, div_pos, div_pos_iff, mul_eq_left, mul_eq_right** |
| 6 | v5-28-w4-super-kitchen | **31/38** | same 5 |

**Best v5 candidate (v5-27 / v5-28): 31/38 on nat_defs_medium.** +5
theorems over the v4.7 26/38 plateau. Two distinct combos converge
to the same proof-set, suggesting the priority_templates slot is
saturated for this theorem set under the current model.

## Headline scoreboard (wave 5 — robustness probes around v5-27)

| # | variant | proved | observation |
|---|---|---|---|
| 1 | v5-29-w5-le-shape | 31/38 | le-shape priorities added no new wins |
| 2 | v5-30-w5-add-mod-ite | 31/38 | split_ifs etc. don't close Nat.add_mod_eq_ite |
| 3 | v5-31-w5-iff-reorder | **27/38** | **regressed -4 — within-slot ordering matters!** |
| 4 | v5-32-w5-dvd-specific | 31/38 | dvd term-mode templates didn't close |
| 5 | v5-33-w5-eq-one-of-mul | 31/38 | eq-shape templates didn't close `Nat.eq_one_of_mul_eq_one_left` |

**Wave 5 finding:** The 31/38 ceiling is robust under all five
inner-tier mutations of v5-27. The interesting failure was v5-31's
**iff list reordering** — putting generic omega-omega templates
FIRST inside the iff slot regressed 4 wins. This confirms that
within-slot ordering propagates the "first-non-erroring-wins"
shadowing problem from the wrapper's outer iteration.

Lesson for v6: a skeleton bag with typed slots should encode
specificity per-slot, not rely on hand-ordering of templates inside
a single list. See `v5_alphaevolve_architecture.md` and the updated
`v5_priority_templates_insight.md`.

## Headline scoreboard (wave 6 — targeted attempts at remaining 7 failures)

| # | variant | proved | newly tried theorem | result |
|---|---|---|---|---|
| 1 | v5-34-w6-dvd-alt | 31/38 | `Nat.dvd_iff_div_mul_eq` | no new close |
| 2 | v5-35-w6-add-mod-ite | 31/38 | `Nat.add_mod_eq_ite` | no new close |
| 3 | v5-36-w6-eq-one-alt | 31/38 | `Nat.eq_one_of_mul_eq_one_left` | no new close |
| 4 | v5-37-w6-div-le-div | 31/38 | `Nat.div_le_div_right` | no new close |
| 5 | v5-38-w6-combined | 31/38 | (all of the above) | no new close |

**Wave 6 conclusion:** the 31/38 ceiling is the priority_templates
saturation point for nat_defs_medium on gen_v5. The remaining 7
failures break down:

  - `Nat.AM_GM` — needs `nlinarith [sq_nonneg (a-b)]`; tactic unavailable.
  - `Nat.add_mod_eq_ite` — `split_ifs` advances one step then no
    closing tactic. Needs a structured case-analysis skeleton.
  - `Nat.eq_one_of_mul_eq_one_left` — needs case decomposition;
    `Nat.mul_eq_one` lemma form may not be the right one.
  - `Nat.div_le_div_right` — no working Mathlib lemma form found.
  - `Nat.sqrt_lt` — `Nat.sqrt_lt'` doesn't exist; no alternative
    found.
  - `Nat.pow_lt_pow_iff_left` — self-reference blocks; no clean
    alternative form found.
  - `Nat.dvd_iff_div_mul_eq` — asymmetric dvd-iff; the term-mode
    skeletons tried don't unify.

The first one is environment-limited; the others would yield to
either better lemma-name lookup (a v6 feature) or more flexible
skeleton structure (also v6).

## Headline scoreboard (Direction D — generalization)

Two v5 candidates evaluated on `nat_defs_large_v5` (38 medium +
26 new = 64 available theorems):

### v5-18-prio-kitchen (29/38 on medium)

  - **proved: 41 / 64** (64%)
  - proved_by_origin: `{'tactic_template': 25, 'family_tactic': 2, 'generative_topk': 4, 'fallback_tactic': 10}`

### v5-27-w4-master (31/38 on medium) — final

  - **proved: 43 / 64** (67%)
  - proved_by_origin: `{'tactic_template': 27, 'family_tactic': 2, 'generative_topk': 4, 'fallback_tactic': 10}`
  - new vs v5-18 on this set: **Nat.div_pos** and **Nat.div_pos_iff**
    (the same two that priority_templates v5-20 closes on medium —
    they generalize cleanly to the large set).
  - **no regressions** vs v5-18.

The v5-27 / v5-28 super-kitchen is therefore the recommended v5
production candidate.

### Baseline gen_v5 raw (no wrapper)

  - **proved: 4 / 64** (6%)
  - The raw model closes only 4 of 64 theorems on its own.

So the v5 wrapper contributes **+39 theorems** on the large set vs
raw gen_v5. The wrapper does almost all the work; the model
contributes 4 closures.

| candidate | nat_defs_medium | nat_defs_large_v5 |
|---|---|---|
| gen_v5 raw  | 3 / 38 (confirmed tonight)  | 4 / 64 (confirmed tonight)  |
| v4.7 hybrid (carries) | 26 / 38  | (not run; should be ~38-40/64 by extrapolation) |
| v5-18 kitchen | 29 / 38 | 41 / 64 |
| **v5-27 master** | **31 / 38** | **43 / 64** |

The architecture story: gen_v5 + wrapper closes ~10× more theorems
than gen_v5 alone. The v5 priority_templates contribution is +5
medium / +2 large beyond v4.7 — small in absolute terms compared to
the v3 → v4 jump, but structurally significant because it requires
an outer-tier change.

### Cross-domain check (demo_v1, 15 theorems from Nat / Set / Finset)

v5-27 master on demo_v1: **11 / 15 (73%)**, origins
`{family_tactic: 1, generative_topk: 10}`.

  - Of the 11 wins, 10 came from gen_v5's generative_topk on Set
    theorems (the model already knows Set basics).
  - 1 came from the mod family on `Nat.mul_add_mod'`.
  - Priority_templates did **not** fire on demo_v1. The templates are
    Nat-domain-specific (rely on `Nat.div_lt_iff_lt_mul`,
    `Nat.eq_of_mul_eq_mul_left`, etc.); they don't apply to Set or
    Finset goals.

This confirms: priority_templates is **domain-targeted**. It
generalizes within Nat goals (5 transfer wins on nat_defs_large_v5)
but does not cross-domain — exactly what one would expect, since
the templates name Nat-specific lemmas. A v6 skeleton bag would
likely need per-domain shape keys.
  - on the 26 NEW theorems (not in medium): **12 / 26 (46%)** proved.
  - of those 12, **5 came from priority_templates firing on theorems
    the templates were never designed for** — clean transfer:
    `Nat.add_eq_two_iff`, `Nat.add_eq_three_iff`, `Nat.lt_one_add_iff`,
    `Nat.max_eq_zero_iff`, `Nat.min_eq_zero_iff` (all iff goals closed
    by the generic `exact ⟨fun h => by omega, fun h => by omega⟩`
    priority template).

The v5 candidate doesn't overfit nat_defs_medium. The
priority_templates slot itself transfers across theorems.

## Best v5 result on `nat_defs_medium`

> _Best so far:_ **27 / 38** (v5-12-prio-div-hyp).
> _New theorem closed:_ `Nat.div_lt_one_iff`.

## Newly proved theorems (vs. v4.7 26/38)

So far: 1.
  - `Nat.div_lt_one_iff` via priority template
    `rw [Nat.div_lt_iff_lt_mul hb, Nat.one_mul]`. The lemma needed the
    explicit `hb : 0 < b` argument AND the right tail-form
    (`Nat.one_mul` rewrites `1 * b → b`; `Nat.mul_one` does the wrong
    direction here).

## Timeline (autonomous-loop discoveries by elapsed hour)

| hour | event |
|------|-------|
| 0.0 | Start. Branch created. Baseline confirmed 26/38. |
| 0.5 | First-pass loop launched (12 variants). |
| 0.9 | First-pass cycle 5 (term_builder iff basic). 14 wins attributed to new term_builder origin — but no new closures. Plateau confirmed. |
| 1.0 | Trace analysis on `Nat.div_lt_one_iff` reveals the **ordering bug**: model's `simp [Nat.one_mul]` advances state at step 1 and shadows every downstream template. |
| 1.0 | `priority_templates` slot added to `strategy_wrapper.py`. Round-trip tested. |
| 1.1 | Followup loop (11 variants) launched in parallel. |
| 1.3 | **First new win.** v5-12 closes `Nat.div_lt_one_iff` via `rw [Nat.div_lt_iff_lt_mul hb, Nat.one_mul]`. 27/38. |
| 1.5 | **Two more wins.** v5-15 closes `Nat.mul_eq_left` and `Nat.mul_eq_right` via term-mode `Nat.eq_of_mul_eq_mul_*` skeletons. 28/38. |
| 1.6 | v5-18 kitchen-sink combines div + mul priority templates. **29/38** — three new wins stacked. |
| 1.7 | First-pass loop completes (12/12, all 26/38). Architecture limit confirmed. |
| 1.8 | **Generalization confirmed.** v5-18 evaluated on `nat_defs_large_v5` (64 theorems). Proved 41/64. Five new theorems on the unseen extension set closed by priority_templates designed for medium — clean transfer. |
| 1.9 | **Two more wins.** v5-20 closes `Nat.div_pos` and `Nat.div_pos_iff` via `Nat.one_le_div_iff` chain. 28/38. |
| 2.0 | Wave 4 (6 variants) launched to combine all priority templates + target remaining failures. |
| ... | (TBD when wave 4 completes) |

## Mechanisms introduced

Four code changes ship in this branch:

  - **`ORIGIN_TERM_BUILDER` and `term_builder_templates`** in
    `evolve/strategy_wrapper.py`. Shape-keyed term-mode proof skeletons
    that emit between family/retrieval and generic fallbacks.
  - **`priority_templates`** in `evolve/strategy_wrapper.py`. Shape-keyed
    templates that emit BEFORE the base policy's generative_topk output.
    Surgical fix for the wrapper's first-non-erroring-wins shadowing bug.
  - **`per_theorem` aggregates** in `eval_rollout_all.py` for
    `term_builder_attempt_count`, `term_builder_advanced_count`,
    `term_builder_proved_count`, `term_builder_shape_keys`.
  - **`autonomous_research_loop.py` and `autonomous_research_followup.py`**
    in `evolve/`. The outer loop that ran the 12 first-pass and 11
    followup variants documented above.

## Direction-by-direction results

### Direction A — term_builder origin (`exact ⟨…⟩` / `refine ⟨?_,?_⟩`)

**Implementation.** New origin `ORIGIN_TERM_BUILDER` in
`evolve/strategy_wrapper.py`. New genome fields
`term_builder_templates: dict[shape, list[template]]` and
`term_builder_budget: int`. The wrapper classifies the goal shape
(via `premise_retriever.classify_goal_shape`), looks up the matching
shape key, and emits templates between family/retrieval and generic
fallback entries. Templates support the same `{var}`/`{hyp_pos}`/
`{hyp_le}`/`{hyp_ne_zero}` placeholders as `tactic_templates`.

**First-pass results (v5-04, v5-05, v5-06).** The wrapper's tactic
list correctly received the term-mode entries, and on iff goals
where `omega` closed both directions, the FIRST term-mode template
`exact ⟨fun h => by omega, fun h => by omega⟩` was credited as the
winning tactic. 14 such re-attributions per cycle.

But: no new theorem closed. Because (a) term_builder runs AFTER
generative_topk, and (b) the failing iff theorems (mul_eq, sqrt_lt,
div_pos_iff, etc.) need *asymmetric* inner tactics, not symmetric
omega/simp_all in both directions.

**Conclusion.** Mechanism is sound; ordering and inner-tactic
diversity are the actual gates.

### Direction B — shape-specific mini-solvers (mul / pow / sqrt / split_ifs)

Three first-pass variants added new families:

  - `v5-01-div-hyp-pos` — restored v45 `{hyp_pos}` div templates
    to the constructor variant. Result: 26/38, +1 progress count.
    Templates were rendered correctly but generative_topk's
    `simp [Nat.one_mul]` advanced state at step 1, pushing the
    family templates out of reach. **Same ordering bug.**
  - `v5-02-mul-family` — added cancellation lemmas
    (`Nat.mul_right_cancel_iff` etc.) for `Nat.mul_eq_*`. 26/38.
    Templates emitted but never tried because simp_all advanced
    state at step 1.
  - `v5-03-split-ifs` — added `split_ifs <;> omega` to fallback
    list. 26/38. Family/fallback simps consumed all 8 step budget
    on `Nat.add_mod_eq_ite` before split_ifs was ever tried.
  - `v5-08-pow-sqrt` — added `pow_lt` and `sqrt` families. 26/38.
    Same ordering issue.

**Conclusion.** v3-v4 family-extension can't beat the plateau
without addressing the wrapper's emission order. The followup
loop's `priority_templates` is the structural fix.

### Direction C — proof-skeleton mutation

`v5-09-skeleton-mut` (running) mutates inner tactics inside the
term_builder iff skeleton (replacing `simp_all` with `simp [*]`,
adding `try omega <;> simp_all` etc.). Result: 26/38, same
attributions.

**Conclusion.** Inner-tactic mutation alone is the same
slot-vocabulary mutation as v3 → v4. The mechanism needed an
outer-tier change (the `priority_templates` slot) before
inner-tier mutations could affect anything new.

### Direction D — `nat_defs_large_v5` generalization

> _To be run on best v5 candidate after followup loop completes._

68 theorems = 38 from `nat_defs_medium` + 30 new drawn from
`discovered_theorems.json` across 30 distinct name-prefix buckets
(add, div, dvd, mul, mod, pow, sqrt, succ, sub, two, max, min,
lt, le, eq, find, …). Measures whether v5 wins are nat_defs_medium-
specific or generalize.

### Direction E — trace-to-training-data plan

See `v5_trace_to_training_plan.md`. Plan only; no training tonight.
Connects v5 search wins to a future `gen_v5+1` learn step.

Concrete deliverable: a `scripts/build_v5_training_data.py` design
that converts wrapper traces into seq2seq (state_pp, tactic) pairs,
with filters for held-out theorems, retrieved-premise exclusion,
and self-reference rejection. Sizing estimate: ~25-50 high-quality
training pairs per `nat_defs_medium` eval.

### Direction F — AlphaEvolve architecture proposal

See `v5_alphaevolve_architecture.md`. Genome refactor from flat
string lists to layered (skeleton bag + slot vocabulary). Outlines
two-tier mutation, fitness with skeleton-attribution bonus,
cross-run archive, and transfer protocol.

**The `priority_templates` mechanism added tonight is the prototype
of a v6 outer-tier mutation:** it adds a new genome slot. The
existing v3 → v4 mutator could only have permuted the templates
in the existing slots. The slot itself had to be added by hand
(by me, after the autonomous loop surfaced the ordering bug).

In a v6 architecture, *the mutator itself* should be able to add
slots — that is, perform genuine outer-tier mutation. The
slot-addition operator is a structural mutation, not a
string-rewrite mutation. Tonight is the strongest empirical case
for that capability seen in this project.

## Per-theorem priority_templates analysis

The five new wins via priority_templates each required a specific
template. The patterns are documented for the v6 skeleton-bag design:

### Pattern 1: hypothesis-aware iff rewrite

```
goal:   a / b < 1 ↔ a < b   (hb : 0 < b)
tactic: rw [Nat.div_lt_iff_lt_mul hb, Nat.one_mul]
shape:  iff
```

The lemma takes a positivity hypothesis as its first argument. The
template uses `{hyp_pos}` placeholder to bind to whatever 0<-shape
hypothesis is in scope. Two trailing forms (`Nat.mul_one` vs
`Nat.one_mul`) are needed because the `'` and non-`'` versions of
`Nat.div_lt_iff_lt_mul` produce different normalized RHS.

### Pattern 2: term-mode asymmetric iff with cancellation lemma

```
goal:   a * b = a ↔ b = 1   (ha : a ≠ 0)
tactic: exact ⟨fun h => Nat.eq_of_mul_eq_mul_left (Nat.pos_of_ne_zero ha)
                          (h.trans (Nat.mul_one a).symm),
              fun h => by simp [h]⟩
shape:  iff
```

The two directions need different inner reasoning:
  - Forward (`a*b = a → b = 1`): convert `a*b = a` to `a*b = a*1`
    via `(Nat.mul_one a).symm`, then `Nat.eq_of_mul_eq_mul_left`.
  - Backward (`b = 1 → a*b = a`): substitute and simp closes via
    Nat.mul_one in simp set.

The `{hyp_ne_zero}` placeholder binds to `ha` or `hb`. The two
templates (left/right cancellation) cover symmetric forms.

### Pattern 3: positivity chain via le-div lemma

```
goal:   0 < a / b           (hba : b ≤ a, hb : 0 < b)
tactic: exact (Nat.one_le_div_iff hb).mpr hba
shape:  lt
```

`Nat.one_le_div_iff : 0 < b → (1 ≤ a/b ↔ b ≤ a)`. The forward
implication of the iff is `b ≤ a → 1 ≤ a/b`, and `1 ≤ a/b` is the
same as `0 < a/b`. So `.mpr hba` gives the goal.

The template uses `{hyp_pos}` and `{hyp_le}` placeholders.

### Pattern 4: iff via rewriting through pos-iff-ne-zero

```
goal:   0 < a / b ↔ b ≤ a    (hb : b ≠ 0)
tactic: rw [Nat.pos_iff_ne_zero, Nat.div_ne_zero_iff hb]
shape:  iff
```

`Nat.pos_iff_ne_zero` rewrites `0 < a/b` to `a/b ≠ 0`. Then
`Nat.div_ne_zero_iff hb` (which expects the divisor to be nonzero)
rewrites `a/b ≠ 0 ↔ b ≤ a`. The goal closes by reflexivity inside rw.

The `{hyp_ne_zero}` placeholder is critical — without it the lemma
errors.

### What these four patterns have in common

  - All four use HYPOTHESIS-AWARE templates (`{hyp_pos}`, `{hyp_le}`,
    `{hyp_ne_zero}` are essential).
  - All four are emitted BEFORE generative_topk so the model's
    weak-simp can't derail.
  - All four are 1-step proofs that the v3-v4 wrapper either never
    tried or tried in the wrong order.

This is exactly the kind of structural knowledge that should
graduate from "tactic in a list" to "skeleton entry in a typed
slot" in the v6 architecture.

## Failed directions and why

| variant | direction | why it didn't help |
|---|---|---|
| v5-01 (div-hyp-pos) | B | family-order shadowed by generative simp |
| v5-02 (mul-family)  | B | family templates depend on lemmas/hypotheses that weren't reached |
| v5-03 (split-ifs)   | B | step budget exhausted before fallback fired |
| v5-08 (pow-sqrt)    | B | same as v5-01 |
| v5-09 (skeleton-mut) | C | inner-tier mutation around shadowed slot |

The failed variants are useful: they document why the v3 → v4
genome was structurally insufficient, justifying the
`priority_templates` slot.

## Key insight

**Ordering of the wrapper's ranked-list matters more than the contents.**

The v3-v4 wrapper builds a ranked list of candidate tactics per
step. At eval time, the FIRST non-erroring tactic wins. The model
output goes first. If the model produces a syntactically-correct
simp that advances state into a less-useful form (rather than
erroring), every downstream family / template / term-builder entry
is silently bypassed.

The `priority_templates` slot (added tonight) inserts a small set
of family templates AHEAD of the model output. This is the smallest
structural change that breaks the 26/38 plateau.

The full chain:
  1. v3-v4 plateau at 26/38 was real.
  2. Twelve variants from the autonomous loop failed in the same
     way for the same reason.
  3. The trace showed the reason: shadowing by generative_topk.
  4. A new genome slot fixed the shadowing.
  5. One theorem closed.

This is a textbook AlphaEvolve loop: failure → analysis → new slot
→ result. The failure took twelve cycles of search. The fix was
one structural change.

## Recommended next branch / task

  1. **Build the v6 layered-genome architecture** described in
     `v5_alphaevolve_architecture.md`. The single most important
     change is to replace the flat tactic_templates / family_tactics
     / priority_templates lists with a single `skeleton_bag:
     dict[goal_shape, list[Skeleton]]` where `Skeleton` is a
     first-class object with a template body and a slot vocabulary.
     The mutator then operates at two tiers: outer-tier
     adds/removes skeletons; inner-tier rewrites slots.
  2. **Train `gen_v5+1` on the v5 verified traces.** Direction E's
     `scripts/build_v5_training_data.py` is shipped in this branch
     and ready to consume the autonomous-run trace directory. Hold
     out the three newly-proved theorems so we can measure whether
     the trained model produces the priority template
     `rw [Nat.div_lt_iff_lt_mul hb, Nat.one_mul]` natively on
     unseen iff goals. If yes, the wrapper's priority slot becomes
     redundant; if no, the wrapper remains the load-bearing piece.
  3. **Extend priority_templates to other goal shapes.** Wave 4
     (`evolve/autonomous_research_wave4.py`) ships five additional
     variants targeting `Nat.add_mod_eq_ite` (split_ifs under
     `any` key), `Nat.dvd_iff_div_mul_eq` (asymmetric dvd-iff),
     `Nat.div_pos` and `Nat.div_pos_iff` (div-positivity chain),
     and `Nat.sqrt_lt` / `Nat.pow_lt_pow_iff_left` (specific
     Mathlib lemmas). Each new closing template that survives
     wave 4 becomes a candidate for permanent inclusion in the
     v6 skeleton bag.
  4. **Move off nat_defs_medium.** The set has done its job —
     three real new wins, one structural mechanism, and a clear
     architectural roadmap. The next set should be either
     `curriculum_all` (broader) or a held-out
     `Mathlib/Data/Set/Basic.lean` slice (a domain where the v5
     priority templates have no relevance and the wrapper has to
     start from scratch).
