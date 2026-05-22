# v5 central findings — what the autonomous loop taught us

This document distills the autonomous-research-loop findings into a
small set of central claims. Each claim is supported by a specific
experimental cycle and includes a v6 design implication.

## Claim 1: The wrapper's ranked-list order is load-bearing

**Evidence:** v5 first-pass (12 variants, Directions A/B/C without
priority_templates) all plateaued at 26/38, despite adding new
templates, new family tactics, term-mode skeletons, and inner-tactic
mutations.

**Mechanism:** the wrapper's eval semantics is "first non-erroring
tactic wins". At step 1, generative_topk's output is tried first.
When the model emits a `simp [...]` that *advances* state into a
less-useful form (rather than erroring), every downstream template
is silently bypassed.

**Demonstration:** trace of `Nat.div_lt_one_iff` in v5-04-term-iff-basic.
The wrapper's emitted ranked list includes
`exact ⟨fun h => by omega, fun h => by omega⟩` and
`rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]` — but at step 1
`simp [Nat.one_mul]` (from gen_v5) advanced state to a form where
neither template applies. The whole proof never recovers.

**v6 implication:** the genome must encode template *priority* not
just *presence*. Tonight's `priority_templates` slot is the simplest
fix — a single dict that emits BEFORE the model — but the deeper
v6 design needs typed slots so the mutator can reason about which
template is "more specific" without manual hand-ordering.

## Claim 2: Outer-tier (slot-adding) mutation is qualitatively different from inner-tier

**Evidence:** the v3 → v4 mutator could only rewrite existing slot
contents. Twelve first-pass variants couldn't escape 26/38 by such
rewrites. One structural change (adding `priority_templates`) opened
a 5-theorem improvement to 31/38.

**Mechanism:** when the genome lacks a slot for "this should run
before the model", no within-slot mutation can express that
constraint. Adding the slot is a structural mutation; the within-slot
content is a state mutation.

**v6 implication:** AlphaEvolve should be able to add slots, not
just shuffle contents. The proposed v6 architecture
(`v5_alphaevolve_architecture.md`) defines this two-tier mutator
explicitly. Until then, slot additions are a human-in-the-loop
event — exactly what tonight required.

## Claim 3: Hypothesis-aware placeholders are necessary for tractable templates

**Evidence:** of the 5 newly-proved theorems, 4 use `{hyp_pos}`,
`{hyp_ne_zero}`, or `{hyp_le}` placeholders that bind to
hypothesis names in the proof state.

  - `Nat.div_lt_one_iff` (`hb : 0 < b`) — `rw [..., {hyp_pos}, ...]`
  - `Nat.mul_eq_left` (`ha : a ≠ 0`) — uses `{hyp_ne_zero}`
  - `Nat.mul_eq_right` (`hb : b ≠ 0`) — uses `{hyp_ne_zero}`
  - `Nat.div_pos` (`hba : b ≤ a, hb : 0 < b`) — uses `{hyp_pos}`+`{hyp_le}`
  - `Nat.div_pos_iff` (`hb : b ≠ 0`) — uses `{hyp_ne_zero}`

The fifth, the version of Nat.div_pos_iff rewritten via
`Nat.pos_iff_ne_zero`, also uses `{hyp_ne_zero}`.

**Mechanism:** Lean lemmas like `Nat.div_lt_iff_lt_mul` take a
positivity hypothesis as their first argument. The template
`rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]` binds at
emission time to whichever 0<-shape hypothesis is in scope.

**v6 implication:** the skeleton bag's slot vocabulary should be
*typed* by which placeholders it accepts. A skeleton declared for
`iff + hyp_pos` should not be tried on iff goals without a
positivity hypothesis.

## Claim 4: Lemma-name lookup is the slow bottleneck for new wins

**Evidence:** of the 12 unsolved theorems on nat_defs_medium, 5 closed
once the right Mathlib lemma name was added to the priority_templates
list. The other 7 remained unsolved primarily because we could not
identify the right lemma name within the time budget.

  - `Nat.sqrt_lt'` — name doesn't exist; no other obvious form found.
  - `Nat.pow_lt_pow_iff_left` — self-reference; no `Nat.pow_lt_pow_iff_right`
    that closes.
  - `Nat.div_le_div_right` — no clean Mathlib lemma found that closes.
  - `Nat.dvd_iff_div_mul_eq` — `Nat.div_mul_cancel` doesn't unify directly.

**v6 implication:** the system needs a *premise-retrieval-for-templates*
pipeline. The retriever already finds rw-shaped premises; extend it
to suggest term-mode skeletons too. When wave 4 failed to close
`Nat.sqrt_lt`, an LLM-driven mutator could have proposed alternative
lemma forms. That's the natural next step for the mutator beyond
the deterministic genome we have today.

## Claim 5: Term-mode mechanism is sound; effectiveness gated by ordering

**Evidence:** v5-04-term-iff-basic introduced the `term_builder`
origin and tagged 14 wins to it — all on iff theorems where
omega-omega closes. No net new wins because those theorems were
already closeable by `omega` in the fallback list.

In wave 4's v5-27 master, term-mode skeletons close 2 *new*
theorems (`Nat.mul_eq_left`, `Nat.mul_eq_right`) — but only because
they're inside the priority_templates slot, firing BEFORE
generative_topk.

**Mechanism:** term-mode is a mechanism, not a solution. Whether
it works depends on:

  - Where in the ranked list it sits (priority vs. post-model).
  - Whether the inner-tactic content is rich enough (omega-omega
    only closes trivial iffs; full lemma-based term-mode closes
    the harder ones).

**v6 implication:** term-mode is a skeleton in the bag; its slot
should be `iff + asymmetric` and its slot vocabulary should include
both omega and lemma-based inner tactics. The mutator should be
able to swap the inner tactics independently.

## Claim 6: Cross-theorem generalization is real; cross-domain is not (yet)

**Evidence:**

  - On nat_defs_large_v5 (38 medium + 26 unseen Nat theorems),
    v5-27 closes 12 of the 26 unseen theorems. **5 of those 12
    close via priority_templates** — the same generic
    omega-omega template on iff that closed easy theorems on
    medium fires on `Nat.add_eq_two_iff`, `Nat.add_eq_three_iff`,
    `Nat.lt_one_add_iff`, `Nat.max_eq_zero_iff`, `Nat.min_eq_zero_iff`.
    Clean transfer.
  - On demo_v1 (Set + Finset domain), v5-27 closes 11/15 —
    but **none** of the wins are from priority_templates. The
    templates name Nat-specific lemmas; they don't fire on Set
    goals.

**v6 implication:** transfer is per-domain. The skeleton bag should
have per-domain shape keys (or accept a domain tag at instantiation
time). A future `gen_v6` should hold out at least one whole
domain to test honest cross-domain transfer of skeleton vocabularies.

## Claim 7: gen_v5 raw is severely undermatched for nat_defs

**Evidence:**

  - gen_v5 raw on nat_defs_medium: 3 / 38 (~8%)
  - gen_v5 raw on nat_defs_large_v5: 4 / 64 (~6%)
  - v5-27 master: 31 / 38 (82%) and 43 / 64 (67%)

**Mechanism:** the wrapper does ~10× the work of the model on
nat_defs. The model is solid on Set goals (demo_v1: 10/12 Set
wins) but Nat-arithmetic goals overwhelm its training distribution.

**v6 implication:** the wrapper's contribution is the primary
research artifact. Fine-tuning gen_v5+1 on the v5 verified traces
(Direction E) tests whether the *model* can absorb what the wrapper
discovered. Until that experiment is done, the wrapper is the
production artifact.

## Summary table

| claim | strength | implication |
|---|---|---|
| Ranked-list order is load-bearing | strong (5 new wins from this) | priority_templates slot |
| Outer-tier mutation differs in kind | strong | v6 two-tier mutator |
| Hypothesis-aware placeholders are essential | strong | typed slot vocabulary |
| Lemma-name lookup is the bottleneck | medium | premise-retrieval-for-templates |
| Term-mode is mechanism not solution | medium | skeleton + slot vocabulary |
| Generalization is per-domain | medium | per-domain skeleton bag |
| gen_v5 alone is undermatched | strong (10× factor) | wrapper IS the system |

## Reading order for someone new to this branch

  1. `v5_research_plan.md` — what we set out to do.
  2. `nat_defs_medium_failure_classification_v5.md` — what was unsolved.
  3. `v5_priority_templates_insight.md` — the structural finding.
  4. `v5_autonomous_exploration.md` — the full scoreboard.
  5. **This document** — the distilled claims.
  6. `v5_alphaevolve_architecture.md` — v6 design proposal.
  7. `v5_trace_to_training_plan.md` — Direction E.
  8. `nat_defs_medium_summary.md` — the running history.
