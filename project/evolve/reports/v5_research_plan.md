# v5 research plan — AlphaEvolve, deeper

The v3 → v4 work was a careful study of *tactic ordering*: which fallback,
which family, which retrieved lemma, in which order, for which goal shape.
That work succeeded — 0/38 → 26/38 — but has plateaued. Tactic ordering
is the wrong knob for what remains.

This document lays out where to go next, then ranks mechanisms by what
can actually be tested tonight without retraining.

---

## 1. What AlphaEvolve-style evolution has achieved so far

| stage | proved on `nat_defs_medium` | who/what closed the new goals |
|---|---|---|
| gen_v5 raw (no wrapper) | 3 / 38 | the base T5-small alone |
| v3.6 hybrid_evolved      | 25 / 38 | omega/simp_all fallbacks; mod family |
| v4.6 constructor seed    | 26 / 38 | div family + premise retrieval; one new div win (`Nat.div_lt_iff_lt_mul'`) |
| v4.7 evolution sweep     | 26 / 38 | family-budget tuning improved `progress_count`, no new closure |

The 22-theorem gap from baseline to wrapper is real, generalizes across the
nat_defs_subset / nat_defs_medium split, and survives ordering mutation.

## 2. The current bottleneck

Of the 12 unsolved theorems:

  - 4 are **iff** goals whose two directions need different inner tactics
    (asymmetric iff split).
  - 3 are **arithmetic** goals where the canonical close (`nlinarith`,
    `split_ifs`) isn't tried.
  - 3 need **division reasoning** the policy can't compose
    (`Nat.div_pos`, `Nat.div_pos_iff`, `Nat.div_le_div_right`).
  - 2 are **template-shaped** — they would yield to a single template
    that the constructor seed dropped during the v4.6 minimization
    (`Nat.div_lt_one_iff`, `Nat.pow_lt_pow_iff_left`).

Every category is structural. None is going to be fixed by reshuffling
the fallback list.

## 3. Why tactic-order evolution has plateaued

The mutator operates over six tactic-string lists and a small set of
integer knobs (top_k, family budgets, retrieval k). All of them affect
**which tactic is tried first / how many tactics are tried** on a single
proof state. None of them affects:

  - the **branching structure** of the proof (sequential vs split goals);
  - the **term-mode** vs tactic-mode encoding;
  - the **inner tactic per subgoal** when goals diverge;
  - the **proof-skeleton family** (constructor split / dvd witness /
    induction / case-analysis).

The genome is a flat list of strings. The proof we're searching for is
a tree. Mutation in string space cannot navigate tree space efficiently.

## 4. What "deeper AlphaEvolve" means in this project

AlphaEvolve, applied seriously, evolves *programs*: a population of
short proof-search programs, each of which calls a base policy and
arranges its outputs. The genome should contain things that affect
program *structure*, not just the order of leaf actions.

Concretely, "deeper" looks like:

  - **Proof-skeleton genome**. Each candidate carries a small set of
    skeletons keyed by goal shape: for iff, `⟨FORWARD, BACKWARD⟩`; for
    dvd, `⟨WITNESS, PROOF⟩`; for ite, `split_ifs <;> _`. Forward,
    backward, witness, proof are themselves picked from a small
    inner-tactic vocabulary that the mutator can rewrite.

  - **Two-tier mutation**. Outer-tier mutates which skeleton to try on
    which goal shape; inner-tier mutates the inner-tactic slots
    inside a skeleton. The outer tier is structural, the inner tier
    is the existing string-ordering loop in miniature.

  - **Origin discrimination in the trace**. The wrapper already tags
    `family_tactic` / `retrieved_premise` etc.; add `term_builder`,
    `shape_solver`, etc. so the evaluator can attribute wins to a
    specific mechanism and surface that to the next mutation.

  - **Archive across runs**. Successful (skeleton, theorem) pairs
    should be persisted so a future run starts from a richer skeleton
    pool, not from the v3.6 seed every time.

## 5. Plausible mechanisms

| # | mechanism                                  | testable tonight? | likely yield |
|---|--------------------------------------------|-------------------|--------------|
| 1 | restore dropped {hyp_pos} div templates    | yes, trivial      | +1-2 closures |
| 2 | new mul-family (`Nat.eq_of_mul_eq_mul_*`)  | yes               | +1-2 closures |
| 3 | new pow-family + sqrt template             | yes               | +0-1 closure  |
| 4 | term-mode proof skeletons (term_builder)   | yes, new code     | structural — direct closure unlikely on first try, but unlocks skeleton-mutation |
| 5 | shape-aware mini-solvers (iff / dvd / ite) | yes, new code     | +1-3 closures |
| 6 | proof-skeleton mutation (Direction C)      | yes, new code     | one knob beyond ordering |
| 7 | nat_defs_large generalization (Direction D) | yes, eval only    | confirm v5 doesn't overfit medium |
| 8 | learned strategy selection                  | no, needs training | future |
| 9 | retraining on verified traces              | no, out of scope   | future |
| 10| trace-to-training-data plan (Direction E)   | doc only           | future |
| 11| AlphaEvolve architecture proposal (F)       | doc only           | future |

## 6. Tonight's plan

Phase 1 — quick wins (Bucket I, ≤ 1h)
  - Restore the v45 `{hyp_pos}` div templates to the constructor variant.
  - Add a `mul` family with the cancellation lemmas.
  - Add `split_ifs <;> omega` to the generic fallback list.
  - Run one eval; expect 27-28 / 38.

Phase 2 — term_builder origin (Direction A, 1-2h)
  - Code: add `ORIGIN_TERM_BUILDER` and a goal-shape-aware emitter to
    the wrapper. Emit `exact ⟨fun h => by INNER1, fun h => by INNER2⟩`
    for iff, `refine ⟨_, ?_⟩` for dvd. Inner tactics are a small
    vocabulary (omega, simp_all, rfl, trivial, simp [LEMMA]).
  - Genome: add `term_builder_enabled`, `term_builder_inner_tactics` fields.
  - Run one eval against the seed; compare wins by origin.

Phase 3 — shape mini-solvers (Direction B, 1-2h)
  - For each goal shape (iff, dvd, mul-eq), declare a "mini-solver" —
    a sequence of templates targeted at that shape, gated by goal-shape
    detection (which the retriever already provides).
  - Compose with the existing family layer.

Phase 4 — skeleton mutation (Direction C, 1h)
  - Tiny mutation operators that rewrite a slot inside a term-mode
    skeleton: swap omega ↔ simp_all, swap iff direction order, replace
    `by INNER` with `LEMMA EXPR`.
  - Use the autonomous loop to sample 5-10 variants around any new
    win.

Phase 5 — large-set generalization (Direction D, 30 min if time)
  - Run the best v5 candidate on a 60-80-theorem extension of nat_defs.
  - Confirm wins didn't come from overfitting to medium.

Phase 6 — writeups (1h)
  - Final report.
  - AlphaEvolve architecture proposal (Direction F).
  - Trace-to-training plan (Direction E).

Stopping rules:
  - Hard floor 5 hours regardless of intermediate results.
  - Hard ceiling 8 hours.
  - Negative results documented; no late-stage refactor.
