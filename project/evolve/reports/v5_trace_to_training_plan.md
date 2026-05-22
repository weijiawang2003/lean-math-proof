# Direction E — trace-to-training-data plan

**Scope:** plan only; no training tonight. This document describes how to
convert verified rollout traces into seq2seq training data so AlphaEvolve
search and gen_v5 fine-tuning close the loop (Learn → Search → Learn).

## Why now is the right moment to plan it

The v3 → v4 wrapper has produced ~1300 verified Lean transitions per
nat_defs_medium eval. Many of them are *high-quality positives* that the
existing gen_v5 dataset never saw:

  - `omega` closures on `Nat.add_eq_*`, `Nat.le_*`, `Nat.lt_*` — these
    already worked in gen_v5; the wrapper's contribution is the
    *retrieved-premise* and *family-tactic* transitions.
  - `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]` closures on
    `Nat.add_mod_eq_add_mod_left/right` — gen_v5 did not generate this
    tactic.
  - `simp_arith` on `Nat.lt_iff_add_one_le` — gen_v5 generated.
  - The lone `rw [Nat.div_lt_iff_lt_mul]` (retrieved_premise) closure
    on `Nat.div_lt_iff_lt_mul'` — gen_v5 never proposed this; the
    premise-retriever did.

The wrapper has surfaced tactics from outside gen_v5's distribution.
If we train on those, gen_v5+1 can generate them natively — the wrapper
becomes a *teacher*, not just a runtime crutch.

## What to convert, what to drop

Two kinds of transitions in the trace:

  - **Advancing**: `result_kind == TacticState`, num_goals_after ≤
    num_goals_before. The tactic moved the state forward.
  - **Closing**: `result_kind == Finished`, num_goals_after == 0.

Train on these; drop everything else.

Further filtering:

| filter                                  | rationale                                                     |
|----------------------------------------|----------------------------------------------------------------|
| origin in {fallback_tactic, family_tactic, retrieved_premise, generative_topk, term_builder} | exclude `SkippedBloatingApply` and other synthetic markers |
| not loop_detected                       | drop transitions to states already seen in the same theorem    |
| produced_seen_state == false            | drop redundant transitions                                     |
| theorem not in held-out set             | preserve a fair eval slice                                     |
| tactic length ≤ 200 chars               | reject pathological one-shot encodings (`exact ⟨very long⟩`)  |
| state_pp length ≤ 2500 chars            | T5-small input-length budget                                   |
| not theorem-specific hack (manual flag) | exclude wins that only work for one theorem name              |

The last filter is the one that requires care. If we ship a template
`exact ⟨_, by simp [Nat.dvd_iff_div_mul_eq]⟩` and it closes
`Nat.dvd_iff_div_mul_eq` *only because the theorem is its own
substitutand*, training on it teaches the model nothing transferable.
The training pipeline should reject (theorem, tactic) pairs where the
tactic string contains the theorem's own full_name.

## Output format

Use the existing `seq2seq_data_v*.jsonl` schema so the existing
`train_tactic_generator.py` ingests it without changes:

```json
{
  "prompt": "Theorem: {full_name}\n\nProof state:\n{state_pp}\n",
  "completion": "{tactic}",
  "origin": "fallback_tactic|family_tactic|retrieved_premise|term_builder|generative_topk",
  "theorem": "{full_name}",
  "file_path": "Mathlib/Data/Nat/Defs.lean",
  "domain": "mathlib4",
  "source_run_id": "{eval_run_id}"
}
```

A separate JSON header records:

  - source eval run_id(s)
  - filter set used
  - per-origin counts
  - per-theorem counts
  - md5 of the resulting jsonl

so the dataset is reproducible from logs alone.

## Sizing — measured

After running the v5 autonomous loops, the actual dataset size is
**157 unique (state, tactic) pairs** across 29 distinct theorems
(after held-out filtering of the three new wins). Origin breakdown:

| origin            | pairs |
|-------------------|-------|
| fallback_tactic   | 64    |
| tactic_template   | 36    |
| term_builder      | 31    |
| generative_topk   | 20    |
| family_tactic     | 6     |
| **total**         | **157** |

The 36 `tactic_template` pairs are the priority_templates pattern
the wrapper discovered; teaching the model to natively generate
these is the v5 → gen_v5+1 graduation hypothesis. The 31 term_builder
pairs are the asymmetric term-mode iff splits that the wrapper
attributes; same hypothesis.

This is ~0.15% of the existing 103k transitions — it will not shift
the underlying distribution, but it adds *new* tactic forms the
existing data doesn't contain.

## Connection to AlphaEvolve

The full Learn → Search → Learn cycle is:

```
                    ┌──────────────────┐
       v5 corpus    │  AlphaEvolve     │  generates new tactics
   ────────────────►│  search / wrapper├────────────┐
                    └──────────────────┘            │
                            ▲                       │
                            │  verified traces      ▼
                            │                ┌────────────────┐
                            └────────────────│   training     │ → gen_v5+k
                                             │   pipeline     │
                                             └────────────────┘
```

Each Learn step shrinks what the wrapper has to do at runtime: tactics
that the model can now propose natively don't need the
fallback/family/retrieval layer. The wrapper's measured contribution
should *decrease* over Learn iterations — that's the success signal.

## Tonight's deferred steps (concrete)

1. `scripts/build_v5_training_data.py` is shipped in this branch. It
   reads `project/evolve/autonomous_runs/<run_id>/eval/**/traces.jsonl`,
   applies the filters above, writes
   `project/seq2seq_data_v5_evolve.jsonl` and a header JSON. Default
   held-out set includes the three new theorems v5 closed
   (`Nat.div_lt_one_iff`, `Nat.mul_eq_left`, `Nat.mul_eq_right`) so the
   v5 → v6 training cycle has a fair test.
2. Hold out 4 theorems from nat_defs_medium for fair eval
   (suggest: `Nat.div_lt_iff_lt_mul'`,
   `Nat.add_mod_eq_add_mod_left`, `Nat.mod_two_ne_zero`,
   `Nat.succ_succ_ne_one`).
3. After collecting ≥500 new pairs, train `gen_v5+1` on
   (original gen_v5 data) ∪ (v5 evolve data) and re-run
   nat_defs_medium.
4. Compare: does **raw** `gen_v5+1` (no wrapper) close more than
   gen_v5 raw? That is the only honest measure of whether the wrapper
   taught the model anything.

## Risks

  - **Memorization, not transfer.** If the wrapper closes `Nat.div_pos`
    via a hard-coded skeleton, training on that exact (state_pp, tactic)
    pair teaches the model the skeleton for that exact state, not the
    underlying tactic. Mitigation: hold out the originating theorem from
    the eval set, and check the model proposes the tactic *on other
    theorems* with similar state.
  - **Distribution shift.** v5 corpus is `nat_defs_medium`-centric;
    seq2seq trained on it may regress on `Set.Basic` / `Finset.Basic`
    that gen_v5 already proves. Mitigation: merge with original corpus,
    don't replace.
  - **Self-amplification.** If wrapper wins are partially due to the
    retrieval engine, training the model to predict the retrieved tactic
    without the retriever in the loop will fail at inference. Treat
    `retrieved_premise` origin as a special case — either include the
    premise in the prompt, or exclude these transitions until the model
    has its own retrieval.
