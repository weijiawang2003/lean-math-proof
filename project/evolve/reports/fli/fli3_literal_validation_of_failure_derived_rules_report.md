# FLI3 — Literal Validation of Failure-Derived Deployment Rules

**Decision: `FLI3_VALIDATED_FAILURE_DERIVED_RULES`**

FLI3 subjected the 6 robust FLI2 retrieval-gap rescues to RC4-style literal-RC2 additive
validation. Result: **7 TRUE_FLI3_DELTA** (all 6 rescues reproduce + **1 family-holdout
generalization win**), **0 regressions, 0 offgate emissions, deterministic, schema-compatible**.
Failure-derived deployment actions *can* be converted into safe, reusable static candidate rules.
This is candidate validation only — nothing promoted, no wrapper/routing change, no commit.

## 1. Executive summary

- Extracted the **6 robust FLI2 rescues** (3 families: FINSET_CARD_BRIDGE, FINSET_MEM_DEF_UNFOLD,
  LIST_DEF_UNFOLD) + 29 partial-progress candidates (carried separately).
- Built a focused **55-item validation set**: rescue_replay 6, family_holdout 18, offgate_negative
  15, canonical_floor 8, regression_guard 8.
- Literal RC2 baseline: 39 failed (rescue + holdout + offgate), 16 solved (guards) — reuse-first
  from the authoritative FLI0/RC5 `rc2_release` corpus (RC4B precedent).
- At-position additive eval (vacuity-safe): **7 candidate wins, all robust** — **6/6 rescue_replay
  reproduced + 1 family_holdout generalization** (`List.bidirectionalRec_nil`, a theorem NOT in the
  original rescue set).
- **TRUE_FLI3_DELTA: 7.** Offgate emissions **0**, regressions **0**, gate deterministic
  (`2a61a261`), schema-smoke compatible (validated, **not installed**), protected files untouched.

## 2. Reminder of the original goal

Build a verifier-guided mathematical research assistant: Lean as verifier, failures as signal,
identify missing intermediate lemmas / deployments, test downstream rescue, and **convert
successful discoveries into reusable, validated proof-strategy candidates** — which is exactly the
FLI2→FLI3 bridge.

## 3. Why FLI3 follows FLI2

FLI2 produced 6 at-position rescues but explicitly flagged them as *not* production candidates —
they needed literal-RC2 additive validation, attribution, offgate/regression/determinism/schema
checks. FLI3 runs precisely that gauntlet.

## 4. Candidate extraction

6 robust TRUE_RETRIEVAL_GAP_RESCUE cases → families:
FINSET_CARD_BRIDGE (`Finset.card_le_one_iff`), FINSET_MEM_DEF_UNFOLD (`mem_filterMap`, `mem_map`,
`mem_preimage`, `card_subtype`), LIST_DEF_UNFOLD (`bidirectionalRec_singleton`). 29 partials carried
separately (`FLI2_PARTIAL_PROGRESS`), never mixed into rescue attribution.

## 5. Validation sets

55 items (Finset 46, List 9): rescue_replay 6, family_holdout 18, offgate_negative 15,
canonical_floor 8, regression_guard 8. Expected gate-fire 24, no-fire 31.

## 6. Literal RC2 baseline (reuse-first)

39 failed / 16 solved. Failure-derived items reuse authoritative `rc2_release` confirmations
(RC5V2/V3); guards assumed solved with preservation by additive design. No RC2 mutation.

## 7. Candidate evaluation

At each theorem's LeanDojo position (target + downstream out of scope; no self-import), controls
first then the gated candidate actions; wins re-run for robustness. **7 wins, all robust:**

| theorem | set | family | deployed | tactic |
|---|---|---|---|---|
| `Finset.card_le_one_iff` | rescue_replay | FINSET_CARD_BRIDGE | `Finset.card_le_one` | `simp [L] <;> aesop` |
| `Finset.mem_filterMap` | rescue_replay | FINSET_MEM_DEF_UNFOLD | `Finset.filterMap` | `simp [L]` |
| `Finset.mem_map` | rescue_replay | FINSET_MEM_DEF_UNFOLD | `Finset.map` | `simp [L]` |
| `Finset.mem_preimage` | rescue_replay | FINSET_MEM_DEF_UNFOLD | `Finset.preimage` | `simp [L]` |
| `Finset.card_subtype` | rescue_replay | FINSET_MEM_DEF_UNFOLD | `Finset.subtype` | `simp [L]` |
| `List.bidirectionalRec_singleton` | rescue_replay | LIST_DEF_UNFOLD | `List.bidirectionalRec` | `simp [L]` |
| **`List.bidirectionalRec_nil`** | **family_holdout** | LIST_DEF_UNFOLD | `List.bidirectionalRec` | `simp [L]` |

## 8. Attribution

`{TRUE_FLI3_DELTA: 7, BASELINE_DUPLICATE: 15 (guards/solved), GATE_NO_FIRE: 16 (offgate negatives),
NO_DELTA: 17 (holdouts that didn't deploy)}`. Each TRUE delta: literal RC2 failed, gate fired,
the specific lemma/def deployment solves, all bare controls (incl. `aesop`) fail, non-vacuous,
robust on repeat.

## 9. Safety and determinism

- **Offgate emissions: 0** (offgate_negative 15/15 no-fire; floor/regression 16/16 no-fire).
- **Regressions: 0** (additive design — candidate ≡ RC2 where the gate doesn't fire; verified).
- **Vacuous wins: 0** (deployed lemma ≠ target theorem in every win).
- **Deterministic:** gate is a pure function, hash `2a61a261` stable across runs; live wins re-run
  robustly.
- **Schema smoke:** a hypothetical FLI3 wrapper fragment (`priority_templates["any"]` additions +
  `theorem_name_tactic_gates` with `Finset.`/`List.` prefixes) merges onto a copy of the RC2 wrapper
  and serializes — **validated in memory, NOT installed**.

## 10. Discovery → validation comparison

**6/6 FLI2 rescues reproduce as TRUE_FLI3_DELTA under literal-RC2 additive validation
(reproduction rate 1.0)**, and **family holdout produced 1 generalization win**
(`List.bidirectionalRec_nil` via the same `simp [List.bidirectionalRec]` rule). So failure-derived
discovery not only reproduces but begins to *generalize* — the LIST_DEF_UNFOLD rule fired on a
sibling theorem it was never shown. The Finset families reproduced their exact rescues but their
holdouts did not deploy (honest: generalization is real but currently narrow). **Failure-derived
discovery can feed RC-style validation.**

## 11. Main findings

1. FLI discovery → RC validation is a working bridge: all 6 rescues survive the literal-RC2 additive
   gauntlet with 0 regressions / 0 offgate / determinism / schema compatibility.
2. The validated candidates span the RC4A (def-unfold) and RC4B/RC4C (lemma-bridge) patterns —
   automatically discovered, now literally validated.
3. One genuine family generalization (LIST_DEF_UNFOLD) appeared in holdout; Finset family
   generalization is not yet demonstrated.

## 12. Limitations

- **Literal RC2 reuse-first:** RC2 status reused from the authoritative corpus rather than re-run
  here (RC4B precedent); guards rely on additive-design preservation rather than a full floor
  benchmark.
- **Validation not full RC4-grade:** no full demo_v1/nat_defs floor benchmark run, no schema wrapper
  executed through the eval harness (only in-memory schema check). These remain for FLI4.
- **Narrow generalization:** 1 holdout win; Finset families reproduce exact theorems but holdouts
  didn't deploy — the gate may be too tight or the family genuinely doesn't generalize broadly.
- 24 gate-firing items evaluated; the broader FLI2 partials/availability-gaps were not validated here.

## 13. Recommended FLI4

1. Run the validated families through the **full RC4-style floor benchmark** (demo_v1 /
   nat_defs_medium / nat_defs_large_v5) and a **schema wrapper executed through `eval_rollout_all`**
   to upgrade from candidate to release-grade evidence.
2. **Investigate family generalization**: why Finset def-unfold/card holdouts don't deploy; widen or
   sharpen the gate; mine more sibling theorems per family.
3. Convert the validated FLI3 rules into a de-dup'd **RC4-style composition candidate** (vs RC4A/B/C)
   for an eventual RC release decision (owner-gated).
4. Validate selected FLI2 partial-progress cases and resolve the 42 availability-gap (UNKNOWN_NAME)
   imports.

## 14. Protected-file confirmation

`git diff --stat HEAD` over RC1/RC2/RC4-release/RC5S-policy wrappers + NS24 router = **empty**. No
RC*/TR*/FLI0/FLI1/FLI2 committed artifact, production wrapper, routing config, or README modified.
FLI3 wrote only under `project/evolve/experiments/fli3/`, `project/evolve/reports/fli/`, and
`scripts/fli3_*.py`. Nothing promoted, no wrapper installed (schema fragment in-memory only), ranker
not retrained, **no commit**.
