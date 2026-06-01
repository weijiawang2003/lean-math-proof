# RC3 Candidate — SX3_SET_ITE_AESOP (experimental, off-by-default)

**Candidate:** `RC3 candidate = RC2 ⊕ SX3_SET_ITE_AESOP`
**Added sequence:** `simp [Set.ite] <;> aesop` (gated to theorem names containing `Set.ite`)
**Status:** `RC3_CANDIDATE_CONFIRMED` — experimental, **not** RC3 production, owner approval pending.

> This is NOT wired into `project/evolve/routing/ns24_router.json` and does not modify the
> RC2 release artifacts. It is a standalone candidate wrapper for evaluation only.

## Why SX3 is separate from RC2

- **RC2 official delta** = single-shot `simp [Set.ite]` (+5 credited: `Set.ite_empty`,
  `Set.ite_empty_left`, `Set.ite_empty_right`, `Set.ite_left`, `Set.ite_right`).
- **SX3 candidate delta** = depth-2 sequence `simp [Set.ite] <;> aesop`.

For the four RC2-deferred theorems, `bare aesop`, `simp_all`, and **single-shot
`simp [Set.ite]` all fail**, while `simp [Set.ite] <;> aesop` succeeds — so this is a
genuine depth-2 enabling step, attributed separately and never folded into RC2's +5.

## Live evidence (LeanDojo, `Mathlib/Data/Set/Basic.lean`)

| surface | result |
|---|---|
| deferred +4 reproduction | **4/4** solved by `simp [Set.ite] <;> aesop`; all controls fail (hash `c0144cd63fd5`, reproduced ≥2×) |
| fresh Set.ite/dite holdout (13) | **+1 fresh true win:** `Set.ite_inter_inter`; `Set.ite_univ` = single-shot duplicate (→ RC2, excluded); rest baseline/no-win (hash `ed6b9ef789a0`) |
| general Set cluster (11 live) | **0** depth-2 wins from ext/iff/subset-antisymm families → broad sequence search does NOT generalize |
| negative controls (Nat/Int/Multiset/List) | **0** Set-sequence emissions |
| canonical smoke (Nat/Bool/List) | **0** Set-sequence emissions |

- **TRUE_DEPTH2_SEQUENCE_WIN = 5** (4 deferred + 1 fresh)
- **off-gate emissions = 0**, **regressions vs RC2 = 0**, **deterministic**

## Files

- `rc3_candidate_wrapper.json` — RC2 wrapper + the gated depth-2 sequence (functional copy; RC2 untouched).
- `sx3_set_ite_aesop_gate.json` — gate definition + wrapper-expression details.
- `component_summary.json` — delta, safety, attribution rollup.
- `out/rc3_candidate_eval_results.json` — per-theorem candidate behaviour vs RC2.
- `out/rc3_candidate_comparison.json` — RC2-vs-candidate delta + verdict inputs.

## How the candidate is expressed in the wrapper

The sequence is a single-line grouped tactic (`<;>`), which `env.run_transition` accepts as one
transition, so it slots into `priority_templates["any"]` exactly like RC2's single-shot
`simp [Set.ite]`, gated by `theorem_name_tactic_gates["simp [Set.ite] <;> aesop"] = ["Set.ite"]`.
It is placed **after** the single-shot entry so RC2's credited mechanism is always tried first;
the depth-2 sequence only contributes when single-shot fails. The wrapper expresses depth-2
sequences cleanly — `NEEDS_SEQUENCE_WRAPPER_SUPPORT` does **not** apply.

## Decision

`RC3_CANDIDATE_CONFIRMED`: positive fresh delta (+1), all 4 deferred reproduced, 0 off-gate,
0 regressions, deterministic, minimal attribution confirms, wrapper expresses the sequence cleanly.
Keep narrow `Set.ite` gate only; do not promote broad sequence search. Next: owner approval →
RC3 literal-wrapper validation + full canonical floors before any production status.
