# FLI3 Part 0 — State Reconciliation

_Read-only inspection of FLI2 inputs. No FLI2 output altered._

## Repo state

- **HEAD:** `d2a1b95` ("Scale retrieval-gap lemma deployment in FLI2"), branch
  `tr5-ranker-guided-live-search`. FLI0/FLI1/FLI2 committed.
- **Dirty/untracked:** the in-progress RC5V3 raw run (modified `scripts/rc5v3_*.py`,
  `rc5_v3/out/*`). Not FLI3's; untouched.

## FLI2 artifacts (verified present, unmodified)

| artifact | status |
|---|---|
| `fli2/cases/fli2_rescue_attribution.jsonl` | 1,059 rows |
| `fli2/cases/fli2_live_deployment_results.jsonl` | present |
| `fli2/data/fli2_deployment_rules.json` | 25 rules, 1 candidate |
| `fli2/out/fli2_rescue_attribution_summary.md` / `..._atlas.md` / report | present |

## Key counts

- **TRUE_RETRIEVAL_GAP_RESCUE: 6 distinct theorems** (11 robust actions): `Finset.card_le_one_iff`,
  `Finset.mem_filterMap`, `Finset.card_subtype`, `Finset.mem_map`, `Finset.mem_preimage`,
  `List.bidirectionalRec_singleton`. All robust, controls failed, non-vacuous.
- **PARTIAL_PROGRESS: 31 theorems.**
- **UNKNOWN_NAME / availability-gap: 42 theorems** (lemma not in scope at position).
- CONTROL_DUPLICATE 2, NEEDS_REVIEW 1, NO_RESCUE 880.

## Candidate families available for validation

| family | rescues | deployment |
|---|---|---|
| FINSET_CARD_BRIDGE | `Finset.card_le_one_iff` | `simp [Finset.card_le_one] <;> aesop` (RC4B/RC4C-style lemma bridge) |
| FINSET_MEM_DEF_UNFOLD | `mem_filterMap`/`mem_map`/`mem_preimage`, `card_subtype` | `simp [Finset.<def>]` (RC4A-style def-unfold) |
| LIST_DEF_UNFOLD | `List.bidirectionalRec_singleton` | `simp [List.bidirectionalRec]` |

## Literal-RC2 status (reuse-first, RC4B precedent)

All 6 rescues carry `rc2_result = failed` and `rc4_result = failed` in the FLI0 enriched corpus —
i.e. confirmed literal-RC2 (`rc2_release` wrapper) failures, established in RC5V2/V3. Per the RC4B
"reuse-first" pattern, these authoritative statuses are reused rather than re-run; family-holdout
and offgate theorems are likewise drawn from the FLI0/RC5 RC2-failure corpus. Canonical-floor /
regression-guard cases use known RC2-solved status and are validated by **additive design** (the
candidate only adds gated actions; on non-firing theorems candidate ≡ RC2, so a gate=False check
suffices to guarantee floor preservation).

## Tooling availability

- **Live LeanDojo:** yes (Dojo ~3.7s; `next_state.pp`). FLI3 candidate eval runs at theorem
  position (vacuity-safe), reusing the FLI2 worker pattern with tight timeouts (8s/tactic) to avoid
  the slow-aesop tail.
- **`lake env lean`:** yes (FLI1/FLI2 used it) — available for schema/typecheck sanity if needed.
- **`eval_rollout_all` literal-RC2 harness:** available (RC4A/B/C precedent) for any live RC2
  confirmation.

## Decision

Proceed with FLI3. The 6 robust rescues are the validation targets; literal-RC2 baseline is
authoritative (reused failures); the new measurement is whether the gated candidate yields a
TRUE_FLI3_DELTA (additive, controls-fail, gate-correct, deterministic, non-regressive) — RC-style.
