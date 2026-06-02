# FLI3 Methodology

## Goal

Subject the 6 robust FLI2 retrieval-gap rescues (+ selected partials) to the same literal-RC2
additive validation discipline used for RC4A/RC4B/RC4C, and decide whether any become
VALIDATED_DEPLOYMENT_RULEs. Metric: TRUE_FLI3_DELTA count + safety, not solved count.

## Candidate extraction (Part 2)

All 6 robust TRUE_RETRIEVAL_GAP_RESCUE cases, each tagged with its family
(FINSET_CARD_BRIDGE / FINSET_MEM_DEF_UNFOLD / LIST_DEF_UNFOLD), lemma, tactic, controls-failed,
robust, non-vacuous flags. Selected high-quality PARTIAL_PROGRESS cases are carried as a *separate*
source (`FLI2_PARTIAL_PROGRESS`) and never mixed into the rescue attribution.

## Validation sets (Part 3)

- **rescue_replay** — the 6 robust rescue (theorem, action) pairs.
- **family_holdout** — related theorems from the FLI0/FLI2 pool sharing namespace + constant family
  + pattern (e.g. other `Finset.card_*`, other `Finset.mem_*` over map/filterMap/preimage/subtype).
  Tests whether the *family* (not just the exact theorem) deploys.
- **offgate_negative** — nearby theorems where the candidate must NOT fire (Finset without
  card/mem/preimage/filterMap/subtype constants; List without bidirectionalRec; namespace mismatch).
- **canonical_floor** — RC2 known-solved guard theorems.
- **regression_guard** — RC2-solved cases sensitive to simp/aesop perturbation.

Target 40–120 (theorem, action) items — focused, not a broad benchmark.

## Gates (Part 4)

Conservative per-family classifiers (`fli3_gate.py`): fire only when namespace matches AND a
trigger constant is present in the statement/residual AND the lemma is the matching
family-lemma/definition with constant overlap. Hard constraints: no broad namespace-only firing,
no unknown lemma, no root namespace, no Order family, no simp_all, no bare aesop as credited
candidate, no depth-3 chains. The offgate-negative set must yield 0 emissions (else NEEDS_REVIEW).

## Literal RC2 baseline (Part 5) — reuse-first

Failure-derived theorems (rescue_replay, family_holdout, offgate from the FLI0/RC5 corpus) carry
authoritative `rc2_result` (rc2_release wrapper, established RC5V2/V3) → reused as `failed`
(RC4B precedent). Floor/regression theorems use known RC2-solved status; their preservation is
guaranteed by additive design (candidate ≡ RC2 when the gate does not fire) and verified by a
gate=False check. Any theorem lacking a reusable status is run live via the `eval_rollout_all` RC2
harness.

## Additive candidate eval (Part 6) — at position, vacuity-safe

For each validation theorem where the gate fires and RC2 failed, open a LeanDojo Dojo at the
theorem's real file position and run, from the initial state: the bare controls
(simp / aesop / classical<;>aesop / constructor<;>simp / ext<;>simp) then the gated candidate
action(s). An additive win requires RC2-failed + gate-fired + candidate-solved + all-controls-failed
+ non-vacuous (deployed lemma ≠ target theorem). Candidate wins re-run once for robustness. Tight
per-tactic timeout (8s) avoids the slow-aesop tail. Offgate theorems are eval'd to confirm the gate
does not fire (0 emissions).

## Attribution (Part 7)

TRUE_FLI3_DELTA / CONTROL_DUPLICATE / BASELINE_DUPLICATE / OFFGATE_WIN /
UNKNOWN_NAME_OR_IMPORT_GAP / FLAKE / NEEDS_REVIEW, with the full control battery
(incl. `simp [L]`, `simp [L] <;> aesop`, `exact/simpa using L`, `gcongr`, `ext<;>simp`) so that a
delta is only TRUE when the *specific* lemma deployment is needed.

## Safety / determinism / schema (Part 8)

0 regressions (floor/regression preserved), 0 offgate emissions, deterministic repeat (gate is a
pure function; live re-run hash-stable modulo known flakes), wrapper-schema compatibility of a
*hypothetical* FLI3 fragment (validated, not installed), protected files untouched.

## Discovery↔validation comparison (Part 9)

FLI2 true-rescue count vs FLI3 true-additive-delta count; which rescues reproduce under literal-RC2
additive validation; family-holdout generalization; control-duplicate / import-gap fallout; and
whether failure-derived discovery can feed RC-style validation — the key project narrative.

## Determinism & safety

Non-live steps are pure functions of artifacts (sorted, no RNG/clock). Live steps checkpoint and
re-run wins for robustness. No protected file touched; schema fragment not installed; no commit.
