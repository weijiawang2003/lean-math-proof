# FLI3 — Literal Validation of Failure-Derived Deployment Rules

FLI3 bridges **FLI discovery → RC-style validation**. FLI2 discovered 6 robust retrieval-gap
rescues (failure-derived `simp [L]` deployments). FLI3 asks the research question:

> Can failure-derived deployment actions be converted into safe, reusable, *literal* static
> candidate rules over RC2?

This is **candidate validation only** — not a release, not a promotion. Nothing is installed into a
wrapper or routing config; no production file is modified; no commit.

## Definitions

- **FAILURE_DERIVED_CANDIDATE** — a proof action discovered from FLI failure analysis.
- **LITERAL_RC2_BASELINE** — the unmodified `rc2_release` wrapper on the validation set
  (reused authoritative confirmations where available, per the RC4B reuse-first precedent).
- **ADDITIVE_CANDIDATE_DELTA** — theorem failed by literal RC2 but solved by RC2 + the gated candidate.
- **TRUE_FLI3_DELTA** — an additive delta that is not a control duplicate, non-vacuous, gate-correct,
  deterministic, and non-regressive.
- **VALIDATED_DEPLOYMENT_RULE** — a candidate family passing attribution + offgate + regression +
  determinism + schema-smoke.

## Candidate families

| family | rescues | actions |
|---|---|---|
| FINSET_CARD_BRIDGE | card_le_one_iff | `simp [L]`, `simp [L] <;> aesop` |
| FINSET_MEM_DEF_UNFOLD | mem_filterMap/map/preimage, card_subtype | `simp [Finset.<def>]`, `… <;> aesop`, `ext x <;> simp [L]` |
| LIST_DEF_UNFOLD | bidirectionalRec_singleton | `simp [List.bidirectionalRec]`, `… <;> aesop` |

## Pipeline (scripts)

1. `fli3_extract_rescue_candidates.py` — 6 robust rescues (+ selected partials, marked separately).
2. `fli3_build_validation_sets.py` — rescue_replay / family_holdout / offgate_negative / canonical_floor / regression_guard.
3. `fli3_gate.py` — conservative per-family gates (namespace + constant + lemma match).
4. `fli3_run_literal_rc2.py` — literal RC2 baseline (reuse-first + live where needed).
5. `fli3_run_candidate_eval.py` — at-position additive eval (gate fires → deploy), controls, robustness.
6. `fli3_attribution.py` — TRUE_FLI3_DELTA / CONTROL_DUPLICATE / OFFGATE_WIN / … .
7. `fli3_safety_audit.py`, `fli3_determinism_check.py`, `fli3_schema_smoke.py`.
8. `fli3_compare_discovery_to_validation.py` — FLI2 discovery ↔ FLI3 validation.
9. `fli3_write_validation_atlas.py` — atlas.

Report: `project/evolve/reports/fli/fli3_literal_validation_of_failure_derived_rules_report.md`.

## Vacuity / safety

All candidate eval runs at the theorem's real file position (target theorem + downstream out of
scope; no fresh full-import). Candidate wins re-run for robustness. Gates are conservative
(no broad namespace firing, no simp_all, no bare aesop credited, no depth-3). Offgate-negative set
must produce 0 emissions. Schema smoke validates a hypothetical wrapper fragment **without
installing it**. No protected file touched; no commit.
