# v4.6 — div-family template failure diagnostics

Source: `project/evolve/runs/evolve-20260522-050325-0fe236/eval/seed-baseline/eval-fa216dba/traces.jsonl`
(v4.5 seed-baseline run, 25/38 proved, 0 div-family closures).

This document walks every template currently shipped in the v4.5
`theorem_family_tactics["div"]` bucket, lists its observed outcomes on
the six div/dvd targets, and assigns a v4.6 verdict:

  - **keep** — useful tactic; produces an advance on at least one
    theorem, or is a cheap noop on the rest.
  - **disable** — fails on every observed call; references a constant
    in `_UNAVAILABLE_LEMMAS` or `_TYPE_MISMATCH_CONSTANTS`. Filtered out
    by `evolve/template_verifier.py`.
  - **rewrite** — the intent is reasonable but the template form is
    wrong (often the wrong hypothesis placeholder, or wrong lemma form).
  - **theorem-specific** — useful only for an iff goal / a specific
    div lemma; gate by family-key narrowing in a later iteration.

## Trace census (v4.5 run, family_tactic origin only)

| target                       | family-tactic attempts | proved | dominant error                          |
|------------------------------|------------------------|--------|------------------------------------------|
| Nat.div_le_div_right         | 107                    | no     | simp made no progress / apply unify fail |
| Nat.div_lt_iff_lt_mul'       | 117                    | no     | simp made no progress / unknown constant |
| Nat.div_lt_one_iff           | 85                     | no     | simp made no progress / constructor fail |
| Nat.div_pos                  | 25                     | no     | simp made no progress / constructor fail |
| Nat.div_pos_iff              | 25                     | no     | simp made no progress / constructor fail |
| Nat.dvd_iff_div_mul_eq       | 100                    | no     | simp made no progress / constructor fail |

`Nat.div_le_div_right` and `Nat.left_comm` are the two unique constants
reported with `unknown constant '...'` at any v4.5 call site (26 and 24
hits respectively across all theorems).

## Per-template verdict

| # | template                                                                                                            | observed result kind(s)              | verdict           |
|---|---------------------------------------------------------------------------------------------------------------------|--------------------------------------|-------------------|
| 1 | `omega`                                                                                                             | LeanError (omega could not prove)    | keep              |
| 2 | `simp`                                                                                                              | LeanError (made no progress)         | keep              |
| 3 | `simp_all`                                                                                                          | LeanError / TacticState              | keep              |
| 4 | `simp [Nat.div_eq_of_lt]`                                                                                           | LeanError (made no progress)         | keep              |
| 5 | `simp [Nat.div_eq_of_lt, Nat.lt_of_lt_of_le]`                                                                       | LeanError (made no progress)         | keep              |
| 6 | `rw [Nat.div_eq_of_lt]`                                                                                             | TacticState (pattern not found)      | keep              |
| 7 | `rw [Nat.div_lt_iff_lt_mul']`                                                                                       | LeanError (pattern not found)        | keep              |
| 8 | `rw [Nat.div_lt_iff_lt_mul]`                                                                                        | LeanError (pattern not found)        | keep              |
| 9 | `rw [Nat.div_le_iff_le_mul]`                                                                                        | LeanError (proof expected)           | **disable** (constant in `_UNAVAILABLE_LEMMAS`)    |
| 10 | `exact Nat.div_le_div_right ‹_›`                                                                                   | LeanError (type mismatch / unknown)  | **disable** (constant in `_UNAVAILABLE_LEMMAS`)    |
| 11 | `apply Nat.div_le_div_right`                                                                                        | LeanError (apply / unknown)          | **disable** (constant in `_UNAVAILABLE_LEMMAS`)    |
| 12 | `simp [Nat.div_lt_iff_lt_mul, Nat.mul_one]`                                                                         | LeanError (made no progress)         | keep / **suspect**: never advances |
| 13 | `simp_all [Nat.div_lt_iff_lt_mul, Nat.mul_one]`                                                                     | LeanError (made no progress)         | keep / **suspect** |
| 14 | `simp_all [Nat.div_lt_iff_lt_mul', Nat.mul_one]`                                                                    | LeanError (made no progress)         | keep / **suspect** |
| 15 | `rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]`                                                                 | LeanError (pattern not found)        | rewrite (need `≤ 1` form, not `< b * 1`) |
| 16 | `constructor <;> intro h_split <;> omega`                                                                           | LeanError (not iff / omega fail)     | theorem-specific (iff only) |
| 17 | `constructor <;> intro h_split <;> simp_all`                                                                        | LeanError (not iff / no progress)    | theorem-specific (iff only) |
| 18 | `induction {hyp_le} <;> simp_all`                                                                                   | LeanError (simp_all no progress)     | rewrite           |
| 19 | `induction {hyp_le} with | refl => exact Nat.le_refl _ | step h_step ih => exact ih.trans (Nat.div_le_succ_div _ _)` | LeanError (type mismatch on `Nat.le_refl`) | **disable** (constant in `_TYPE_MISMATCH_CONSTANTS`) |

Items 9-11 + 19 are the four templates removed by
`evolve/template_verifier.verify_template`. None of them ever advanced
a div goal in v4.1–v4.5.

## Constants seen vs. environment

| constant                  | category                               | action                                     |
|---------------------------|----------------------------------------|--------------------------------------------|
| `Nat.div_eq_of_lt`        | available                              | keep                                       |
| `Nat.lt_of_lt_of_le`      | available, but `apply` bloats goals    | retained, retrieval-side filter handles bloat |
| `Nat.div_lt_iff_lt_mul`   | available                              | keep                                       |
| `Nat.div_lt_iff_lt_mul'`  | available                              | keep                                       |
| `Nat.mul_one`             | available                              | keep                                       |
| `Nat.div_le_div_right`    | unavailable at the v4.5 eval positions | filter via `template_verifier`             |
| `Nat.div_le_iff_le_mul`   | unavailable in nat_defs_medium closure | filter via `template_verifier`             |
| `Nat.le_refl`             | available, but wrong arity in our use  | filter via `template_verifier`             |
| `Nat.div_le_succ_div`     | wrong arity / does not match shape     | filter (caught by `Nat.le_refl` in same template) |
| `Nat.left_comm`           | unavailable                            | already in fallback list; left in place but filtered by verifier if added to family templates |

## v4.6 variant candidates (sketch — concrete config in run_evolve)

  **A (v4.5 reference)** — no code or config changes; baseline 25/38.

  **B (verified-conservative)** — exactly v4.5's div family minus the 4
  filtered templates (#9, #10, #11, #19). Expected: equal proved count,
  cleaner error log, fewer wasted Lean roundtrips.

  **C (constructor-only)** — div family = `[constructor <;> intro h
  <;> simp_all, constructor <;> intro h <;> omega, omega, simp_all]`
  only. Tests whether iff-shaped div theorems benefit from the iff
  constructor decomposition alone.

  **D (div-rewrite)** — div family = `[simp [Nat.div_eq_of_lt],
  rw [Nat.div_eq_of_lt], rw [Nat.div_lt_iff_lt_mul'], simp [Nat.div_eq_of_lt,
  Nat.lt_of_lt_of_le], simp_all, omega]`. Tests whether the rewrite-side
  templates alone close any div goal.

  **E (mixed-small)** — union of C + D plus the cheap base
  `[omega, simp, simp_all]`. Tests whether the union is strictly better
  than either subset.
