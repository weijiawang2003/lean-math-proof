# SX4 — Sequence Attribution Harness

**Branch:** `sx3-depth2-sequence-search`  ·  **Date:** 2026-05-30  ·  **No commit made.**
Builds a reusable, regression-tested harness that credits a depth-k sequence candidate **only** when
it yields a genuine new win over **literal production search**.

---

## 1. Executive summary

- **Why SX4 exists.** The RC3 candidate (`RC2 ⊕ SX3_SET_ITE_AESOP`, sequence `simp [Set.ite] <;> aesop`)
  was initially credited +5 by the custom SX3 runner, then **rejected** under literal-production
  validation (`REJECT_NO_LITERAL_DELTA`): literal RC2 = literal RC3 = 17/30, credited delta **0**. The
  +5 were a **methodology bug** — depth-1 controls cannot see a depth-2 search close.
- **What the harness implements.** `scripts/sx4_sequence_attribution.py` classifies every theorem into
  `TRUE_SEQUENCE_DELTA | PRODUCTION_SUBSUMED | DEPTH1_DUPLICATE | ROUTING_DUPLICATE | TRACE_INSUFFICIENT
  | FAILED_SEQUENCE | NEEDS_REVIEW`, crediting **only** `TRUE_SEQUENCE_DELTA`. The decisive new check:
  **does the literal-production trace already reach an `A`-advanced state and apply `B`?** A heuristic
  trace detector (`scripts/sx4_trace_sequence_detector.py`) provides corroborating evidence.
- **Regression test result.** Re-running SX4 on `SX3_SET_ITE_AESOP` reclassifies all **5** previously
  credited wins as **`PRODUCTION_SUBSUMED` (credit=false)**, `over_credit_caught = True`,
  `num_credited = 0`. **The harness catches the RC3 over-credit bug.**

---

## 2. Methodology problem

**Naive depth-2 attribution** (what the SX3 runner used): credit `A <;> B` if (1) `A <;> B` succeeds,
(2) `A` alone fails, (3) `B` alone fails — all measured single-shot from the **initial** goal.

**Why it over-credited `SX3_SET_ITE_AESOP`.** A best-first search with `max_steps > 1` applies `A` at
step *i* (advancing the goal, not closing it) and `B` at step *i+1* from the advanced state. Those two
ordinary search steps **are** `A <;> B`. `B` failing *on the initial goal* is irrelevant — production
never applies `B` to the initial goal. For `simp [Set.ite] <;> aesop` on `Set.ite_inter`, the literal
RC2 trace is: step 1 `simp [Set.ite]` → `TacticState` (advanced); step 2 `aesop` → `ProofFinished`.

**Literal-production-baseline requirement.** A sequence is a genuine delta only if the **literal
production run** (same `max_steps`/`top_k`) does **not** already solve the theorem, **and** its trace
contains no equivalent `A`-advanced → `B`-close continuation. See `sx4_methodology.md` for the full
argument and definitions.

---

## 3. Attribution schema

`project/evolve/experiments/sx4/sequence_attribution_schema.json`. Per-theorem record carries
`baseline_finished`, `candidate_finished`, `sequence_runner_solved`, `controls{A_initial,B_initial,
A_then_B,baseline_controls_solved}`, `production_trace_analysis{baseline_reaches_A_equivalent_state,
baseline_applies_B_after_A_state, equivalent_sequence_observed, baseline_winning_path,
trace_confidence}`, `classification`, `credit`, `notes`.

**Categories & credit:**

| class | when | credit |
|---|---|---|
| `TRUE_SEQUENCE_DELTA` | baseline fails, candidate solves, depth-1 controls fail, **no** equivalent prod `A`→`B` continuation, trace not insufficient | **true** |
| `PRODUCTION_SUBSUMED` | literal production already solves (same/equivalent intermediate-state continuation) | false |
| `DEPTH1_DUPLICATE` | `A` alone or `B` alone solves | false |
| `ROUTING_DUPLICATE` | baseline (not finished) closed by an equivalent generic family | false |
| `TRACE_INSUFFICIENT` | logs cannot distinguish | false (default) |
| `FAILED_SEQUENCE` | candidate does not solve | false |
| `NEEDS_REVIEW` | none cleanly applies | false |

Rules are evaluated **in order**; `baseline_finished` is checked **first**, so any production-solved
theorem is `PRODUCTION_SUBSUMED` regardless of how a proxy runner scored it. Default-to-no-credit for
`TRACE_INSUFFICIENT` / `NEEDS_REVIEW`.

---

## 4. `SX3_SET_ITE_AESOP` re-analysis (regression test)

Inputs: literal RC2 (`literal_rc2_results.json`) as baseline, literal RC3 (`literal_rc3_results.json`)
as candidate, `sx3_minimal_attribution.json` as the proxy sequence-runner result.
Outputs: `project/evolve/experiments/sx4/out/sx3_set_ite_aesop_reattribution.{json,md}`.

| metric | value |
|---|---|
| theorems analyzed | 39 (union of all result sets) |
| classification histogram | `PRODUCTION_SUBSUMED: 17`, `FAILED_SEQUENCE: 22` |
| **credited TRUE_SEQUENCE_DELTA** | **0** |
| proxy runner credited (SX3 "wins") | 5 (`Set.ite_inter`, `_inter_self`, `_compl`, `_inter_compl_self`, `_inter_inter`) |
| **over_credit_caught** | **True** — all 5 proxy-credited reclassified to non-credit |

**The 5 previously-credited wins → `PRODUCTION_SUBSUMED`**, each with
`equivalent_sequence_observed = true`, `trace_confidence = full`, baseline winning path
`['simp [Set.ite]', 'aesop']`:

| theorem | prior SX3 class | literal RC3 validation | **SX4 class** | credit |
|---|---|---|---|---|
| `Set.ite_inter` | TRUE_DEPTH2_SEQUENCE_WIN | RC2_ALREADY_SOLVED | **PRODUCTION_SUBSUMED** | — |
| `Set.ite_inter_self` | TRUE_DEPTH2_SEQUENCE_WIN | RC2_ALREADY_SOLVED | **PRODUCTION_SUBSUMED** | — |
| `Set.ite_compl` | TRUE_DEPTH2_SEQUENCE_WIN | RC2_ALREADY_SOLVED | **PRODUCTION_SUBSUMED** | — |
| `Set.ite_inter_compl_self` | TRUE_DEPTH2_SEQUENCE_WIN | RC2_ALREADY_SOLVED | **PRODUCTION_SUBSUMED** | — |
| `Set.ite_inter_inter` (fresh) | TRUE_DEPTH2_SEQUENCE_WIN | RC2_ALREADY_SOLVED | **PRODUCTION_SUBSUMED** | — |

This matches the RC3 validation (`rc3_validation_comparison.json`: credited delta 0) and the RC3
minimal relabel (all 5 = `RC2_ALREADY_SOLVED`). SX4 reproduces the correct verdict **mechanically from
artifacts**, which is the regression guarantee for future candidates.

---

## 5. Future-candidate checklist

`project/evolve/reports/sx4/sequence_candidate_checklist.md` — 10 gating items (baseline-not-solved →
candidate-solves → depth-1-controls-fail → no-equivalent-prod-continuation → fresh-over-production →
generic → off-gate → floors → deterministic → fresh-holdout), with the explicit warning:

> **Never credit a depth-k sequence based only on depth-(k-1) controls. Always compare against a
> literal production run with the same `max_steps`/`top_k` and inspect its trace for an equivalent
> `A`-advanced → `B`-close continuation.**

---

## 6. Optional trace detector

`scripts/sx4_trace_sequence_detector.py` → `project/evolve/experiments/sx4/out/trace_sequence_detection.json`.

Detects, from production traces, an already-present `A`-advanced → `B`-close continuation for known
patterns (`simp [Set.ite]`→`aesop`, `simp [Set.ite]`→`simp_all`, `ext`→`aesop`,
`constructor/intro`→`aesop`), with confidence `exact|likely|weak|insufficient`.

On literal RC2: histogram `{exact: 5, weak: 13, insufficient: 12}`; the **5 `exact`** are exactly the
SX3-credited theorems — independent corroboration of `PRODUCTION_SUBSUMED`.

**Limitations:** heuristic, pattern-limited, trace-format dependent. `weak`/`likely` mean "signal
present but not a verified close." **Do not make release decisions solely on heuristic detection** —
the authoritative credit decision is `sx4_sequence_attribution.py` (which additionally requires
`baseline_finished == false` and depth-1 control failure).

---

## 7. Recommendation

- **Keep RC2 as production** (`rc2_release/rc2_production_wrapper.json`). No RC4; no sequence family
  promoted.
- **Keep SX3 / `SX3_SET_ITE_AESOP` off-by-default / training-only.** It is subsumed by production
  search; reclassify its status `SUBSUMED_BY_PRODUCTION_SEARCH`.
- **Use the SX4 harness before any future RC candidate.** Required gate: run the literal-production
  baseline + candidate, then `sx4_sequence_attribution.py`; credit only `TRUE_SEQUENCE_DELTA`. A
  candidate that fails checklist #1/#4/#5 has no literal delta and must not advance.

---

## 8. Protected-file confirmation

`git diff --stat HEAD` for `rc1_production_wrapper.json`, `rc2_release/rc2_production_wrapper.json`,
`ns24_router.json` → **empty (untouched)**. All new files are under
`project/evolve/experiments/sx4/`, `project/evolve/reports/sx4/`, `scripts/sx4_*`. README not modified.
No RC4 created. No sequence family promoted. **No commit made** (see
`project/evolve/experiments/sx4/out/protected_files_check.txt`).
