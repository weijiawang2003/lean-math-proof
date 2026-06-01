# RC5V3 Methodology

## Goal

Measure whether the RC5V2 hardened-hybrid fresh delta (+8 over RC4 on 240 theorems) is a **stable,
scalable** phenomenon, and characterise the **cost/yield economics** of the safe dynamic
guided-search mode at larger scale. This is a scaling/economics benchmark, not a release.

## Pipeline (17 parts)

1. **Workspace** — `rc5_v3/{cases,out,data}` + this methodology.
2. **Large fresh frontier** (`rc5v3_build_large_fresh_frontier.py`) — TR6 ∪ RC5V2 frontier pools ∪
   discovered catalog, minus an exclusion registry that removes **every prior-used theorem**:
   RC4D/RC4R wins, TR6 wins + batch, RC5H/RC5S/RC5V2 true wins + batches, the RC5V2 eval batch and
   dynamic-examples corpus, and TR7's comparison corpus. Target ≥800 strict-fresh candidates.
3. **Large eval batch** (`rc5v3_select_eval_batch.py`) — stratified (Set 100 / Finset 140 / List 120
   / Multiset 120 / Nat 80 / Order-Other 40), target 600, minimum useful 400, with focused dynamic-
   tail slices. Deterministic ordering (no RNG).
4. **RC2 baseline** (`rc5v3_run_rc2_baseline.py`) — literal RC2 wrapper, live, via rc4r_bench_common
   (top-k 8, max-steps 8, hybrid_evolved). Checkpoint + resume.
5. **RC4 static stage** (`rc5v3_run_static_stage.py`) — RC4R wrapper. Additive: non-gate-firing
   theorems forced RC4≡RC2 (reuse RC2), only gate-firing run live. Reports new-over-RC2,
   regressions (≡0), gate emissions.
6. **Dynamic eligibility** (`rc5v3_build_dynamic_eligibility.py`) — eligible = RC4 failed ∧ non-flake
   ∧ allowed namespace ∧ not disabled Order/root. Classify exclusions.
7. **Retrieval** (`rc5v3_retrieve_lemmas.py`) — TR6 retrieval (TR3∪SF5 index), top-20 lemmas/theorem.
8. **Safe ranked plans B1/B3/B5** (`rc5v3_generate_safe_dynamic_plans.py`) — TR6 generator (TR4 HGB
   ranker) → STRICT RC5S grammar filter (off-policy rejected before scoring) → top-5; mark B1
   (rank 1), B3 (ranks 1–3), B5 (ranks 1–5) slices. Final off-policy count must be 0.
9. **Incremental dynamic run** (`rc5v3_run_safe_dynamic_incremental.py`) — RC5S timeout-safe runner:
   - B1: rank 1 only;
   - B3: ranks 2–3 for cases unsolved after B1;
   - B5: ranks 4–5 for cases unsolved after B3.
   Stop after first dynamic success per theorem. Per-theorem process-group kill, hard wall cap,
   checkpoint + deterministic resume.
10. **Attribution** (`rc5v3_apply_attribution.py`) — classify each dynamic success against the bare
    controls + RC2/RC4 status + freshness: FRESH_TRUE_RC5V3_DELTA / RC4_DUPLICATE /
    BASELINE_DUPLICATE / RC2_ALREADY_SOLVED / SOURCE_SPECIFIC_DYNAMIC_WIN / OPEN_FLAKE /
    TIMEOUT_BOUNDED / NEEDS_REVIEW / NO_DYNAMIC_WIN. Break out by budget B1/B3/B5.
11. **System comparison + cost curve** (`rc5v3_compare_systems_and_cost.py`) — RC2 vs RC4 vs
    RC5V3-B1/B3/B5; cumulative + marginal wins and probes/win per budget; recommend a budget.
12. **Namespace/feature yield** (`rc5v3_namespace_feature_yield.py`) — per-namespace + per-feature
    eligible/wins/probes/probes-per-win, classify each namespace HIGH/MODERATE/LOW yield /
    DISABLE / NEED_MORE_DATA.
13. **Safety audit** (`rc5v3_safety_audit.py`) — off-policy / timeouts / killed / max wall / flake /
    unknown-name / namespace violations / probes-per-win; classify SAFE_DYNAMIC_SCALING_CONFIRMED /
    PARTIAL / TOO_EXPENSIVE / NO_VALUE / UNSAFE_TIMEOUT_BEHAVIOR.
14. **Maintenance decision** (`rc5v3_maintenance_decision.py`) — combine cost/yield/safety into one
    owner-facing recommendation (MAINTAIN_GUIDED_SEARCH_MODE / MAINTAIN_BUT_NAMESPACE_LIMITED /
    KEEP_RESEARCH_ONLY / DISABLE_DYNAMIC_DEFAULT / NEED_BETTER_RANKER / NEED_BETTER_RETRIEVAL).
15. **Export examples** (`rc5v3_export_examples.py`) — one safe attempt = one row, with budget slice
    / result / attribution / safety flags / freshness. Ranker NOT retrained.
16. **Report** — `project/evolve/reports/rc5/rc5v3_hardened_hybrid_scaling_cost_report.md`.
17. **Verification** — protected-file diff empty, git status, headline cross-check.

## Attribution discipline

A win counts as **FRESH_TRUE_RC5V3_DELTA** only when **all** hold:
RC2 failed ∧ RC4 static failed ∧ a strict safe program solved ∧ every bare control (simp / simp_all
/ aesop / classical;aesop / exact L / simpa using L / simp [L]) failed ∧ the theorem is strict-fresh
(not in any prior win set). Source-specific wins (the winning lemma is the theorem itself, or a
known source-specific family) are excluded.

## Safety invariants (inherited from RC5S, re-verified at scale)

- Every program in the final plan matches the RC5S strict grammar; **final off-policy count = 0**.
- Each theorem runs in a process-group-killable subprocess with a hard wall cap (60s + 5s SIGKILL
  grace); no per-tactic SIGALRM dependence.
- `no_global_stalls` ⇔ no theorem exceeds cap+grace without being killed.
- Deterministic order + per-theorem checkpoint ⇒ reproducible / resumable.

## Reuse / economy

- RC4 static is additive ⇒ only gate-firing theorems run live; the rest reuse RC2.
- Incremental B1→B3→B5 only escalates cases still unsolved, so probe spend is measured marginally.
- All live runs checkpoint per theorem; partial progress is valid and resumable.
