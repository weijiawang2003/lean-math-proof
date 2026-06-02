# FLI1 — Live Residual Goal Capture and Candidate Lemma Synthesis

FLI1 is the first real step from proof automation toward **verifier-guided mathematical
discovery**. It takes the 40 FLI0 seed failures, reruns them live in LeanDojo to capture the
**residual proof states** FLI0 could not, turns those goals into **candidate intermediate
lemmas**, checks whether they already exist, typechecks and tries to prove them, and finally tests
whether a proved candidate **rescues** the original failed theorem.

This is **not** an RC benchmark. The metric is **downstream rescue through invented intermediate
lemmas**, not solved count. A theorem that solves directly during rerun is recorded but is **not**
an FLI1 success.

## Core definitions

- **RESIDUAL_GOAL** — the final Lean goal state after a controlled, non-finishing proof attempt.
- **CANDIDATE_LEMMA** — a proposed intermediate theorem that may help close the residual goal.
- **PROVED_CANDIDATE_LEMMA** — a candidate whose proof Lean accepts.
- **DOWNSTREAM_RESCUE** — the original failed theorem becomes provable when the candidate lemma is
  available (tested faithfully at the theorem's file position, with controls).

## Important honesty constraint

The 40 seeds are **real Mathlib lemmas** that the *restricted* RC5 safe-tactic battery could not
rediscover at the theorem's position in its source file (where the lemma itself and everything
after it are out of scope). They are not open problems. Two consequences:

1. **Proving a candidate** against a full `import Module` can be trivial (the candidate may already
   be in Mathlib). Part 6 (existing-lemma check) records whether the candidate is `RETRIEVAL_GAP`
   (already exists) vs `PROBABLY_NEW`.
2. **Rescue must be tested at the theorem's position in LeanDojo**, never against a fresh full
   import, or the result is vacuous. We inline the candidate as a local `have` (or reference an
   existing earlier lemma) and require the matching control (same tactic without the candidate) to
   fail.

## Pipeline (scripts)

1. `fli1_build_live_rerun_plan.py` — controlled rerun tactics per seed/pattern.
2. `fli1_capture_residual_goals.py` — **live** LeanDojo residual-goal capture.
3. `fli1_normalize_and_cluster_goals.py` — normalize + cluster residual goals.
4. `fli1_synthesize_candidate_lemmas.py` — candidate lemma shapes.
5. `fli1_check_existing_lemmas.py` — exists / close / new / retrieval-gap.
6. `fli1_typecheck_candidate_lemmas.py` — `lake env lean` `:= by sorry`.
7. `fli1_prove_candidate_lemmas.py` — `lake env lean` safe tactics.
8. `fli1_test_downstream_rescue.py` — **live** at-position rescue with controls.
9. `fli1_write_lemma_invention_atlas.py` — researcher-facing atlas.

Outputs: `cases/` (jsonl records), `out/` (summaries + atlas md), `data/` (atlas json),
`live_traces/` (per-seed capture traces). Report:
`project/evolve/reports/fli/fli1_live_residual_goal_and_candidate_lemma_report.md`.

## Safety

No production wrapper / router / report modified; nothing promoted; ranker not retrained; temp
Lean files only (never modifies Mathlib source); per-tactic + per-process timeouts bound all live
work; no commit.
