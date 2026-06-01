# TR5 — Ranker-Guided Live Search and RC4B/RC4C Evidence Collection

## Purpose

TR4 showed **offline** that the HGB program ranker recovers 100 % of TR3 successes at a
top-5-per-theorem budget (88.8 % probe reduction), but leakage-free only *within seen
namespaces*. TR5 asks the live question: **does that probe-reduction result survive real
LeanDojo execution?** And it collects evidence for the next two RC4 candidate families:

- **RC4B** — the `Set.disjoint_left` bridge (3 TR3 wins go through it).
- **RC4C** — `d2_simp_aesop` (retrieval-depth `simp [L] <;> aesop`, 3 TR3 wins).

TR5 is a **live-search** task. It produces data and validation evidence only. It does
**not** promote the ranker, does **not** alter production routing, and does **not** create
an RC4 release.

## Unit of evaluation

Theorem-level live search: for each target, run the TR4-ranker's top-B programs (B = 5
first, then 10/20 on the still-open theorems), stop after the first success, attribute
every win against **literal RC2**.

## Main success metrics

- true wins found live (`RANKER_GUIDED_WIN`)
- credited wins over literal RC2 (`TRUE_RANKER_DELTA`)
- successes per probe (live yield)
- probe reduction vs the TR3 full battery (4,377 programs over 92 theorems)
- whether the top-5 / top-10 ranker budget recovers the known TR3 successes live
- whether RC4B (`Set.disjoint_left`) and RC4C (`d2_simp_aesop`) have enough live
  evidence to warrant a separate literal-RC2⊕candidate validation

## Definitions

- **RANKER_GUIDED_WIN** — a ranker-selected program solves a theorem live.
- **TRUE_RANKER_DELTA** — literal RC2 failed, the ranker-selected program solved it, the
  bare controls did not, the win is not source-specific, and it is over literal RC2.
- **RC4B_EVIDENCE** — a `TRUE_RANKER_DELTA` whose winning program uses the
  `Set.disjoint_left` bridge.
- **RC4C_EVIDENCE** — a `TRUE_RANKER_DELTA` whose winning program is `d2_simp_aesop`
  (`simp [L] <;> aesop`).
- **RANKER_FALSE_POSITIVE** — a high-ranked program fails live.
- **RANKER_MISSED_WIN** — a known TR3/RC4A success not recovered under the chosen budget.

## Attribution discipline (inherited from SX4 / TR3 / RC4A)

Every win must beat **literal production (RC2)**. A win is only credited if:

1. literal RC2 is a `CONFIRMED_RC2_FAILURE` on that theorem,
2. the bare controls (`simp` / `simp_all` / `aesop` / `classical <;> aesop`) do **not**
   solve it (else `BASELINE_DUPLICATE`),
3. the program is not source-specific (the lemma exists in Mathlib / the def is real).

Anything literal RC2 already solves is `PRODUCTION_SUBSUMED` and credited 0.

## Configs (frozen — never modified)

- strategy: `project/evolve/experiments/rc2_release/rc2_production_wrapper.json`
- route: `project/evolve/routing/ns24_router.json`
- policy: `hybrid_evolved`, top-k 8, max-steps 8
- ranker: TR4 `hgb_program_ranker.joblib` + `tr4_vectorizers.joblib`
  (full-data model; scoring identical to `tr4_featurize_programs.py`)

## Live-Lean notes (carried from prior tasks)

LeanDojo opens a Dojo in ~6 s, ~2 s/theorem typical; hard Set theorems ~80 s. macOS has
no `timeout` — every worker runs under `scripts/run_with_timeout.py` with a hard wall
clock, and each tactic is SIGALRM-bounded. Workers are serialized (one Dojo per theorem)
and the driver checkpoints after every theorem so the run resumes.

## What TR5 must NOT do

- not promote the ranker,
- not alter production routing,
- not create an RC4 / RC4B / RC4C release or validation artifacts (unless explicitly
  instructed),
- not modify any protected file (RC1/RC2 wrappers, ns24 router, NS9, REL/RC reports,
  TR1/TR2/SF/TR source datasets).
