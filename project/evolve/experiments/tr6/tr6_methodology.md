# TR6 — Ranker-Guided Fresh Multi-Namespace Frontier Sweep

## Purpose

TR5 confirmed the TR4 ranker recovers TR3's known wins ~5× more efficiently
([[project_tr5_ranker_guided_live_search]]), but found **0 fresh wins** — it proved
*efficiency*, not *coverage*. TR6 runs the same ranker-guided live procedure on a **fresh,
larger, multi-namespace** frontier — theorems never used in TR1/TR2/SF4/SF5/TR3/TR4/TR5 —
to answer:

1. Can ranker-guided search find **fresh** TRUE_DELTA wins beyond TR3?
2. Can it find positives **outside Set** (Finset/List/Multiset/Nat/Order)?
3. Does the ranker stay useful on fresh cases?
4. Do RC4B (`Set.disjoint_left`) / RC4C (`d2_simp_aesop`) get **fresh holdout** support?
5. Is the next bottleneck candidate validation, retrieval, or larger frontier mining?

## TR6 vs TR5

- **TR5** mostly re-ran TR3's known 92-failure frontier to verify ranker efficiency
  (0 fresh wins by construction — the programs were a subset of TR3's battery).
- **TR6** expands to a **fresh** frontier source-scanned from the traced Mathlib tree
  (cx1-style regex extractor over a broad multi-namespace file list + discovered_theorems),
  minus a strict exclusion registry. The goal is **new coverage**.

## Fresh frontier source (resolvability verified)

The reliable known-good source `project/discovered_theorems.json` spans only 3 files
(Nat/Defs, Set/Basic, Finset/Basic) — already mined by TR3. TR6 broadens by source-scanning
a curated multi-namespace file list from the traced cache
(`~/.cache/lean_dojo/.../mathlib4/`). A smoke test confirmed theorems extracted from
List/Basic, Order/Basic, Multiset/Basic and Finset/Card **open in LeanDojo**, so the broad
scan yields Dojo-resolvable theorems. The live RC2 confirmation step (Part 5) doubles as the
final availability filter: any theorem that does not open becomes PATH_ERROR /
TRACE_INSUFFICIENT and is excluded from the search.

## Success metrics

fresh confirmed RC2 failures · fresh TRUE_DELTA wins · useful new labels · positives outside
Set · ranker efficiency on fresh cases (success/probe, first-success rank) · RC4B/RC4C fresh
evidence · training examples exported.

## Definitions

- **FRESH_CASE** — theorem not in TR1/TR2/SF4/SF5/TR3/TR4/TR5 train/eval sets (per the
  exclusion registry).
- **FRESH_RC2_FAILURE** — a fresh case literal RC2 fails.
- **FRESH_TRUE_DELTA** — a fresh RC2 failure solved by a ranker-guided program after
  attribution (controls fail, not source-specific, over literal RC2).
- **FRESH_NONSET_POSITIVE** — a fresh true delta in a namespace ≠ Set.
- **RANKER_FRESH_USEFUL** — the ranker finds true wins / high-quality labels with materially
  fewer probes than a full TR3-style battery.

## Attribution discipline (SX4 / TR3 / TR5)

Every win must beat **literal RC2**: RC2 confirmed-failure + bare controls
(`simp`/`simp_all`/`aesop`/`classical <;> aesop`) fail + not source-specific. Anything RC2
already solves is PRODUCTION_SUBSUMED. Known TR3/TR5 wins are excluded by the registry, so
any credited win here is fresh by construction.

## Configs (frozen)

- strategy `rc2_release/rc2_production_wrapper.json`, route `ns24_router.json`,
  policy `hybrid_evolved`, top-k 8, max-steps 8.
- ranker: TR4 `hgb_program_ranker.joblib` + `tr4_vectorizers.joblib` (full-data; scoring
  identical to `tr4_featurize_programs.py`, via `scripts/tr5_score.py`).

## Live-Lean notes

Serialized one-Dojo-per-theorem workers under `scripts/run_with_timeout.py` (macOS has no
`timeout`); each tactic SIGALRM-bounded; per-theorem checkpoint + resume. Dojo opens ~6 s;
hard theorems up to ~80 s.

## What TR6 must NOT do

Not promote the ranker or any candidate; not change production routing; not create an
RC4/RC4B/RC4C release; not modify any protected file; no commit.
