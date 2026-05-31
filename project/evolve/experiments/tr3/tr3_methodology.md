# TR3 — Retrieval-Aware Depth Search at Scale

## Purpose

TR3 tests one hypothesis at a larger scale: **does combining existing-lemma
retrieval with bounded depth-2/3 proof programs, judged by SX4 attribution, produce
genuine wins over literal RC2?**

It is motivated by the convergent findings of the preceding experiments:

- **SF4** mined 27 confirmed literal-RC2 failures and found **0** cheap
  tactic/sequence TRUE_DELTA → `MISSING_LEMMA_TRIAGE_READY`.
- **TR2** exhausted the fresh frontier (47/47 already TR1-labelled, 0 net-new) →
  `INCONCLUSIVE_TOO_SMALL`.
- **SF5** retrieved existing lemmas for the 20 missing-bridge targets and concluded
  **0 TRUE_MISSING_BRIDGE_LEMMA** — every target is an existing Mathlib theorem; the
  blocker is *retrieval-aware multi-step proof depth*, not lemma synthesis. SF5 still
  found 5 single-shot retrieval wins (4 definitional `simp [Pred,PredOn]` unfolds + 1
  hinted aesop).

TR3 is the natural follow-up: take the SF5 signal (retrieval helps) and the SF5
blocker (most targets need multi-step proofs) and ask whether **bounded-depth proof
programs seeded with retrieved lemmas** close the depth gap.

## What TR3 is NOT

- Not production routing (RC2 and ns24 are read-only; nothing is promoted).
- Not theorem/lemma synthesis (no new lemmas invented).
- Not end-to-end neural proof generation (no large model trained; the optional Part 11
  retrain reuses only the small TR1-class models).
- Not RC4.

## Definitions (win taxonomy)

A program is a single-line tactic of bounded depth (≤3 `<;>`-composed stages),
optionally seeded with retrieved lemmas `L`.

- **RETRIEVAL_ONLY_WIN** — literal RC2 fails; a depth-1 retrieval probe
  (`exact/simpa/rw/simp [L]`) closes directly.
- **RETRIEVAL_DEPTH_WIN** — literal RC2 fails; a depth-2/3 program *using* a retrieved
  lemma closes (`simp [L] <;> aesop`, `rw [L] <;> simp_all`, `ext x <;> simp [L]`, …).
- **DEPTH_ONLY_WIN** — literal RC2 fails; a bounded depth program with **no** retrieved
  lemma closes (`ext x <;> aesop`, `constructor <;> intro h <;> aesop`, …).
- **PRODUCTION_SUBSUMED** — literal RC2 already solves (stale failure / now-solved); no
  credit. The SX3 over-credit guard.
- **TRUE_DELTA** — a win confirmed over **literal RC2** after SX4 attribution. Only
  TRUE_*_DELTA classes carry credit.
- **PROOF_DEPTH_GAP** — retrieval finds plausible lemmas but no generated program of
  depth ≤3 closes the goal.

## Method

1. **Case pool** (Part 2): SF5 depth-gap targets (A), SF5 retrieval/routing targets
   (B), confirmed SF4 RC2 failures not in SF5 (C), fresh SF1 frontier (D), and a
   multi-namespace expansion sampled from `discovered_theorems.json` covering
   Set/Finset/Multiset/Nat/List/Order (E). Deduped by `full_name`; cases without a
   `file_path` are recorded separately in `tr3_case_pool_unresolved.jsonl`.
2. **RC2 confirmation at scale** (Part 3): SF4 and TR2 ran literal RC2 with the
   *identical* config TR3 uses (rc2_release wrapper, ns24 router, hybrid_evolved,
   top-k 8, max-steps 8, repaired finished-key semantics), so those verified results
   are reused; only cases with no identical-config record are run live via
   `eval_rollout_all`. Only `CONFIRMED_RC2_FAILURE` cases are eligible for TRUE_DELTA.
3. **Retrieval index** (Part 4): reuse + expand the SF5 lemma index (local traced
   Mathlib source incl. `def`s + the project catalog). Coverage reported.
4. **Retrieval** (Part 5): top-20 lemmas per confirmed failure (lexical TF-IDF +
   namespace/path proximity + feature overlap + name-pattern + cluster-shared +
   SF5 winning lemmas + goal-driven defs).
5. **Depth program generation** (Part 6): gated depth-1/2/3 programs, ≤10 lemmas and
   ≤60 programs/target, deterministic ordering, no source-specific scripts. Gates by
   goal shape (Set-eq → `ext`; Set-iff → `constructor/intro`; subset → antisymm;
   Nat/arith → `omega/nlinarith`; Multiset.toFinset → toFinset simp).
6. **Live search** (Part 7): one LeanDojo Dojo per theorem (serialized worker under OS
   hard timeout), gated programs only, per-theorem incremental checkpoint + resume,
   `--stop-after-win` records skipped programs.
7. **SX4 attribution** (Part 8): every claimed win must beat literal RC2; controls
   (`simp/simp_all/aesop/classical <;> aesop`) guard BASELINE_DUPLICATE; a win whose
   depth-1 sub-tactics or the literal RC2 trace already reach the closing state is
   PRODUCTION_SUBSUMED.
8. **Analysis + export** (Parts 9-10): per-family / per-lemma rollups with a
   *do-not-promote-yet* threshold (≥2 TRUE_*_DELTA, 0 off-gate, deterministic,
   generic, SX4-survived); additive training export (never overwrites TR1/TR2/SF5).
9. **Optional retrain** (Part 11): TR1-class models on TR1 / TR1+SF5 / TR1+SF5+TR3,
   exploratory only.

## Guardrails

- Protected configs (RC1/RC2 wrappers, ns24 router, NS9 genome/checkpoints,
  REL1/RC1/RC2 reports, TR1/TR2 datasets) are read-only.
- No production routing change, no RC4, no README update, no commit unless instructed.
- Determinism: programs are emitted in a fixed order; live outcomes are recorded with
  provenance; the win taxonomy credits only literal-RC2-beating, SX4-survived deltas.
