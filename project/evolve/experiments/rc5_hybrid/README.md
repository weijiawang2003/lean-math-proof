# RC5H — Hybrid Static + Ranker-Guided Retrieval Prototype

RC5H is a **non-production prototype** that tests TR7's recommendation
(`RC5_HYBRID_STATIC_PLUS_RANKER`): does a gated dynamic retrieval/ranker stage, run **only when
the RC4 static core fails**, recover the theorem-specific tail that the static wrapper cannot cover?

    RC5H = RC4 static safe core  ⊕  gated dynamic ranker-guided retrieval stage

## Two stages

- **Stage 1 (static):** the frozen **RC4R** wrapper (RC2 ⊕ RC4A ⊕ RC4B ⊕ RC4C_residue). Reused as-is.
- **Stage 2 (dynamic):** runs **only if** RC4 fails AND the dynamic gate allows. Retrieves top-20
  lemmas (TR3 ∪ SF5 index), generates candidate programs from a small grammar, scores them with
  the **TR4 HGB ranker**, and probes the top-B (B5 → B10 → B20) live, stopping at first success.

RC5H is **not** production, **not** a release, **not** a promotion. It only measures whether the
dynamic stage adds a genuine `TRUE_HYBRID_DELTA` over RC4 at acceptable cost/safety.

## Definitions

- **STATIC_WIN** — RC4 solves the theorem (dynamic stage never runs).
- **DYNAMIC_WIN** — RC4 fails, the dynamic ranker-guided stage solves it.
- **TRUE_HYBRID_DELTA** — RC2 fails ∧ RC4 static fails ∧ a dynamic program solves ∧ bare controls
  do not solve ∧ not source-specific.
- **STATIC_DUPLICATE** — dynamic stage solves but RC4 already solved (should be gated out).
- **DYNAMIC_FALSE_POSITIVE** — a high-ranked dynamic program fails.
- **DYNAMIC_STAGE_GATED_OUT** — theorem not eligible for dynamic retrieval (namespace/confidence gate).

## Reuse

The dynamic stage reuses the validated TR3/TR5/TR6 pipeline verbatim: SF5/TR3 retrieval scorer,
the TR3 program grammar, the TR4 HGB `RankerScorer` (`tr5_score`), and the `run_budget`
B5/B10/B20 live driver (`tr5_run_ranked_live_search`). The static stage reuses the RC4R wrapper
and the `rc4r_bench_common` runner. Nothing in TR1–TR7 / RC4* is modified.

## Protected

RC1/RC2/NS24, NS9, REL1/RC1/RC2 reports, TR1–TR7 datasets, RC4A/B/C/D + RC4R artifacts —
**untouched**. No README update, no routing change, no RC5 release, no promotion, no commit.
