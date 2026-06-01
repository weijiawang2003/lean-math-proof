# RC5V3 — Hardened Hybrid Scaling and Cost Benchmark

**Type:** scaling / economics benchmark (NOT a release, NOT a promotion, NO wrapper change, NO commit).
**Date:** 2026-06-01.

## What RC5V3 is

RC5V2 confirmed that the hardened hybrid — **RC4R static core + RC5S strict safe dynamic B5** —
produces **fresh, attributable deltas over RC4** on a 240-theorem fresh frontier (+8), safely.
RC5V3 asks the next question:

> Is RC5V2's +8 fresh delta a **stable phenomenon** that scales, or a lucky draw from one
> 240-theorem frontier?

RC5V3 does **not** change the policy. It re-runs the exact RC5V2 pipeline on a **larger, disjoint,
fresh out-of-sample frontier** and measures the real **cost/yield curve** of the safe dynamic
guided-search mode:

- fresh true deltas over RC4 (does the +8 rate hold at scale?),
- probes per win (economics),
- namespace-level yield (where is dynamic worth running?),
- a **B1 / B3 / B5 cost curve** (what budget is worth paying?),
- whether dynamic eligibility can be narrowed,
- whether an off-by-default guided-search mode is worth maintaining.

## What it reuses (unchanged)

- **STATIC_CORE** = RC4R static wrapper
  (`project/evolve/experiments/rc4_release_candidate/rc4_release_candidate_wrapper.json`).
- **SAFE_DYNAMIC_STAGE** = RC5S strict B5 dynamic retrieval stage
  (`project/evolve/experiments/rc5_safety/rc5s_strict_policy.json` + `scripts/rc5s_grammar.py`).
- **RC5S timeout-safe runner** (`scripts/rc5s_timeout_safe_runner.py`) — the process-group-kill
  guarantee that keeps the dynamic stage bounded.
- **TR4 ranker** (`project/evolve/experiments/tr4/`), **TR6/TR5 retrieval + generation**
  (`scripts/tr6_retrieve_lemmas.py`, `scripts/tr6_generate_ranked_programs.py`).
- **rc4r_bench_common** for the static/baseline benchmark harness; **rc4d_gate** for gate firing.

RC5V3 **RC5V3 = RC4R static core + safe dynamic B5**, exactly RC5V2's system, only larger and
incremental-budgeted (B1 → B3 → B5).

## Definitions

- **STATIC_CORE** — RC4R static wrapper.
- **SAFE_DYNAMIC_STAGE** — RC5S strict B5 dynamic retrieval stage.
- **RC5V3** — RC4R static core + safe dynamic B5.
- **FRESH_TRUE_RC5V3_DELTA** — RC2 failed ∧ RC4 failed ∧ RC5V3 safe dynamic solved ∧ bare controls
  did not solve ∧ theorem is strict-fresh.
- **COST_PER_WIN** — dynamic programs attempted / fresh true dynamic wins.
- **ECONOMICALLY_USEFUL** — the dynamic stage produces fresh wins with acceptable cost and no
  safety failures.

## Layout

- `cases/` — large fresh frontier pool, eval batch, dynamic-eligible set.
- `out/` — per-stage JSON/MD (RC2 baseline, RC4 static, eligibility, retrieval, safe plan, B1/B3/B5
  dynamic results, attribution, system comparison + cost curve, namespace/feature yield, safety
  audit, maintenance decision).
- `data/` — exported safe-attempt examples + summary (ranker NOT retrained).

## Guardrails

RC2 stays production. RC4 stays the best always-on static candidate. RC5V3 is a **measurement of
the off-by-default guided-search mode** — nothing is promoted, no wrapper is changed, no release is
created, and no commit is made. Protected files (RC1/RC2/NS24/RC4R/RC5S/RC5H/RC5V2 + NS9 + RC4*/TR1–7)
are never modified.
