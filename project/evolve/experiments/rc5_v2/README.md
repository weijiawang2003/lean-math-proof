# RC5V2 — Hardened Hybrid Fresh Benchmark

RC5V2 benchmarks the **hardened** hybrid system on a **fresh out-of-sample frontier**:

    RC5V2 = RC4R static core  +  RC5S strict safe dynamic B5 stage (timeout-safe)

It answers the question RC5H/RC5S left open: **does the now-safe dynamic stage produce fresh,
stable, attributable deltas over RC4 — without the RC5H safety failures (stalls, off-policy,
unbounded budgets)?** This is a benchmark/prototype, not a release.

## Two stages

- **Stage 1 (static):** the frozen RC4R wrapper.
- **Stage 2 (dynamic):** runs ONLY after RC4 static failure — the **RC5S strict safe B5** stage:
  strict low-risk grammar (no `simp_all`, no depth-3, no off-policy), TR4-ranker-ordered top-5,
  run through the **RC5S timeout-safe runner** (per-theorem process-group kill). **No B10/B20
  mainline.**

## Key definitions

- **STATIC_SOLVED** — RC4R solves.
- **STATIC_FAILED_DYNAMIC_ELIGIBLE** — RC4R fails and the RC5S dynamic gate allows (allowed
  namespace, non-flake, retrieval present).
- **SAFE_DYNAMIC_WIN** — RC5S safe B5 solves after RC4 fails.
- **TRUE_RC5V2_DELTA** — RC2 fails ∧ RC4R fails ∧ safe B5 solves ∧ bare controls do not solve.
- **FRESH_TRUE_RC5V2_DELTA** — a TRUE_RC5V2_DELTA on a theorem not in any TR6/RC5H/RC5S win set
  (strict/soft fresh).

## Reuse (nothing modified)

RC4R wrapper + `rc4r_bench_common` (static/RC2 stages); `rc5s_strict_policy.json` +
`rc5s_grammar.py` + `rc5s_timeout_safe_runner.py` (safe dynamic stage); TR6 retrieval + TR4 HGB
ranker (program generation/scoring). The fresh frontier excludes every prior-used theorem
(TR1–7 / RC4* / RC5H / RC5S).

## Protected

RC1/RC2/NS24, RC4R wrapper, RC5H policy, **RC5S strict policy**, NS9, REL1/RC1/RC2 reports,
TR1–7 datasets, RC4* + RC5H + RC5S originals — untouched. No README update, no routing change,
no RC5 release, no promotion, no commit.
