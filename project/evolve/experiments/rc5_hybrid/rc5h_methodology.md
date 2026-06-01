# RC5H Methodology

## Pipeline

1. **Policy** (`rc5h_policy.json`) — static stage = RC4R wrapper; dynamic stage = TR4 HGB ranker
   over a small program grammar, gated to {Set, Finset, List, Multiset, Nat}, enabled only after
   static failure, with B5/B10/B20 budgets and a retrieval-confidence gate.
2. **Benchmark sets** — TR6 dynamic-tail replay (the 8 RC4-missed TR6 wins), TR6 static-covered
   controls, RC4R fresh no-delta cases, a fresh dynamic-candidate frontier, multi-namespace hard
   negatives, canonical floors, off-gate controls.
3. **Static stage** — RC4R wrapper over the benchmark (reuse RC4R results where the theorem/config
   match; live otherwise).
4. **RC2 baseline** — for `TRUE_HYBRID_DELTA` classification.
5. **Retrieve** — top-20 lemmas for the static failures that are dynamic-eligible.
6. **Generate + score** — candidate programs from the grammar, featurized + scored by the TR4 HGB
   ranker, top-20 kept, B5/B10/B20 tagged. No live Lean.
7. **Dynamic stage live** — B5 (top-5 + controls), then B10 (ranks 6–10 on B5-unsolved), then B20
   (ranks 11–20 on B10-unsolved). Stop at first success; deterministic ordering; per-theorem
   checkpoint.
8. **Hybrid attribution** — controls (simp/simp_all/aesop/classical;aesop/exact L/simpa/simp[L])
   per dynamic win → TRUE_HYBRID_DELTA / STATIC_DUPLICATE / BASELINE_DUPLICATE / RC2_ALREADY_SOLVED
   / DYNAMIC_ONLY_BUT_SOURCE_SPECIFIC / UNKNOWN_NAME_FAILURE / OPEN_FLAKE / NO_DYNAMIC_WIN.
9. **System comparison** — RC2 vs RC4 static vs RC5H B5/B10/B20: solved, new wins over RC2/RC4,
   regressions, probes, dynamic probes per additional win, floors, by namespace/set.
10. **Safety audit** — dynamic gate firings, unknown-name rate, flake/timeout rate, off-policy /
    broad programs, namespace violations, emitted-and-failed, source-specific risk, cost.
11. **Optional retrain** — TR4 vs TR4+TR6 vs TR4+TR6+RC5H ranker, grouped CV.

## Key invariants

- Dynamic stage runs ONLY on static failures (RC4 ⊇ RC2 ⇒ never on RC2/RC4 wins).
- The static core is byte-identical to RC4R; floors and known wins are preserved by construction.
- A dynamic win counts only if RC2 AND RC4 both failed and bare controls fail (additive over RC4).
- RC4A gate tightening is recorded as a recommendation only; NOT implemented in the static core.
- Deterministic program ordering (ranker score, then lexical); flakes excluded per RC4* methodology.

## Scope

Prototype: the fresh frontier is sized for a tractable live dynamic stage. The decisive test is the
TR6 dynamic-tail replay (RC4 fails, TR6 program works) — does the ranker-guided stage re-find it?
