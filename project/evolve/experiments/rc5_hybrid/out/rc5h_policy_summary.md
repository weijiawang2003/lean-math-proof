# RC5H policy

- family: rc5_hybrid_static_plus_ranker | base static core: RC4R | promotion_allowed: False

## Static stage
- wrapper: `project/evolve/experiments/rc4_release_candidate/rc4_release_candidate_wrapper.json` (frozen RC4R, unchanged)
- config: hybrid_evolved, top-k 8, max-steps 8

## Dynamic stage (only after static failure)
- ranker: TR4 HGB | retrieval top-k: 20
- budgets: {'B5': 5, 'B10': 10, 'B20': 20}
- grammar (9): ['exact L', 'simpa using L', 'simp [L]', 'rw [L]', 'simp [L] <;> aesop', 'simp [L] <;> simp_all', 'rw [L] <;> aesop', 'ext x <;> simp [L]', 'constructor <;> intro h <;> aesop']
- gates: namespaces ['Set', 'Finset', 'List', 'Multiset', 'Nat'], max_unknown_name_rate 0.1, disable_if_no_retrieval_confidence True; Order family disabled

## RC4A gate tightening
- **RECOMMENDATION_ONLY_NOT_IMPLEMENTED** (precision 0.092) — recommendation only, not implemented.

## Evaluation
- TRUE_HYBRID_DELTA: RC2 failed AND RC4 static failed AND a dynamic program solved AND bare controls did not solve AND not source-specific
- additive: dynamic stage runs only on static failures -> regressions over RC4 structurally impossible.
