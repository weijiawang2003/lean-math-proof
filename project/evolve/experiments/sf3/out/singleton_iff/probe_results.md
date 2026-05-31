# SF3 Deep Probe Ladder — `Multiset.toFinset_eq_singleton_iff`

- probes: 13 | solved: 0 | histogram: `{'proof_failed': 6, 'max_recursion': 6, 'parse_error': 1}`
- design: single Dojo session; per-probe SIGALRM + checkpoint
- **No probe closed the goal.**

## Per-probe outcomes

| family | outcome | solved | tangle | probe | error |
|---|---|---|---|---|---|
| F1_baselines | proof_failed | False | True | `simp` |  |
| F1_baselines | proof_failed | False | True | `simp_all` |  |
| F2_iff_decomposition | max_recursion | False | True | `constructor <;> intro h <;> simp_all` | tactic 'simp' failed, nested error: maximum recursion depth has been r |
| F2_iff_decomposition | max_recursion | False | True | `refine ⟨fun H => ?_, fun H => ?_⟩ <;> simp_all` | tactic 'simp' failed, nested error: maximum recursion depth has been r |
| F3_finset_singleton_rewrite | proof_failed | False | True | `simp only [Finset.eq_singleton_iff_unique_mem, Multiset` |  |
| F3_finset_singleton_rewrite | max_recursion | False | True | `constructor <;> intro h <;> simp_all [Finset.eq_singlet` | tactic 'simp' failed, nested error: maximum recursion depth has been r |
| F4_membership_bridge | proof_failed | False | True | `simp only [Multiset.mem_toFinset, Multiset.mem_singleto` | simp made no progress |
| F4_membership_bridge | max_recursion | False | True | `constructor <;> intro h <;> simp_all [Multiset.mem_toFi` | tactic 'simp' failed, nested error: maximum recursion depth has been r |
| F5_official_dep_bridge_source_inspired | proof_failed | False | True | `simp [Multiset.toFinset_nsmul, Multiset.toFinset_single` |  |
| F5_official_dep_bridge_source_inspired | max_recursion | False | True | `constructor <;> intro h <;> simp_all [Multiset.toFinset` | tactic 'simp' failed, nested error: maximum recursion depth has been r |
| F5_official_dep_bridge_source_inspired | max_recursion | False | True | `constructor <;> intro h <;> simp_all [Multiset.toFinset` | tactic 'simp' failed, nested error: maximum recursion depth has been r |
| F6_known_bad_induction_diagnostic | proof_failed | False | True | `induction s using Multiset.induction_on <;> simp_all` |  |
| F7_source_copy_multiline | parse_error | False | True | `refine ⟨fun H => ⟨fun h => ?_, ?_⟩, fun H => ?_⟩ /   · rw` | <stdin>:2:26: expected end of input |

> No solve is a confirmed win; NS23 minimal-sufficient relabel + deterministic reproduction required before promotion. RC1/production configs not modified.