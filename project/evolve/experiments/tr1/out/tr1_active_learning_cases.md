# TR1 active-learning case selection

- frontier considered: 37; selected top **25**
- criteria: high entropy + underrepresented predicted family + Set iff/subset shape + not already in dataset; excludes RC2-solved

| rank | theorem | predicted | entropy | score |
|---|---|---|---|---|
| 1 | `Set.subset_pair_iff_eq` | SET_ITE_SIMP | 1.093 | 1.593 |
| 2 | `Set.ssubset_iff_sdiff_singleton` | BASELINE_DUPLICATE | 1.0851 | 1.5851 |
| 3 | `Function.Injective.nonempty_apply_iff` | NO_CHEAP_ACTION | 0.9707 | 1.4707 |
| 4 | `Set.strictMonoOn_iff_strictMono` | SET_ITE_SIMP | 0.8514 | 1.3514 |
| 5 | `Set.monotoneOn_iff_monotone` | BASELINE_DUPLICATE | 0.7176 | 1.2176 |
| 6 | `Set.pair_diff_left` | SET_ITE_SIMP | 0.694 | 1.194 |
| 7 | `Set.antitoneOn_iff_antitone` | SET_ITE_SIMP | 0.6936 | 1.1936 |
| 8 | `Set.subset_singleton_iff_eq` | SET_ITE_SIMP | 0.677 | 1.177 |
| 9 | `Set.strictAntiOn_iff_strictAnti` | SET_ITE_SIMP | 0.6232 | 1.1232 |
| 10 | `Set.pair_eq_pair_iff` | SET_ITE_SIMP | 0.525 | 1.025 |
| 11 | `Prop.compl_singleton` | PROOF_SEARCH_DEPTH_GAP | 0.0184 | 1.0184 |
| 12 | `Set.ssubset_iff_insert` | BASELINE_DUPLICATE | 0.4125 | 0.9125 |
| 13 | `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le` | BASELINE_DUPLICATE | 0.4065 | 0.9065 |
| 14 | `Set.diff_singleton_subset_iff` | BASELINE_DUPLICATE | 0.307 | 0.807 |
| 15 | `Set.Nonempty.subset_pair_iff_eq` | SET_ITE_SIMP | 0.257 | 0.757 |
| 16 | `Set.ssubset_singleton_iff` | SET_ITE_SIMP | 0.2549 | 0.7549 |
| 17 | `Set.subset_insert_iff` | BASELINE_DUPLICATE | 0.2478 | 0.7478 |
| 18 | `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` | BASELINE_DUPLICATE | 0.2041 | 0.7041 |
| 19 | `Set.ite_inter` | SX3_PRODUCTION_SUBSUMED | 0.6622 | 0.6622 |
| 20 | `Set.ite_inter_of_inter_eq` | SX3_PRODUCTION_SUBSUMED | 0.6557 | 0.6557 |
| 21 | `Eq.subset` | NO_CHEAP_ACTION | 0.0405 | 0.5405 |
| 22 | `Set.union_empty_iff` | SET_ITE_SIMP | 0.0053 | 0.5053 |
| 23 | `Set.diff_union_inter` | BASELINE_DUPLICATE | 0.0038 | 0.5038 |
| 24 | `Set.insert_diff_eq_singleton` | BASELINE_DUPLICATE | 0.0015 | 0.5015 |
| 25 | `Set.ite_eq_of_subset_right` | SET_ITE_SIMP | 0.0006 | 0.5006 |