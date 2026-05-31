# SX2 — SET2 Minimal-Sufficient Relabel (Attribution)

- attribution histogram: `{'NEEDS_DEEPER_SEQUENCE': 26, 'TRUE_SET2_WIN': 5, 'BASELINE_DUPLICATE': 1}`
- **TRUE_SET2_WIN = 5** by gate: {'SET_ITE_SIMP': 5}
- off-gate emissions: 0
- A SET2 solve is a TRUE_SET2_WIN only if RC1(proxy) and ALL baselines failed, the gate is non-baseline and theorem-agnostic. Mirrors NS23 minimal-sufficient attribution. No promotion claim without a TRUE_SET2_WIN here.

| set | theorem | rc1 | gate | set2_tactic | solved | attribution | reason |
|---|---|---|---|---|---|---|---|
| selected | `Set.diff_singleton_subset_iff` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| selected | `Set.ite_eq_of_subset_left` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| selected | `Set.pair_eq_pair_iff` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| selected | `Set.subset_insert_iff` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| selected | `Set.subset_singleton_iff_eq` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| selected | `Set.union_empty_iff` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| selected | `Set.antitoneOn_iff_antitone` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| selected | `Set.ssubset_singleton_iff` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| selected | `Set.ite_empty_right` | False | SET_ITE_SIMP | `simp [Set.ite]` | True | **TRUE_SET2_WIN** | RC1(proxy) failed, all baselines failed, non-baseline mined gate SET_I |
| selected | `Set.ite_inter` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| selected | `Set.ite_inter_self` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| selected | `Set.ite_right` | False | SET_ITE_SIMP | `simp [Set.ite]` | True | **TRUE_SET2_WIN** | RC1(proxy) failed, all baselines failed, non-baseline mined gate SET_I |
| holdout | `Set.ite_compl` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| holdout | `Set.ite_empty` | False | SET_ITE_SIMP | `simp [Set.ite]` | True | **TRUE_SET2_WIN** | RC1(proxy) failed, all baselines failed, non-baseline mined gate SET_I |
| holdout | `Set.ite_empty_left` | False | SET_ITE_SIMP | `simp [Set.ite]` | True | **TRUE_SET2_WIN** | RC1(proxy) failed, all baselines failed, non-baseline mined gate SET_I |
| holdout | `Set.ite_inter_of_inter_eq` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| holdout | `Set.ite_left` | False | SET_ITE_SIMP | `simp [Set.ite]` | True | **TRUE_SET2_WIN** | RC1(proxy) failed, all baselines failed, non-baseline mined gate SET_I |
| holdout | `Set.mem_dite` | True | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| holdout | `Set.mem_dite_empty_left` | True | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| holdout | `Set.mem_dite_empty_right` | True | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| holdout | `Set.Nonempty.subset_pair_iff_eq` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| holdout | `Set.diff_union_inter` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| holdout | `Set.eq_of_inclusion_surjective` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| holdout | `Set.inclusion_inclusion` | True | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| holdout | `Set.inclusion_right` | False | SET_EXT_SIMP | `ext x <;> simp` | True | **BASELINE_DUPLICATE** | gate SET_EXT_SIMP is speculative (mined_support=0<2); `ext x <;> simp` |
| holdout | `Set.inclusion_self` | True | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| holdout | `Set.insert_diff_eq_singleton` | True | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| holdout | `Set.insert_diff_of_mem` | True | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| holdout | `Set.insert_diff_of_not_mem` | True | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| holdout | `Set.insert_diff_self_of_not_mem` | True | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| holdout | `Set.monotoneOn_iff_monotone` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |
| holdout | `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` | False | None | `` | False | **NEEDS_DEEPER_SEQUENCE** | SET2 emitted but did not close the goal |