# RC2 — SET_ITE_SIMP Minimal-Sufficient Relabel (vs literal RC1)

- attribution histogram: `{'TRUE_SET_ITE_SIMP_WIN': 5, 'NEEDS_DEEPER_SEQUENCE': 16, 'RC1_ALREADY_SOLVED': 11}`
- **TRUE_SET_ITE_SIMP_WIN (unique theorems) = 5**: ['Set.ite_empty', 'Set.ite_empty_left', 'Set.ite_empty_right', 'Set.ite_left', 'Set.ite_right']
- A candidate solve is TRUE_SET_ITE_SIMP_WIN only if literal RC1 AND all four baselines failed and non-baseline `simp [Set.ite]` closed it. RC1_ALREADY_SOLVED / BASELINE_DUPLICATE are not wins.

| theorem | sets | rc1 | gate | set_ite | baseline_solved_by | attribution |
|---|---|---|---|---|---|---|
| `Set.ite_empty_right` | set_ite_known_wins,set_ite_selected_failures | False | True | True | None | **TRUE_SET_ITE_SIMP_WIN** |
| `Set.ite_right` | set_ite_known_wins,set_ite_selected_failures | False | True | True | None | **TRUE_SET_ITE_SIMP_WIN** |
| `Set.ite_empty` | set_ite_known_wins,set_ite_fresh_holdout | False | True | True | None | **TRUE_SET_ITE_SIMP_WIN** |
| `Set.ite_empty_left` | set_ite_known_wins,set_ite_fresh_holdout | False | True | True | None | **TRUE_SET_ITE_SIMP_WIN** |
| `Set.ite_left` | set_ite_known_wins,set_ite_fresh_holdout | False | True | True | None | **TRUE_SET_ITE_SIMP_WIN** |
| `Set.diff_singleton_subset_iff` | set_ite_selected_failures | False | False | False | None | **NEEDS_DEEPER_SEQUENCE** |
| `Set.ite_eq_of_subset_left` | set_ite_selected_failures | False | True | False | None | **NEEDS_DEEPER_SEQUENCE** |
| `Set.pair_eq_pair_iff` | set_ite_selected_failures | False | False | False | None | **NEEDS_DEEPER_SEQUENCE** |
| `Set.subset_insert_iff` | set_ite_selected_failures | False | False | False | None | **NEEDS_DEEPER_SEQUENCE** |
| `Set.subset_singleton_iff_eq` | set_ite_selected_failures | False | False | False | None | **NEEDS_DEEPER_SEQUENCE** |
| `Set.union_empty_iff` | set_ite_selected_failures | False | False | False | None | **NEEDS_DEEPER_SEQUENCE** |
| `Set.antitoneOn_iff_antitone` | set_ite_selected_failures | False | False | False | None | **NEEDS_DEEPER_SEQUENCE** |
| `Set.ssubset_singleton_iff` | set_ite_selected_failures | False | False | False | None | **NEEDS_DEEPER_SEQUENCE** |
| `Set.ite_inter` | set_ite_selected_failures | False | True | False | None | **NEEDS_DEEPER_SEQUENCE** |
| `Set.ite_inter_self` | set_ite_selected_failures | False | True | False | None | **NEEDS_DEEPER_SEQUENCE** |
| `Set.ite_compl` | set_ite_fresh_holdout | False | True | False | None | **NEEDS_DEEPER_SEQUENCE** |
| `Set.ite_inter_of_inter_eq` | set_ite_fresh_holdout | False | True | False | None | **NEEDS_DEEPER_SEQUENCE** |
| `Set.mem_dite` | set_ite_fresh_holdout | True | False | False | None | **RC1_ALREADY_SOLVED** |
| `Set.mem_dite_empty_left` | set_ite_fresh_holdout | True | False | False | None | **RC1_ALREADY_SOLVED** |
| `Set.mem_dite_empty_right` | set_ite_fresh_holdout | True | False | False | None | **RC1_ALREADY_SOLVED** |
| `Set.Nonempty.subset_pair_iff_eq` | set_ite_fresh_holdout | False | False | False | None | **NEEDS_DEEPER_SEQUENCE** |
| `Set.diff_union_inter` | set_ite_fresh_holdout | True | False | False | None | **RC1_ALREADY_SOLVED** |
| `Set.eq_of_inclusion_surjective` | set_ite_fresh_holdout | False | False | False | None | **NEEDS_DEEPER_SEQUENCE** |
| `Set.inclusion_inclusion` | set_ite_fresh_holdout | True | False | False | None | **RC1_ALREADY_SOLVED** |
| `Set.inclusion_right` | set_ite_fresh_holdout | True | False | False | None | **RC1_ALREADY_SOLVED** |
| `Set.inclusion_self` | set_ite_fresh_holdout | True | False | False | None | **RC1_ALREADY_SOLVED** |
| `Set.insert_diff_eq_singleton` | set_ite_fresh_holdout | True | False | False | None | **RC1_ALREADY_SOLVED** |
| `Set.insert_diff_of_mem` | set_ite_fresh_holdout | True | False | False | None | **RC1_ALREADY_SOLVED** |
| `Set.insert_diff_of_not_mem` | set_ite_fresh_holdout | True | False | False | None | **RC1_ALREADY_SOLVED** |
| `Set.insert_diff_self_of_not_mem` | set_ite_fresh_holdout | True | False | False | None | **RC1_ALREADY_SOLVED** |
| `Set.monotoneOn_iff_monotone` | set_ite_fresh_holdout | False | False | False | None | **NEEDS_DEEPER_SEQUENCE** |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` | set_ite_fresh_holdout | False | False | False | None | **NEEDS_DEEPER_SEQUENCE** |