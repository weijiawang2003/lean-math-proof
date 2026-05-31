# RC2 — RC1 + SET_ITE_SIMP Candidate Eval

- total=32 | literal RC1 solved=11 | candidate solved=16 | **new wins over literal RC1=5** | regressions=0 | off-gate=0
- gate fired=10 | precision: {'emitted_and_solved': 5, 'emitted_and_failed': 5, 'not_emitted': 22}

| theorem | sets | rc1 | gate | set_ite | candidate | new_win | off_gate |
|---|---|---|---|---|---|---|---|
| `Set.ite_empty_right` | set_ite_known_wins,set_ite_selected_failures | False | True | True | True | True | False |
| `Set.ite_right` | set_ite_known_wins,set_ite_selected_failures | False | True | True | True | True | False |
| `Set.ite_empty` | set_ite_known_wins,set_ite_fresh_holdout | False | True | True | True | True | False |
| `Set.ite_empty_left` | set_ite_known_wins,set_ite_fresh_holdout | False | True | True | True | True | False |
| `Set.ite_left` | set_ite_known_wins,set_ite_fresh_holdout | False | True | True | True | True | False |
| `Set.diff_singleton_subset_iff` | set_ite_selected_failures | False | False | False | False | False | False |
| `Set.ite_eq_of_subset_left` | set_ite_selected_failures | False | True | False | False | False | False |
| `Set.pair_eq_pair_iff` | set_ite_selected_failures | False | False | False | False | False | False |
| `Set.subset_insert_iff` | set_ite_selected_failures | False | False | False | False | False | False |
| `Set.subset_singleton_iff_eq` | set_ite_selected_failures | False | False | False | False | False | False |
| `Set.union_empty_iff` | set_ite_selected_failures | False | False | False | False | False | False |
| `Set.antitoneOn_iff_antitone` | set_ite_selected_failures | False | False | False | False | False | False |
| `Set.ssubset_singleton_iff` | set_ite_selected_failures | False | False | False | False | False | False |
| `Set.ite_inter` | set_ite_selected_failures | False | True | False | False | False | False |
| `Set.ite_inter_self` | set_ite_selected_failures | False | True | False | False | False | False |
| `Set.ite_compl` | set_ite_fresh_holdout | False | True | False | False | False | False |
| `Set.ite_inter_of_inter_eq` | set_ite_fresh_holdout | False | True | False | False | False | False |
| `Set.mem_dite` | set_ite_fresh_holdout | True | False | False | True | False | False |
| `Set.mem_dite_empty_left` | set_ite_fresh_holdout | True | False | False | True | False | False |
| `Set.mem_dite_empty_right` | set_ite_fresh_holdout | True | False | False | True | False | False |
| `Set.Nonempty.subset_pair_iff_eq` | set_ite_fresh_holdout | False | False | False | False | False | False |
| `Set.diff_union_inter` | set_ite_fresh_holdout | True | False | False | True | False | False |
| `Set.eq_of_inclusion_surjective` | set_ite_fresh_holdout | False | False | False | False | False | False |
| `Set.inclusion_inclusion` | set_ite_fresh_holdout | True | False | False | True | False | False |
| `Set.inclusion_right` | set_ite_fresh_holdout | True | False | False | True | False | False |
| `Set.inclusion_self` | set_ite_fresh_holdout | True | False | False | True | False | False |
| `Set.insert_diff_eq_singleton` | set_ite_fresh_holdout | True | False | False | True | False | False |
| `Set.insert_diff_of_mem` | set_ite_fresh_holdout | True | False | False | True | False | False |
| `Set.insert_diff_of_not_mem` | set_ite_fresh_holdout | True | False | False | True | False | False |
| `Set.insert_diff_self_of_not_mem` | set_ite_fresh_holdout | True | False | False | True | False | False |
| `Set.monotoneOn_iff_monotone` | set_ite_fresh_holdout | False | False | False | False | False | False |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` | set_ite_fresh_holdout | False | False | False | False | False | False |