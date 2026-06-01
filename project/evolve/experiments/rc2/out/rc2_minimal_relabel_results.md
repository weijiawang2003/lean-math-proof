# RC2 — Minimal-Sufficient Relabel of New Wins

- attribution histogram: `{'TRUE_SET_ITE_SIMP_WIN': 5, 'UNEXPECTED_WIN_NEEDS_REVIEW': 4}`
- **TRUE_SET_ITE_SIMP_WIN = 5** (credited delta): ['Set.ite_empty', 'Set.ite_empty_left', 'Set.ite_empty_right', 'Set.ite_left', 'Set.ite_right']
- Promotion credits only TRUE_SET_ITE_SIMP_WIN. Any BASELINE_DUPLICATE / UNEXPECTED is excluded from the delta.

| theorem | surface | baseline_solved_by | all_failed | attribution |
|---|---|---|---|---|
| `Set.ite_empty_left` | set_ite_known_wins | None | True | **TRUE_SET_ITE_SIMP_WIN** |
| `Set.ite_empty_right` | set_ite_known_wins | None | True | **TRUE_SET_ITE_SIMP_WIN** |
| `Set.ite_left` | set_ite_known_wins | None | True | **TRUE_SET_ITE_SIMP_WIN** |
| `Set.ite_empty` | set_ite_known_wins | None | True | **TRUE_SET_ITE_SIMP_WIN** |
| `Set.ite_right` | set_ite_known_wins | None | True | **TRUE_SET_ITE_SIMP_WIN** |
| `Set.ite_inter_self` | set_ite_selected_failures | None | True | **UNEXPECTED_WIN_NEEDS_REVIEW** |
| `Set.ite_inter` | set_ite_selected_failures | None | True | **UNEXPECTED_WIN_NEEDS_REVIEW** |
| `Set.ite_compl` | set_ite_fresh_holdout | None | True | **UNEXPECTED_WIN_NEEDS_REVIEW** |
| `Set.ite_inter_compl_self` | sf1_frontier_runnable_subset | None | True | **UNEXPECTED_WIN_NEEDS_REVIEW** |