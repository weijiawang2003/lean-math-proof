# RC2 vs RC1 — Comparison

- **total delta = 18** | new wins = 18 | regressions = 0 | off-gate = 0
- canonical floors pass: **True** — {'demo_v1': {'rc2_solved': 11, 'floor': '>=11/15', 'pass': True}, 'nat_defs_medium': {'rc2_solved': 37, 'floor': '>=37/38', 'pass': True}, 'nat_defs_large_v5': {'rc2_solved': 49, 'floor': '>=49/65', 'pass': True}}

| surface | role | rc1 | rc2 | delta | new_wins | regr | gate_emit |
|---|---|---|---|---|---|---|---|
| demo_v1 | canonical_floor | 11 | 11 | 0 | 0 | 0 | 1 |
| nat_defs_medium | canonical_floor | 37 | 37 | 0 | 0 | 0 | 0 |
| nat_defs_large_v5 | canonical_floor | 49 | 49 | 0 | 0 | 0 | 0 |
| set_ite_known_wins | candidate_validation | 0 | 5 | 5 | 5 | 0 | 5 |
| set_ite_selected_failures | candidate_validation | 0 | 4 | 4 | 4 | 0 | 5 |
| set_ite_fresh_holdout | candidate_validation | 11 | 15 | 4 | 4 | 0 | 5 |
| sf1_frontier_runnable_subset | fresh_frontier | 5 | 10 | 5 | 5 | 0 | 7 |
| set_ite_negative_controls | negative_control | 0 | 0 | 0 | 0 | 0 | 0 |

## New-win classification
- `Set.ite_empty_left` (set_ite_known_wins) → **fresh_SET_ITE_win** via `simp [Set.ite]`
- `Set.ite_empty_right` (set_ite_known_wins) → **expected_SET_ITE_win** via `simp [Set.ite]`
- `Set.ite_left` (set_ite_known_wins) → **fresh_SET_ITE_win** via `simp [Set.ite]`
- `Set.ite_empty` (set_ite_known_wins) → **fresh_SET_ITE_win** via `simp [Set.ite]`
- `Set.ite_right` (set_ite_known_wins) → **expected_SET_ITE_win** via `simp [Set.ite]`
- `Set.ite_inter_self` (set_ite_selected_failures) → **search_perturbation_multistep_NOT_credited** via `aesop`
- `Set.ite_inter` (set_ite_selected_failures) → **search_perturbation_multistep_NOT_credited** via `aesop`
- `Set.ite_empty_right` (set_ite_selected_failures) → **expected_SET_ITE_win** via `simp [Set.ite]`
- `Set.ite_right` (set_ite_selected_failures) → **expected_SET_ITE_win** via `simp [Set.ite]`
- `Set.ite_compl` (set_ite_fresh_holdout) → **search_perturbation_multistep_NOT_credited** via `aesop`
- `Set.ite_left` (set_ite_fresh_holdout) → **fresh_SET_ITE_win** via `simp [Set.ite]`
- `Set.ite_empty` (set_ite_fresh_holdout) → **fresh_SET_ITE_win** via `simp [Set.ite]`
- `Set.ite_empty_left` (set_ite_fresh_holdout) → **fresh_SET_ITE_win** via `simp [Set.ite]`
- `Set.ite_inter_self` (sf1_frontier_runnable_subset) → **search_perturbation_multistep_NOT_credited** via `aesop`
- `Set.ite_empty_right` (sf1_frontier_runnable_subset) → **expected_SET_ITE_win** via `simp [Set.ite]`
- `Set.ite_right` (sf1_frontier_runnable_subset) → **expected_SET_ITE_win** via `simp [Set.ite]`
- `Set.ite_inter` (sf1_frontier_runnable_subset) → **search_perturbation_multistep_NOT_credited** via `aesop`
- `Set.ite_inter_compl_self` (sf1_frontier_runnable_subset) → **search_perturbation_multistep_NOT_credited** via `aesop`

## Emitted-and-failed (gate fired, no win)
- `Set.ite_univ` (demo_v1) → harmless_failed_emission
- `Set.ite_eq_of_subset_left` (set_ite_selected_failures) → harmless_failed_emission
- `Set.ite_inter_of_inter_eq` (set_ite_fresh_holdout) → harmless_failed_emission
- `Set.ite_eq_of_subset_left` (sf1_frontier_runnable_subset) → harmless_failed_emission
- `Set.ite_eq_of_subset_right` (sf1_frontier_runnable_subset) → harmless_failed_emission

## Regressions
- none