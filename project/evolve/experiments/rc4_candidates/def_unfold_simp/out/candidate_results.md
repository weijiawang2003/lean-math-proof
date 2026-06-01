# RC4A candidate evaluation (additive)

- metrics: {'num_theorems': 61, 'rc2_solved': 30, 'candidate_solved': 35, 'new_wins': 5, 'regressions': 0, 'gate_emissions': 11, 'off_gate_emissions': 0, 'emitted_and_solved': 5, 'emitted_and_failed': 6}
- new wins over literal RC2: **5** ['Finset.mem_disjUnion', 'Set.antitoneOn_iff_antitone', 'Set.monotoneOn_iff_monotone', 'Set.strictAntiOn_iff_strictAnti', 'Set.strictMonoOn_iff_strictMono']

| theorem | sets | rc2 | gate | probe | new_win | off_gate |
|---|---|---|---|---|---|---|
| `Finset.mem_disjUnion` | known_wins | F | True | success | True | False |
| `Set.antitoneOn_iff_antitone` | known_wins | F | True | success | True | False |
| `Set.monotoneOn_iff_monotone` | known_wins | F | True | success | True | False |
| `Set.strictAntiOn_iff_strictAnti` | known_wins | F | True | success | True | False |
| `Set.strictMonoOn_iff_strictMono` | known_wins | F | True | success | True | False |
| `Finset.coe_disjUnion` | fresh_frontier_holdout | S | True | proof_failed | False | False |
| `Finset.disjUnion_eq_union` | fresh_frontier_holdout | S | True | proof_failed | False | False |
| `Finset.disjUnion_singleton` | fresh_frontier_holdout | S | True | proof_failed | False | False |
| `Finset.disjoint_insert_right` | canonical_smoke | F | False |  | False | False |
| `Finset.filter_cons` | same_cluster_holdout | S | True | proof_failed | False | False |
| `Finset.mem_insert` | canonical_smoke | F | False |  | False | False |
| `Finset.mem_singleton` | canonical_smoke | F | False |  | False | False |
| `List.perm_of_nodup_nodup_toFinset_eq` | negative_controls | F | False |  | False | False |
| `List.toFinset.ext_iff` | negative_controls | F | False |  | False | False |
| `List.toFinset_eq` | negative_controls | F | False |  | False | False |
| `List.toFinset_eq_empty_iff` | negative_controls | F | False |  | False | False |
| `List.toFinset_eq_iff_perm_dedup` | negative_controls | F | False |  | False | False |
| `List.toFinset_filter` | negative_controls | F | False |  | False | False |
| `List.toFinset_nonempty_iff` | negative_controls | F | False |  | False | False |
| `List.toFinset_surj_on` | negative_controls | F | False |  | False | False |
| `Multiset.Nodup.toFinset_inj` | negative_controls | F | False |  | False | False |
| `Multiset.toFinset_eq_singleton_iff` | negative_controls | F | False |  | False | False |
| `Multiset.toFinset_ssubset` | negative_controls | F | False |  | False | False |
| `Multiset.toFinset_subset` | negative_controls | F | False |  | False | False |
| `Nat.AM_GM` | canonical_smoke,canonical_smoke | F | False |  | False | False |
| `Nat.add_eq_left` | canonical_smoke,canonical_smoke | S | False |  | False | False |
| `Nat.add_eq_max_iff` | canonical_smoke,canonical_smoke | S | False |  | False | False |
| `Nat.add_eq_min_iff` | canonical_smoke,canonical_smoke | S | False |  | False | False |
| `Nat.add_eq_one_iff` | canonical_smoke,canonical_smoke | S | False |  | False | False |
| `Nat.add_eq_right` | canonical_smoke,canonical_smoke | S | False |  | False | False |
| `Nat.add_mod_eq_add_mod_left` | canonical_smoke,canonical_smoke | S | False |  | False | False |
| `Nat.add_mod_eq_add_mod_right` | canonical_smoke,canonical_smoke | S | False |  | False | False |
| `Nat.add_mod_eq_ite` | canonical_smoke,canonical_smoke | S | False |  | False | False |
| `Nat.add_sub_one_le_mul` | negative_controls | F | False |  | False | False |
| `Nat.diag_induction` | negative_controls | F | False |  | False | False |
| `Nat.div_div_div_eq_div` | negative_controls | F | False |  | False | False |
| `Nat.div_eq_iff_eq_of_dvd_dvd` | negative_controls | F | False |  | False | False |
| `Nat.div_eq_self` | negative_controls | F | False |  | False | False |
| `Nat.div_eq_sub_mod_div` | negative_controls | F | False |  | False | False |
| `Nat.div_le_div_right` | canonical_smoke,canonical_smoke | S | False |  | False | False |
| `Nat.div_le_of_le_mul'` | negative_controls | F | False |  | False | False |
| `Nat.div_le_self'` | negative_controls | F | False |  | False | False |
| `Nat.div_lt_iff_lt_mul'` | canonical_smoke,canonical_smoke | S | False |  | False | False |
| `Nat.div_lt_one_iff` | canonical_smoke,canonical_smoke | S | False |  | False | False |
| `Nat.half_le_of_sub_le_half` | canonical_smoke,canonical_smoke | S | False |  | False | False |
| `Nat.le_and_le_add_one_iff` | canonical_smoke,canonical_smoke | S | False |  | False | False |
| `Nat.le_or_le_of_add_eq_add_pred` | canonical_smoke,canonical_smoke | S | False |  | False | False |
| `Nat.mul_add_mod'` | canonical_smoke | S | False |  | False | False |
| `Set.empty_subset` | canonical_smoke | S | False |  | False | False |
| `Set.empty_union` | canonical_smoke | S | False |  | False | False |
| `Set.inter_comm` | canonical_smoke | S | False |  | False | False |
| `Set.inter_univ` | canonical_smoke | S | False |  | False | False |
| `Set.ite_univ` | canonical_smoke | S | False |  | False | False |
| `Set.mem_inter_iff` | canonical_smoke | S | False |  | False | False |
| `Set.mem_union` | canonical_smoke | S | False |  | False | False |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le` | same_cluster_holdout | F | True | proof_failed | False | False |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` | same_cluster_holdout | F | True | proof_failed | False | False |
| `Set.subset_univ` | canonical_smoke | S | False |  | False | False |
| `Set.union_comm` | canonical_smoke | S | False |  | False | False |
| `Set.union_empty` | canonical_smoke | S | False |  | False | False |
| `Set.univ_inter` | canonical_smoke | S | False |  | False | False |
