# RC4C candidate evaluation (additive, dual-mode)

- metrics: {'num_theorems': 109, 'rc2_solved': 38, 'candidate_solved_all': 57, 'raw_delta_all': 19, 'raw_delta_nonoverlap': 11, 'overlap_rc4b_wins': 8, 'regressions': 0, 'gate_emissions': 42, 'off_gate_emissions': 0, 'emitted_and_solved': 19, 'emitted_and_failed': 19, 'new_wins_by_namespace_all': {'Finset': 1, 'List': 2, 'Multiset': 9, 'Set': 7}, 'new_wins_by_namespace_nonoverlap': {'Finset': 1, 'List': 2, 'Multiset': 7, 'Set': 1}}
- new wins (all): **19** ['Finset.biUnion_subset_iff_forall_subset', 'List.Forall.imp', 'Multiset.disjoint_add_right', 'Set.Nonempty.subset_pair_iff_eq', 'Multiset.disjoint_add_left', 'Multiset.disjoint_cons_left', 'Set.disjoint_iUnion_left', 'Set.disjoint_iUnion_right', 'Set.disjoint_iff_forall_ne', 'Set.disjoint_right', 'Set.disjoint_sUnion_left', 'Set.disjoint_sUnion_right', 'List.forall_map_iff', 'Multiset.disjoint_iff_ne', 'Multiset.disjoint_right', 'Multiset.disjoint_singleton', 'Multiset.disjoint_union_left', 'Multiset.singleton_disjoint', 'Multiset.zero_disjoint']
- new wins (nonoverlap): **11** ['Finset.biUnion_subset_iff_forall_subset', 'List.Forall.imp', 'Multiset.disjoint_add_right', 'Set.Nonempty.subset_pair_iff_eq', 'Multiset.disjoint_add_left', 'List.forall_map_iff', 'Multiset.disjoint_iff_ne', 'Multiset.disjoint_singleton', 'Multiset.disjoint_union_left', 'Multiset.singleton_disjoint', 'Multiset.zero_disjoint']
- overlap-only (RC4B) wins: 8 ['Multiset.disjoint_cons_left', 'Set.disjoint_iUnion_left', 'Set.disjoint_iUnion_right', 'Set.disjoint_iff_forall_ne', 'Set.disjoint_right', 'Set.disjoint_sUnion_left', 'Set.disjoint_sUnion_right', 'Multiset.disjoint_right']

| theorem | sets | ns | rc2 | gate | win_tac | lemma | overlap | new(all) | new(non) | off_gate |
|---|---|---|---|---|---|---|---|---|---|---|
| `Finset.biUnion_subset_iff_forall_subset` | known_wins_all,known_wins_nonoverlap | Finset | F | True | `simp [Finset.biUnion_subset] <;> aesop` | `Finset.biUnion_subset` | none | True | True | False |
| `List.Forall.imp` | known_wins_all,known_wins_nonoverlap | List | F | True | `simp [List.forall_iff_forall_mem] <;> aesop` | `List.forall_iff_forall_mem` | none | True | True | False |
| `List.forall_map_iff` | fresh_holdout_all,fresh_holdout_nonoverlap | List | F | True | `simp [List.forall_iff_forall_mem] <;> aesop` | `List.forall_iff_forall_mem` | none | True | True | False |
| `Multiset.disjoint_add_left` | known_wins_all | Multiset | F | True | `simp [Multiset.disjoint_left] <;> aesop` | `Multiset.disjoint_left` | RC4B | True | True | False |
| `Multiset.disjoint_add_right` | known_wins_all,known_wins_nonoverlap | Multiset | F | True | `simp [Multiset.disjoint_left] <;> aesop` | `Multiset.disjoint_left` | RC4B | True | True | False |
| `Multiset.disjoint_cons_left` | known_wins_all | Multiset | F | True | `simp [Multiset.disjoint_left] <;> aesop` | `Multiset.disjoint_left` | RC4B | True | False | False |
| `Multiset.disjoint_iff_ne` | fresh_holdout_all,fresh_holdout_nonoverlap | Multiset | F | True | `simp [Multiset.disjoint_left] <;> aesop` | `Multiset.disjoint_left` | RC4B | True | True | False |
| `Multiset.disjoint_right` | fresh_holdout_all,fresh_holdout_nonoverlap | Multiset | F | True | `simp [Multiset.disjoint_left] <;> aesop` | `Multiset.disjoint_left` | RC4B | True | False | False |
| `Multiset.disjoint_singleton` | fresh_holdout_all | Multiset | F | True | `simp [Multiset.disjoint_left] <;> aesop` | `Multiset.disjoint_left` | RC4B | True | True | False |
| `Multiset.disjoint_union_left` | fresh_holdout_all | Multiset | F | True | `simp [Multiset.disjoint_left] <;> aesop` | `Multiset.disjoint_left` | RC4B | True | True | False |
| `Multiset.singleton_disjoint` | fresh_holdout_all | Multiset | F | True | `simp [Multiset.disjoint_left] <;> aesop` | `Multiset.disjoint_left` | RC4B | True | True | False |
| `Multiset.zero_disjoint` | fresh_holdout_all | Multiset | F | True | `simp [Multiset.disjoint_left] <;> aesop` | `Multiset.disjoint_left` | RC4B | True | True | False |
| `Set.Nonempty.subset_pair_iff_eq` | known_wins_all,known_wins_nonoverlap | Set | F | True | `simp [Set.subset_pair_iff_eq] <;> aesop` | `Set.subset_pair_iff_eq` | none | True | True | False |
| `Set.disjoint_iUnion_left` | known_wins_all | Set | F | True | `simp [Set.disjoint_left] <;> aesop` | `Set.disjoint_left` | RC4B | True | False | False |
| `Set.disjoint_iUnion_right` | known_wins_all | Set | F | True | `simp [Set.disjoint_left] <;> aesop` | `Set.disjoint_left` | RC4B | True | False | False |
| `Set.disjoint_iff_forall_ne` | known_wins_all | Set | F | True | `simp [Set.disjoint_left] <;> aesop` | `Set.disjoint_left` | RC4B | True | False | False |
| `Set.disjoint_right` | known_wins_all | Set | F | True | `simp [Set.disjoint_left] <;> aesop` | `Set.disjoint_left` | RC4B | True | False | False |
| `Set.disjoint_sUnion_left` | known_wins_all | Set | F | True | `simp [Set.disjoint_left] <;> aesop` | `Set.disjoint_left` | RC4B | True | False | False |
| `Set.disjoint_sUnion_right` | known_wins_all | Set | F | True | `simp [Set.disjoint_left] <;> aesop` | `Set.disjoint_left` | RC4B | True | False | False |
| `Finset.biUnion_subset` | fresh_holdout_all,fresh_holdout_nonoverlap | Finset | F | True | `` | `` |  | False | False | False |
| `Finset.card_union_eq_card_add_card` | negative_controls | Finset | F | False | `` | `` |  | False | False | False |
| `Finset.codisjoint_inf_left` | negative_controls | Finset | F | False | `` | `` |  | False | False | False |
| `Finset.codisjoint_inf_right` | negative_controls | Finset | F | False | `` | `` |  | False | False | False |
| `Finset.disjiUnion_cons` | negative_controls | Finset | S | False | `` | `` |  | False | False | False |
| `Finset.disjiUnion_filter_eq` | negative_controls | Finset | S | False | `` | `` |  | False | False | False |
| `Finset.disjiUnion_filter_eq_of_maps_to` | negative_controls | Finset | S | False | `` | `` |  | False | False | False |
| `Finset.disjiUnion_map` | negative_controls | Finset | S | False | `` | `` |  | False | False | False |
| `Finset.disjoint_biUnion_left` | fresh_holdout_all,fresh_holdout_nonoverlap | Finset | F | True | `` | `` |  | False | False | False |
| `Finset.disjoint_biUnion_right` | fresh_holdout_all,fresh_holdout_nonoverlap | Finset | F | True | `` | `` |  | False | False | False |
| `Finset.disjoint_filter_filter'` | negative_controls | Finset | F | False | `` | `` |  | False | False | False |
| `Finset.disjoint_image` | negative_controls | Finset | F | False | `` | `` |  | False | False | False |
| `Finset.disjoint_insert_right` | canonical_smoke | Finset | F | False | `` | `` |  | False | False | False |
| `Finset.disjoint_map` | negative_controls | Finset | F | False | `` | `` |  | False | False | False |
| `Finset.disjoint_sup_left` | negative_controls | Finset | F | False | `` | `` |  | False | False | False |
| `Finset.disjoint_sup_right` | negative_controls | Finset | F | False | `` | `` |  | False | False | False |
| `Finset.filter_cons` | negative_controls | Finset | S | False | `` | `` |  | False | False | False |
| `Finset.mem_insert` | canonical_smoke | Finset | F | False | `` | `` |  | False | False | False |
| `Finset.mem_singleton` | canonical_smoke | Finset | F | False | `` | `` |  | False | False | False |
| `Finset.pairwise_cons'` | negative_controls | Finset | F | False | `` | `` |  | False | False | False |
| `Finset.powerset_card_disjiUnion` | negative_controls | Finset | F | False | `` | `` |  | False | False | False |
| `List.disjoint_map` | negative_controls | List | F | False | `` | `` |  | False | False | False |
| `List.disjoint_pmap` | negative_controls | List | F | False | `` | `` |  | False | False | False |
| `List.filterMap_eq_map_iff_forall_eq_some` | fresh_holdout_all,fresh_holdout_nonoverlap | List | F | True | `` | `` |  | False | False | False |
| `List.forall_cons` | fresh_holdout_all,fresh_holdout_nonoverlap | List | F | True | `` | `` |  | False | False | False |
| `List.forall_iff_forall_mem` | fresh_holdout_all,fresh_holdout_nonoverlap | List | F | True | `` | `` |  | False | False | False |
| `List.mem_pair` | negative_controls | List | S | False | `` | `` |  | False | False | False |
| `Multiset.Disjoint.symm` | fresh_holdout_all,fresh_holdout_nonoverlap | Multiset | F | True | `` | `` |  | False | False | False |
| `Multiset.add_eq_union_iff_disjoint` | fresh_holdout_all,fresh_holdout_nonoverlap | Multiset | F | True | `` | `` |  | False | False | False |
| `Multiset.add_eq_union_left_of_le` | fresh_holdout_all,fresh_holdout_nonoverlap | Multiset | F | True | `` | `` |  | False | False | False |
| `Multiset.add_eq_union_right_of_le` | fresh_holdout_all,fresh_holdout_nonoverlap | Multiset | F | True | `` | `` |  | False | False | False |
| `Multiset.coe_disjoint` | fresh_holdout_all,fresh_holdout_nonoverlap | Multiset | F | True | `` | `` |  | False | False | False |
| `Multiset.disjoint_comm` | fresh_holdout_all,fresh_holdout_nonoverlap | Multiset | F | True | `` | `` |  | False | False | False |
| `Multiset.disjoint_cons_right` | fresh_holdout_all,fresh_holdout_nonoverlap | Multiset | S | True | `simp [Multiset.disjoint_right] <;> aesop` | `Multiset.disjoint_right` | none | False | False | False |
| `Multiset.disjoint_left` | fresh_holdout_all,fresh_holdout_nonoverlap | Multiset | F | True | `` | `` |  | False | False | False |
| `Multiset.disjoint_map_map` | fresh_holdout_all,fresh_holdout_nonoverlap | Multiset | S | True | `simp [Multiset.disjoint_left] <;> aesop` | `Multiset.disjoint_left` | RC4B | False | False | False |
| `Multiset.disjoint_of_subset_left` | fresh_holdout_all,fresh_holdout_nonoverlap | Multiset | F | True | `` | `` |  | False | False | False |
| `Multiset.disjoint_of_subset_right` | fresh_holdout_all,fresh_holdout_nonoverlap | Multiset | F | True | `` | `` |  | False | False | False |
| `Multiset.disjoint_toFinset` | fresh_holdout_all | Multiset | S | True | `` | `` |  | False | False | False |
| `Multiset.disjoint_union_right` | fresh_holdout_all | Multiset | S | True | `simp [Multiset.disjoint_left] <;> aesop` | `Multiset.disjoint_left` | RC4B | False | False | False |
| `Multiset.inter_eq_zero_iff_disjoint` | fresh_holdout_all | Multiset | F | True | `` | `` |  | False | False | False |
| `Multiset.nodup_bind` | fresh_holdout_all | Multiset | F | True | `` | `` |  | False | False | False |
| `Nat.AM_GM` | namespace_negative_controls,canonical_smoke,canonical_smoke | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.Coprime.dvd_mul_left` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.Coprime.dvd_mul_right` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.Coprime.eq_of_mul_eq_zero` | namespace_negative_controls | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.Coprime.lcm_eq_mul` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.ModEq.add_le_of_lt` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.ModEq.cancel_left_of_coprime` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.ModEq.cancel_right_of_coprime` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.ModEq.comm` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.ModEq.dvd_iff` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.ModEq.eq_of_abs_lt` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.ModEq.eq_of_lt_of_lt` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.ModEq.gcd_eq` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.ModEq.le_of_lt_add` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.ModEq.mul_left_cancel_iff'` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.ModEq.mul_right_cancel_iff'` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.add_def` | namespace_negative_controls | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.add_div` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.add_div_eq_of_add_mod_lt` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.add_div_eq_of_le_mod_add_mod` | namespace_negative_controls | Nat | F | False | `` | `` |  | False | False | False |
| `Nat.add_eq_left` | canonical_smoke,canonical_smoke | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.add_eq_max_iff` | canonical_smoke,canonical_smoke | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.add_eq_min_iff` | canonical_smoke,canonical_smoke | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.add_eq_one_iff` | canonical_smoke,canonical_smoke | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.add_eq_right` | canonical_smoke,canonical_smoke | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.add_mod_eq_add_mod_left` | canonical_smoke,canonical_smoke | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.add_mod_eq_add_mod_right` | canonical_smoke,canonical_smoke | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.add_mod_eq_ite` | canonical_smoke,canonical_smoke | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.div_le_div_right` | canonical_smoke,canonical_smoke | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.div_lt_iff_lt_mul'` | canonical_smoke,canonical_smoke | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.div_lt_one_iff` | canonical_smoke,canonical_smoke | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.half_le_of_sub_le_half` | canonical_smoke,canonical_smoke | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.le_and_le_add_one_iff` | canonical_smoke,canonical_smoke | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.le_or_le_of_add_eq_add_pred` | canonical_smoke,canonical_smoke | Nat | S | False | `` | `` |  | False | False | False |
| `Nat.mul_add_mod'` | canonical_smoke | Nat | S | False | `` | `` |  | False | False | False |
| `Set._root_.Disjoint.image` | fresh_holdout_all | Set | F | True | `` | `` |  | False | False | False |
| `Set.biUnion_compl_eq_of_pairwise_disjoint_of_iUnion_eq_univ` | fresh_holdout_all | Set | F | True | `` | `` |  | False | False | False |
| `Set.empty_subset` | canonical_smoke | Set | S | False | `` | `` |  | False | False | False |
| `Set.empty_union` | canonical_smoke | Set | S | False | `` | `` |  | False | False | False |
| `Set.inter_comm` | canonical_smoke | Set | S | False | `` | `` |  | False | False | False |
| `Set.inter_univ` | canonical_smoke | Set | S | False | `` | `` |  | False | False | False |
| `Set.ite_univ` | canonical_smoke | Set | S | False | `` | `` |  | False | False | False |
| `Set.mem_inter_iff` | canonical_smoke | Set | S | False | `` | `` |  | False | False | False |
| `Set.mem_union` | canonical_smoke | Set | S | False | `` | `` |  | False | False | False |
| `Set.subset_univ` | canonical_smoke | Set | S | False | `` | `` |  | False | False | False |
| `Set.union_comm` | canonical_smoke | Set | S | False | `` | `` |  | False | False | False |
| `Set.union_empty` | canonical_smoke | Set | S | False | `` | `` |  | False | False | False |
| `Set.univ_inter` | canonical_smoke | Set | S | False | `` | `` |  | False | False | False |
