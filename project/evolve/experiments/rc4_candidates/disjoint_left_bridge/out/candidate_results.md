# RC4B candidate evaluation (additive)

- metrics: {'num_theorems': 103, 'rc2_solved': 38, 'candidate_solved': 54, 'raw_delta': 16, 'new_wins': 16, 'regressions': 0, 'gate_emissions': 39, 'off_gate_emissions': 0, 'emitted_and_solved': 16, 'emitted_and_failed': 18, 'new_wins_by_namespace': {'Set': 7, 'Multiset': 9}, 'gate_emissions_by_namespace': {'Set': 15, 'Multiset': 24}}
- new wins over literal RC2: **16** ['Multiset.disjoint_add_left', 'Multiset.disjoint_cons_left', 'Multiset.singleton_disjoint', 'Multiset.zero_disjoint', 'Set.disjoint_iUnion_left', 'Set.disjoint_iUnion_right', 'Set.disjoint_iff_forall_ne', 'Set.disjoint_right', 'Set.disjoint_sUnion_left', 'Set.disjoint_sUnion_right', 'Set.disjoint_singleton_left', 'Multiset.disjoint_add_right', 'Multiset.disjoint_iff_ne', 'Multiset.disjoint_right', 'Multiset.disjoint_singleton', 'Multiset.disjoint_union_left']

| theorem | sets | ns | rc2 | gate | win_tac | new_win | off_gate |
|---|---|---|---|---|---|---|---|
| `Multiset.disjoint_add_left` | known_wins | Multiset | F | True | `simp [Multiset.disjoint_left] <;> aesop` | True | False |
| `Multiset.disjoint_add_right` | fresh_holdout_multiset | Multiset | F | True | `simp [Multiset.disjoint_left] <;> aesop` | True | False |
| `Multiset.disjoint_cons_left` | known_wins | Multiset | F | True | `simp [Multiset.disjoint_left]` | True | False |
| `Multiset.disjoint_iff_ne` | fresh_holdout_multiset | Multiset | F | True | `simp [Multiset.disjoint_left] <;> aesop` | True | False |
| `Multiset.disjoint_right` | fresh_holdout_multiset | Multiset | F | True | `simp [Multiset.disjoint_left] <;> aesop` | True | False |
| `Multiset.disjoint_singleton` | fresh_holdout_multiset | Multiset | F | True | `simp [Multiset.disjoint_left] <;> aesop` | True | False |
| `Multiset.disjoint_union_left` | fresh_holdout_multiset | Multiset | F | True | `simp [Multiset.disjoint_left] <;> aesop` | True | False |
| `Multiset.singleton_disjoint` | known_wins | Multiset | F | True | `simp [Multiset.disjoint_left]` | True | False |
| `Multiset.zero_disjoint` | known_wins | Multiset | F | True | `simp [Multiset.disjoint_left]` | True | False |
| `Set.disjoint_iUnion_left` | known_wins | Set | F | True | `simp [Set.disjoint_left] <;> aesop` | True | False |
| `Set.disjoint_iUnion_right` | known_wins | Set | F | True | `simp [Set.disjoint_left] <;> aesop` | True | False |
| `Set.disjoint_iff_forall_ne` | known_wins | Set | F | True | `simp [Set.disjoint_left] <;> aesop` | True | False |
| `Set.disjoint_right` | known_wins | Set | F | True | `simp [Set.disjoint_left] <;> aesop` | True | False |
| `Set.disjoint_sUnion_left` | known_wins | Set | F | True | `simp [Set.disjoint_left] <;> aesop` | True | False |
| `Set.disjoint_sUnion_right` | known_wins | Set | F | True | `simp [Set.disjoint_left] <;> aesop` | True | False |
| `Set.disjoint_singleton_left` | known_wins | Set | F | True | `simp [Set.disjoint_left]` | True | False |
| `Finset.card_union_eq_card_add_card` | disjoint_negative_controls | Finset | F | False | `` | False | False |
| `Finset.codisjoint_inf_left` | disjoint_negative_controls | Finset | F | False | `` | False | False |
| `Finset.codisjoint_inf_right` | disjoint_negative_controls | Finset | F | False | `` | False | False |
| `Finset.disjiUnion_cons` | disjoint_negative_controls | Finset | S | False | `` | False | False |
| `Finset.disjiUnion_filter_eq` | disjoint_negative_controls | Finset | S | False | `` | False | False |
| `Finset.disjiUnion_filter_eq_of_maps_to` | disjoint_negative_controls | Finset | S | False | `` | False | False |
| `Finset.disjiUnion_map` | disjoint_negative_controls | Finset | S | False | `` | False | False |
| `Finset.disjoint_biUnion_left` | disjoint_negative_controls | Finset | F | False | `` | False | False |
| `Finset.disjoint_biUnion_right` | disjoint_negative_controls | Finset | F | False | `` | False | False |
| `Finset.disjoint_filter_filter'` | disjoint_negative_controls | Finset | F | False | `` | False | False |
| `Finset.disjoint_image` | disjoint_negative_controls | Finset | F | False | `` | False | False |
| `Finset.disjoint_insert_right` | canonical_smoke | Finset | F | False | `` | False | False |
| `Finset.disjoint_map` | disjoint_negative_controls | Finset | F | False | `` | False | False |
| `Finset.disjoint_sup_left` | disjoint_negative_controls | Finset | F | False | `` | False | False |
| `Finset.disjoint_sup_right` | disjoint_negative_controls | Finset | F | False | `` | False | False |
| `Finset.filter_cons` | disjoint_negative_controls | Finset | S | False | `` | False | False |
| `Finset.mem_insert` | canonical_smoke | Finset | F | False | `` | False | False |
| `Finset.mem_singleton` | canonical_smoke | Finset | F | False | `` | False | False |
| `Multiset.Disjoint.symm` | fresh_holdout_multiset | Multiset | F | True | `` | False | False |
| `Multiset.add_eq_union_iff_disjoint` | fresh_holdout_multiset | Multiset | F | True | `` | False | False |
| `Multiset.add_eq_union_left_of_le` | fresh_holdout_multiset | Multiset | F | True | `` | False | False |
| `Multiset.add_eq_union_right_of_le` | fresh_holdout_multiset | Multiset | F | True | `` | False | False |
| `Multiset.coe_disjoint` | fresh_holdout_multiset | Multiset | F | True | `` | False | False |
| `Multiset.disjoint_comm` | fresh_holdout_multiset | Multiset | F | True | `` | False | False |
| `Multiset.disjoint_cons_right` | fresh_holdout_multiset | Multiset | S | True | `` | False | False |
| `Multiset.disjoint_left` | fresh_holdout_multiset | Multiset | F | True | `` | False | False |
| `Multiset.disjoint_map_map` | fresh_holdout_multiset | Multiset | S | True | `simp [Multiset.disjoint_left] <;> aesop` | False | False |
| `Multiset.disjoint_of_subset_left` | fresh_holdout_multiset | Multiset | F | True | `` | False | False |
| `Multiset.disjoint_of_subset_right` | fresh_holdout_multiset | Multiset | F | True | `` | False | False |
| `Multiset.disjoint_toFinset` | fresh_holdout_multiset | Multiset | S | True | `` | False | False |
| `Multiset.disjoint_union_right` | fresh_holdout_multiset | Multiset | S | True | `simp [Multiset.disjoint_left] <;> aesop` | False | False |
| `Multiset.inter_eq_zero_iff_disjoint` | fresh_holdout_multiset | Multiset | F | True | `` | False | False |
| `Multiset.nodup_bind` | fresh_holdout_multiset | Multiset | F | True | `` | False | False |
| `Nat.AM_GM` | namespace_negative_controls,canonical_smoke,canonical_smoke | Nat | F | False | `` | False | False |
| `Nat.Coprime.dvd_mul_left` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.Coprime.dvd_mul_right` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.Coprime.eq_of_mul_eq_zero` | namespace_negative_controls | Nat | S | False | `` | False | False |
| `Nat.Coprime.lcm_eq_mul` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.ModEq.add_le_of_lt` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.ModEq.cancel_left_of_coprime` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.ModEq.cancel_right_of_coprime` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.ModEq.comm` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.ModEq.dvd_iff` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.ModEq.eq_of_abs_lt` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.ModEq.eq_of_lt_of_lt` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.ModEq.gcd_eq` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.ModEq.le_of_lt_add` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.ModEq.mul_left_cancel_iff'` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.ModEq.mul_right_cancel_iff'` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.add_def` | namespace_negative_controls | Nat | S | False | `` | False | False |
| `Nat.add_div` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.add_div_eq_of_add_mod_lt` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.add_div_eq_of_le_mod_add_mod` | namespace_negative_controls | Nat | F | False | `` | False | False |
| `Nat.add_eq_left` | canonical_smoke,canonical_smoke | Nat | S | False | `` | False | False |
| `Nat.add_eq_max_iff` | canonical_smoke,canonical_smoke | Nat | S | False | `` | False | False |
| `Nat.add_eq_min_iff` | canonical_smoke,canonical_smoke | Nat | S | False | `` | False | False |
| `Nat.add_eq_one_iff` | canonical_smoke,canonical_smoke | Nat | S | False | `` | False | False |
| `Nat.add_eq_right` | canonical_smoke,canonical_smoke | Nat | S | False | `` | False | False |
| `Nat.add_mod_eq_add_mod_left` | canonical_smoke,canonical_smoke | Nat | S | False | `` | False | False |
| `Nat.add_mod_eq_add_mod_right` | canonical_smoke,canonical_smoke | Nat | S | False | `` | False | False |
| `Nat.add_mod_eq_ite` | canonical_smoke,canonical_smoke | Nat | S | False | `` | False | False |
| `Nat.div_le_div_right` | canonical_smoke,canonical_smoke | Nat | S | False | `` | False | False |
| `Nat.div_lt_iff_lt_mul'` | canonical_smoke,canonical_smoke | Nat | S | False | `` | False | False |
| `Nat.div_lt_one_iff` | canonical_smoke,canonical_smoke | Nat | S | False | `` | False | False |
| `Nat.half_le_of_sub_le_half` | canonical_smoke,canonical_smoke | Nat | S | False | `` | False | False |
| `Nat.le_and_le_add_one_iff` | canonical_smoke,canonical_smoke | Nat | S | False | `` | False | False |
| `Nat.le_or_le_of_add_eq_add_pred` | canonical_smoke,canonical_smoke | Nat | S | False | `` | False | False |
| `Nat.mul_add_mod'` | canonical_smoke | Nat | S | False | `` | False | False |
| `Set._root_.Disjoint.image` | fresh_holdout_set | Set | F | True | `` | False | False |
| `Set.biUnion_compl_eq_of_pairwise_disjoint_of_iUnion_eq_univ` | fresh_holdout_set | Set | F | True | `` | False | False |
| `Set.disjoint_iUnion` | fresh_holdout_set | Set | F | True | `` | False | False |
| `Set.disjoint_singleton` | fresh_holdout_set | Set | S | True | `simp [Set.disjoint_left]` | False | False |
| `Set.empty_subset` | canonical_smoke | Set | S | False | `` | False | False |
| `Set.empty_union` | canonical_smoke | Set | S | False | `` | False | False |
| `Set.injOn_union` | fresh_holdout_set | Set | F | True | `` | False | False |
| `Set.inter_comm` | canonical_smoke | Set | S | False | `` | False | False |
| `Set.inter_univ` | canonical_smoke | Set | S | False | `` | False | False |
| `Set.ite_univ` | canonical_smoke | Set | S | False | `` | False | False |
| `Set.mem_inter_iff` | canonical_smoke | Set | S | False | `` | False | False |
| `Set.mem_union` | canonical_smoke | Set | S | False | `` | False | False |
| `Set.pairwiseDisjoint_filter` | fresh_holdout_set | Set | F | True | `` | False | False |
| `Set.sigmaToiUnion_bijective` | fresh_holdout_set | Set | F | True | `` | False | False |
| `Set.sigmaToiUnion_injective` | fresh_holdout_set | Set | F | True | `` | False | False |
| `Set.subset_univ` | canonical_smoke | Set | S | False | `` | False | False |
| `Set.union_comm` | canonical_smoke | Set | S | False | `` | False | False |
| `Set.union_empty` | canonical_smoke | Set | S | False | `` | False | False |
| `Set.univ_inter` | canonical_smoke | Set | S | False | `` | False | False |
