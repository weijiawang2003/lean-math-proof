# TR6 B20 live results

- theorems: 137 | live: 135 | total successes: **21** | new this stage: 9
- first-success rank histogram: {1: 1, 9: 1, 2: 4, 20: 1, 16: 3, 6: 1, 3: 2, 12: 3, 11: 1, 17: 1, 7: 1, 4: 1, 5: 1}

| theorem | ns | success | first_rank | winning tactic |
|---|---|---|---|---|
| `Finset.biUnion_subset_iff_forall_subset` | Finset | True | 1 | `simp [Finset.biUnion_subset] <;> aesop` |
| `Finset.image_subset_iff` | Finset | True | 9 | `simp [Finset.subset_iff]` |
| `List.Forall.imp` | List | True | 2 | `simp [List.forall_iff_forall_mem] <;> aesop` |
| `Multiset.Disjoint.symm` | Multiset | True | 20 | `tauto` |
| `Multiset.add_eq_union_right_of_le` | Multiset | True | 16 | `rw [Multiset.add_eq_union_left_of_le] <;> aesop` |
| `Multiset.coe_disjoint` | Multiset | True | 6 | `aesop` |
| `Multiset.disjoint_add_left` | Multiset | True | 3 | `simp [Multiset.disjoint_left] <;> aesop` |
| `Multiset.disjoint_add_right` | Multiset | True | 12 | `simp [Multiset.disjoint_right] <;> aesop` |
| `Multiset.disjoint_comm` | Multiset | True | 16 | `tauto` |
| `Multiset.disjoint_cons_left` | Multiset | True | 2 | `simp [Multiset.disjoint_left] <;> aesop` |
| `Multiset.disjoint_left` | Multiset | True | 11 | `aesop` |
| `Multiset.disjoint_right` | Multiset | True | 17 | `tauto` |
| `Multiset.singleton_disjoint` | Multiset | True | 3 | `simp [Multiset.disjoint_left]` |
| `Multiset.zero_disjoint` | Multiset | True | 7 | `simp [Multiset.disjoint_left]` |
| `Nat.sqrt_pos` | Nat | True | 16 | `exact Nat.le_sqrt` |
| `Set.disjoint_iUnion_left` | Set | True | 4 | `simp [Set.disjoint_left] <;> aesop` |
| `Set.disjoint_iUnion_right` | Set | True | 2 | `simp [Set.disjoint_left] <;> aesop` |
| `Set.disjoint_sUnion_left` | Set | True | 5 | `simp [Set.disjoint_left] <;> aesop` |
| `Set.disjoint_sUnion_right` | Set | True | 2 | `simp [Set.disjoint_left] <;> aesop` |
| `Set.mapsTo_singleton` | Set | True | 12 | `simp [Set.MapsTo]` |
| `mem_lowerBounds_iff_subset_Ici` |  | True | 12 | `aesop` |
| `AntitoneOn.image_lowerBounds_subset_upperBounds_image` | AntitoneOn | False | None | `` |
| `AntitoneOn.image_upperBounds_subset_lowerBounds_image` | AntitoneOn | False | None | `` |
| `Equiv.bijOn` | Equiv | False | None | `` |
| `Finset.biUnion_subset` | Finset | False | None | `` |
| `Finset.card_filter_le_iff` | Finset | False | None | `` |
| `Finset.card_le_one_iff_subset_singleton` | Finset | False | None | `` |
| `Finset.card_union_eq_card_add_card` | Finset | False | None | `` |
| `Finset.codisjoint_inf_left` | Finset | False | None | `` |
| `Finset.codisjoint_inf_right` | Finset | False | None | `` |
| `Finset.disjoint_biUnion_left` | Finset | False | None | `` |
| `Finset.disjoint_biUnion_right` | Finset | False | None | `` |
| `Finset.disjoint_image` | Finset | False | None | `` |
| `Finset.disjoint_map` | Finset | False | None | `` |
| `Finset.disjoint_sup_left` | Finset | False | None | `` |
| `Finset.disjoint_sup_right` | Finset | False | None | `` |
| `Finset.exists_eq_insert_iff` | Finset | False | None | `` |
| `Finset.fiber_card_ne_zero_iff_mem_image` | Finset | False | None | `` |
| `Finset.image_ssubset_image` | Finset | False | None | `` |
| `Finset.image_subset_iff_subset_preimage` | Finset | False | None | `` |
| `Finset.image_subset_image_iff` | Finset | False | None | `` |
| `Finset.le_card_iff_exists_subset_card` | Finset | False | None | `` |
| `Finset.map_ssubset_map` | Finset | False | None | `` |
| `Finset.map_subset_iff_subset_preimage` | Finset | False | None | `` |
| `Finset.map_subset_map` | Finset | False | None | `` |
| `Finset.map_symm_subset` | Finset | False | None | `` |
| `Finset.mem_powersetCard` | Finset | False | None | `` |
| `Finset.powerset_card_disjiUnion` | Finset | False | None | `` |
| `Finset.subset_iff_eq_of_card_le` | Finset | False | None | `` |
| `Finset.subset_image_iff` | Finset | False | None | `` |
| `Finset.subset_map_iff` | Finset | False | None | `` |
| `Finset.subset_map_symm` | Finset | False | None | `` |
| `IsGLB.of_image` | IsGLB | False | None | `` |
| `IsLUB.of_image` | IsLUB | False | None | `` |
| `List.disjoint_map` | List | False | None | `` |
| `List.disjoint_pmap` | List | False | None | `` |
| `List.dropWhile_eq_nil_iff` | List | False | None | `` |
| `List.eq_replicate_length` | List | False | None | `` |
| `List.forall_cons` | List | False | None | `` |
| `List.forall_iff_forall_mem` | List | False | None | `` |
| `List.forall_map_iff` | List | False | None | `` |
| `List.indexOf_cons_ne` | List | False | None | `` |
| `List.indexOf_eq_length` | List | False | None | `` |
| `List.indexOf_lt_length` | List | False | None | `` |
| `List.length_filter_lt_length_iff_exists` | List | False | None | `` |
| `List.map_bijective_iff` | List | False | None | `` |
| `List.map_injective_iff` | List | False | None | `` |
| `List.map_involutive_iff` | List | False | None | `` |
| `List.map_leftInverse_iff` | List | False | None | `` |
| `List.map_rightInverse_iff` | List | False | None | `` |
| `List.map_subset_iff` | List | False | None | `` |
| `List.map_surjective_iff` | List | False | None | `` |
| `List.mem_dedup` | List | False | None | `` |
| `List.mem_map_of_injective` | List | False | None | `` |
| `List.mem_mem_ranges_iff_lt_natSum` | List | False | None | `` |
| `List.mem_pmap` | List | False | None | `` |
| `List.pairwise_pmap` | List | False | None | `` |
| `List.pmap_eq_nil` | List | False | None | `` |
| `List.ranges_disjoint` | List | False | None | `` |
| `List.ranges_join'` | List | False | None | `` |
| `List.subset_singleton_iff` | List | False | None | `` |
| `List.takeWhile_eq_self_iff` | List | False | None | `` |
| `MonotoneOn.image_lowerBounds_subset_lowerBounds_image` | MonotoneOn | False | None | `` |
| `MonotoneOn.image_upperBounds_subset_upperBounds_image` | MonotoneOn | False | None | `` |
| `Multiset.add_eq_union_iff_disjoint` | Multiset | False | None | `` |
| `Multiset.add_eq_union_left_of_le` | Multiset | False | None | `` |
| `Multiset.card_filter_le_iff` | Multiset | False | None | `` |
| `Multiset.disjoint_iff_ne` | Multiset | False | None | `` |
| `Multiset.disjoint_of_subset_left` | Multiset | False | None | `` |
| `Multiset.disjoint_of_subset_right` | Multiset | False | None | `` |
| `Multiset.disjoint_singleton` | Multiset | False | None | `` |
| `Multiset.disjoint_union_left` | Multiset | False | None | `` |
| `Multiset.inter_eq_zero_iff_disjoint` | Multiset | False | None | `` |
| `Multiset.nodup_bind` | Multiset | False | None | `` |
| `Multiset.toFinset_card_eq_card_iff_nodup` | Multiset | False | None | `` |
| `Nat.ModEq.comm` | Nat | False | None | `` |
| `Nat.le_mul_self` | Nat | False | None | `` |
| `Nat.le_sqrt` | Nat | False | None | `` |
| `Nat.le_sqrt'` | Nat | False | None | `` |
| `Nat.lt_mul_self_iff` | Nat | False | None | `` |
| `Nat.modEq_zero_iff` | Nat | False | None | `` |
| `Nat.modEq_zero_iff_dvd` | Nat | False | None | `` |
| `Nat.mod_eq_iff_lt` | Nat | False | None | `` |
| `Nat.mul_self_inj` | Nat | False | None | `` |
| `Nat.one_le_div_iff` | Nat | False | None | `` |
| `Nat.one_lt_pow_iff` | Nat | False | None | `` |
| `Nat.sqrt_eq_zero` | Nat | False | None | `` |
| `Nat.sqrt_lt'` | Nat | False | None | `` |
| `Option.bind_eq_some'` | Option | False | None | `` |
| `Option.map_eq_id` | Option | False | None | `` |
| `Option.map_inj` | Option | False | None | `` |
| `PLift.down_injective` | PLift | False | None | `` |
| `Set.InjOn.image_eq_image_iff` | Set | False | None | `` |
| `Set.InjOn.image_ssubset_image_iff` | Set | False | None | `` |
| `Set.InjOn.image_subset_image_iff` | Set | False | None | `` |
| `Set.InjOn.mem_image_iff` | Set | False | None | `` |
| `Set.MapsTo.mem_iff` | Set | False | None | `` |
| `Set.biUnion_compl_eq_of_pairwise_disjoint_of_iUnion_eq_univ` | Set | False | None | `` |
| `Set.bijective_iff_bijective_of_iUnion_eq_univ` | Set | False | None | `` |
| `Set.injOn_union` | Set | False | None | `` |
| `Set.kernImage_preimage_eq_iff` | Set | False | None | `` |
| `Set.mapsTo'` | Set | False | None | `` |
| `Set.mapsTo_inter` | Set | False | None | `` |
| `Set.mapsTo_range_iff` | Set | False | None | `` |
| `Set.mapsTo_sInter` | Set | False | None | `` |
| `Set.mapsTo_sUnion` | Set | False | None | `` |
| `Set.mapsTo_union` | Set | False | None | `` |
| `Set.surjOn_iff_exists_map_subtype` | Set | False | None | `` |
| `bddAbove_def` |  | False | None | `` |
| `bddAbove_iff_subset_Iic` |  | False | None | `` |
| `bddBelow_bddAbove_iff_subset_Icc` |  | False | None | `` |
| `bddBelow_def` |  | False | None | `` |
| `bddBelow_iff_subset_Ici` |  | False | None | `` |
| `exists_mem_or` |  | False | None | `` |
| `exists_of_exists_mem` |  | False | None | `` |
| `isGreatest_union_iff` |  | False | None | `` |
| `isLeast_union_iff` |  | False | None | `` |
