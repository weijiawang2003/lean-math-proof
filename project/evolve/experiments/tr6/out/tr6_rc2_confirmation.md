# TR6 RC2 confirmation (live, fresh)

- cases: 200 | **confirmed RC2 failures: 137** (≥50 target: True)
- classifications: {'CONFIRMED_RC2_FAILURE': 137, 'OPEN_FLAKE': 4, 'RC2_SOLVED': 59}
- confirmed failures by namespace: {'Set': 21, 'Finset': 30, 'List': 29, 'Nat': 14, 'Multiset': 22, '': 10, 'AntitoneOn': 2, 'MonotoneOn': 2, 'IsGLB': 1, 'IsLUB': 1, 'Option': 3, 'PLift': 1, 'Equiv': 1}

| theorem | ns | class | finished |
|---|---|---|---|
| `Set.disjoint_sUnion_left` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.disjoint_sUnion_right` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.injOn_union` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.disjoint_iUnion_left` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.disjoint_iUnion_right` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.disjoint_iUnion` | Set | OPEN_FLAKE | None |
| `Set._root_.Disjoint.image` | Set | OPEN_FLAKE | None |
| `Set.InjOn.mem_image_iff` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.sUnion_subset_iff` | Set | RC2_SOLVED | True |
| `Set.subset_sInter_iff` | Set | RC2_SOLVED | True |
| `Set.mapsTo_sInter` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.mapsTo_sUnion` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.mapsTo'` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.kernImage_preimage_eq_iff` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.InjOn.image_eq_image_iff` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.InjOn.image_subset_image_iff` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.InjOn.image_ssubset_image_iff` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.surjOn_iff_exists_map_subtype` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.biUnion_compl_eq_of_pairwise_disjoint_of_iUnion_eq_univ` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.iUnion_subset_iff` | Set | RC2_SOLVED | True |
| `Set.subset_iInter_iff` | Set | RC2_SOLVED | True |
| `Set.seq_subset` | Set | RC2_SOLVED | True |
| `Set.image_sInter_subset` | Set | RC2_SOLVED | True |
| `Set.mapsTo_univ_iff` | Set | RC2_SOLVED | True |
| `Set.mapsTo_singleton` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.mapsTo_inter` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.mapsTo_union` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.mapsTo_range_iff` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.MapsTo.mem_iff` | Set | CONFIRMED_RC2_FAILURE | False |
| `Set.bijective_iff_bijective_of_iUnion_eq_univ` | Set | CONFIRMED_RC2_FAILURE | False |
| `Finset.card_union_eq_card_add_card` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.disjoint_biUnion_left` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.disjoint_biUnion_right` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.card_filter_le_iff` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.disjoint_map` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.disjoint_image` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.disjoint_sup_left` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.disjoint_sup_right` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.codisjoint_inf_left` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.codisjoint_inf_right` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.disjiUnion_filter_eq_of_maps_to` | Finset | RC2_SOLVED | True |
| `Finset.disjiUnion_filter_eq` | Finset | RC2_SOLVED | True |
| `Finset.image_subset_iff` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.mem_powersetCard` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.exists_eq_insert_iff` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.biUnion_subset` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.biUnion_subset_iff_forall_subset` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.fiber_card_ne_zero_iff_mem_image` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.disjiUnion_map` | Finset | RC2_SOLVED | True |
| `Finset.powerset_card_disjiUnion` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.subset_iff_eq_of_card_le` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.le_card_iff_exists_subset_card` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.map_ssubset_map` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.map_subset_map` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.map_symm_subset` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.subset_map_symm` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.card_le_one_iff_subset_singleton` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.image_ssubset_image` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.image_subset_image_iff` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.subset_map_iff` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.map_subset_iff_subset_preimage` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.subset_image_iff` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset._root_.Multiset.toFinset_card_eq_one_iff` | Finset | OPEN_FLAKE | None |
| `Finset.image_subset_iff_subset_preimage` | Finset | CONFIRMED_RC2_FAILURE | False |
| `Finset.disjiUnion_cons` | Finset | RC2_SOLVED | True |
| `List.disjoint_pmap` | List | CONFIRMED_RC2_FAILURE | False |
| `List.map_subset_iff` | List | CONFIRMED_RC2_FAILURE | False |
| `List.disjoint_map` | List | CONFIRMED_RC2_FAILURE | False |
| `List.length_filter_lt_length_iff_exists` | List | CONFIRMED_RC2_FAILURE | False |
| `List.mem_map_of_involutive` | List | RC2_SOLVED | True |
| `List.mem_map_swap` | List | RC2_SOLVED | True |
| `List.mem_pmap` | List | CONFIRMED_RC2_FAILURE | False |
| `List.mem_map_of_injective` | List | CONFIRMED_RC2_FAILURE | False |
| `List.filterMap_eq_map_iff_forall_eq_some` | List | OPEN_FLAKE | None |
| `List.Forall.imp` | List | CONFIRMED_RC2_FAILURE | False |
| `List.pairwise_pmap` | List | CONFIRMED_RC2_FAILURE | False |
| `List.forall_iff_forall_mem` | List | CONFIRMED_RC2_FAILURE | False |
| `List.ranges_join'` | List | CONFIRMED_RC2_FAILURE | False |
| `List.subset_singleton_iff` | List | CONFIRMED_RC2_FAILURE | False |
| `List.indexOf_cons_ne` | List | CONFIRMED_RC2_FAILURE | False |
| `List.eq_replicate_length` | List | CONFIRMED_RC2_FAILURE | False |
| `List.ranges_disjoint` | List | CONFIRMED_RC2_FAILURE | False |
| `List.map_bijective_iff` | List | CONFIRMED_RC2_FAILURE | False |
| `List.map_injective_iff` | List | CONFIRMED_RC2_FAILURE | False |
| `List.map_involutive_iff` | List | CONFIRMED_RC2_FAILURE | False |
| `List.map_surjective_iff` | List | CONFIRMED_RC2_FAILURE | False |
| `List.pmap_eq_nil` | List | CONFIRMED_RC2_FAILURE | False |
| `List.forall_map_iff` | List | CONFIRMED_RC2_FAILURE | False |
| `List.map_leftInverse_iff` | List | CONFIRMED_RC2_FAILURE | False |
| `List.map_rightInverse_iff` | List | CONFIRMED_RC2_FAILURE | False |
| `List.forall_cons` | List | CONFIRMED_RC2_FAILURE | False |
| `List.mem_pair` | List | RC2_SOLVED | True |
| `List.mem_pure` | List | RC2_SOLVED | True |
| `List.mem_dedup` | List | CONFIRMED_RC2_FAILURE | False |
| `List.dropWhile_eq_nil_iff` | List | CONFIRMED_RC2_FAILURE | False |
| `List.takeWhile_eq_self_iff` | List | CONFIRMED_RC2_FAILURE | False |
| `List.indexOf_lt_length` | List | CONFIRMED_RC2_FAILURE | False |
| `List.indexOf_eq_length` | List | CONFIRMED_RC2_FAILURE | False |
| `List.mem_mem_ranges_iff_lt_natSum` | List | CONFIRMED_RC2_FAILURE | False |
| `List.exists_mem_cons_iff` | List | RC2_SOLVED | True |
| `Nat.le_of_pred_lt` | Nat | RC2_SOLVED | True |
| `Nat.le_mul_self` | Nat | CONFIRMED_RC2_FAILURE | False |
| `Nat.two_le_iff` | Nat | RC2_SOLVED | True |
| `Nat.lt_mul_self_iff` | Nat | CONFIRMED_RC2_FAILURE | False |
| `Nat.one_lt_iff_ne_zero_and_ne_one` | Nat | RC2_SOLVED | True |
| `Nat.lt_iff_le_pred` | Nat | RC2_SOLVED | True |
| `Nat.one_lt_pow_iff` | Nat | CONFIRMED_RC2_FAILURE | False |
| `Nat.sqrt_pos` | Nat | CONFIRMED_RC2_FAILURE | False |
| `Nat.le_sqrt` | Nat | CONFIRMED_RC2_FAILURE | False |
| `Nat.sqrt_lt` | Nat | RC2_SOLVED | True |
| `Nat.succ_le_iff` | Nat | RC2_SOLVED | True |
| `Nat.le_sqrt'` | Nat | CONFIRMED_RC2_FAILURE | False |
| `Nat.sqrt_eq_zero` | Nat | CONFIRMED_RC2_FAILURE | False |
| `Nat.sqrt_lt'` | Nat | CONFIRMED_RC2_FAILURE | False |
| `Nat.lt_one_add_iff` | Nat | RC2_SOLVED | True |
| `Nat.lt_one_iff` | Nat | RC2_SOLVED | True |
| `Nat.one_add_le_iff` | Nat | RC2_SOLVED | True |
| `Nat.one_le_iff_ne_zero` | Nat | RC2_SOLVED | True |
| `Nat.mul_self_inj` | Nat | CONFIRMED_RC2_FAILURE | False |
| `Nat.lt_iff_add_one_le` | Nat | RC2_SOLVED | True |
| `Nat.lt_pred_iff` | Nat | RC2_SOLVED | True |
| `Nat.pred_eq_self_iff` | Nat | RC2_SOLVED | True |
| `Nat.pred_le_iff` | Nat | RC2_SOLVED | True |
| `Nat.succ_ne_succ` | Nat | RC2_SOLVED | True |
| `Nat.modEq_zero_iff_dvd` | Nat | CONFIRMED_RC2_FAILURE | False |
| `Nat.mul_eq_left` | Nat | RC2_SOLVED | True |
| `Nat.mul_eq_right` | Nat | RC2_SOLVED | True |
| `Nat.le_add_one_iff` | Nat | RC2_SOLVED | True |
| `Nat.modEq_zero_iff` | Nat | CONFIRMED_RC2_FAILURE | False |
| `Nat.mod_eq_iff_lt` | Nat | CONFIRMED_RC2_FAILURE | False |
| `Nat.div_lt_one_iff` | Nat | RC2_SOLVED | True |
| `Nat.mod_two_ne_one` | Nat | RC2_SOLVED | True |
| `Nat.one_le_div_iff` | Nat | CONFIRMED_RC2_FAILURE | False |
| `Nat.ModEq.comm` | Nat | CONFIRMED_RC2_FAILURE | False |
| `Nat.add_eq_max_iff` | Nat | RC2_SOLVED | True |
| `Multiset.nodup_bind` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.disjoint_map_map` | Multiset | RC2_SOLVED | True |
| `Multiset.zero_disjoint` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.Disjoint.symm` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.disjoint_left` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.disjoint_right` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.disjoint_iff_ne` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.disjoint_singleton` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.singleton_disjoint` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.inter_eq_zero_iff_disjoint` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.add_eq_union_iff_disjoint` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.disjoint_cons_left` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.disjoint_cons_right` | Multiset | RC2_SOLVED | True |
| `Multiset.add_eq_union_left_of_le` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.add_eq_union_right_of_le` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.disjoint_union_left` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.disjoint_union_right` | Multiset | RC2_SOLVED | True |
| `Multiset.disjoint_of_subset_right` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.disjoint_of_subset_left` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.disjoint_comm` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.coe_disjoint` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.disjoint_add_left` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.disjoint_add_right` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.card_filter_le_iff` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `Multiset.toFinset_card_eq_card_iff_nodup` | Multiset | CONFIRMED_RC2_FAILURE | False |
| `mem_lowerBounds_iff_subset_Ici` |  | CONFIRMED_RC2_FAILURE | False |
| `mem_upperBounds_iff_subset_Iic` |  | RC2_SOLVED | True |
| `isGreatest_union_iff` |  | CONFIRMED_RC2_FAILURE | False |
| `isLeast_union_iff` |  | CONFIRMED_RC2_FAILURE | False |
| `bddAbove_iff_subset_Iic` |  | CONFIRMED_RC2_FAILURE | False |
| `bddBelow_iff_subset_Ici` |  | CONFIRMED_RC2_FAILURE | False |
| `bddBelow_bddAbove_iff_subset_Icc` |  | CONFIRMED_RC2_FAILURE | False |
| `AntitoneOn.image_lowerBounds_subset_upperBounds_image` | AntitoneOn | CONFIRMED_RC2_FAILURE | False |
| `AntitoneOn.image_upperBounds_subset_lowerBounds_image` | AntitoneOn | CONFIRMED_RC2_FAILURE | False |
| `MonotoneOn.image_lowerBounds_subset_lowerBounds_image` | MonotoneOn | CONFIRMED_RC2_FAILURE | False |
| `MonotoneOn.image_upperBounds_subset_upperBounds_image` | MonotoneOn | CONFIRMED_RC2_FAILURE | False |
| `bddAbove_preimage_ofDual` |  | RC2_SOLVED | True |
| `bddBelow_preimage_ofDual` |  | RC2_SOLVED | True |
| `bddAbove_preimage_toDual` |  | RC2_SOLVED | True |
| `bddBelow_preimage_toDual` |  | RC2_SOLVED | True |
| `IsGLB.of_image` | IsGLB | CONFIRMED_RC2_FAILURE | False |
| `IsLUB.of_image` | IsLUB | CONFIRMED_RC2_FAILURE | False |
| `bddAbove_def` |  | CONFIRMED_RC2_FAILURE | False |
| `bddBelow_def` |  | CONFIRMED_RC2_FAILURE | False |
| `mem_lowerBounds` |  | RC2_SOLVED | True |
| `Option.mem_map` | Option | RC2_SOLVED | True |
| `Option.exists_mem_map` | Option | RC2_SOLVED | True |
| `Option.forall_mem_map` | Option | RC2_SOLVED | True |
| `Option.pbind_eq_some` | Option | RC2_SOLVED | True |
| `Option.pbind_eq_none` | Option | RC2_SOLVED | True |
| `Option.mem_map_of_injective` | Option | RC2_SOLVED | True |
| `exists_of_exists_mem` |  | CONFIRMED_RC2_FAILURE | False |
| `exists_sUnion` |  | RC2_SOLVED | True |
| `forall_sUnion` |  | RC2_SOLVED | True |
| `exists_mem_of_exists` |  | RC2_SOLVED | True |
| `Option.pmap_eq_none_iff` | Option | RC2_SOLVED | True |
| `Option.map_eq_id` | Option | CONFIRMED_RC2_FAILURE | False |
| `Option.map_inj` | Option | CONFIRMED_RC2_FAILURE | False |
| `Option.pmap_eq_some_iff` | Option | RC2_SOLVED | True |
| `Option.bind_eq_some'` | Option | CONFIRMED_RC2_FAILURE | False |
| `Exists.snd` | Exists | RC2_SOLVED | True |
| `PLift.down_injective` | PLift | CONFIRMED_RC2_FAILURE | False |
| `not_exists_mem` |  | RC2_SOLVED | True |
| `Equiv.bijOn` | Equiv | CONFIRMED_RC2_FAILURE | False |
| `exists_mem_or` |  | CONFIRMED_RC2_FAILURE | False |
