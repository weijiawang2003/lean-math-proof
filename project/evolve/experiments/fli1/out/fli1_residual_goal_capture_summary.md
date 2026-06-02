# FLI1 residual goal capture summary

- seeds: 40 | **captured: 40** (high quality 19) | target ≥25: True
- status: {'captured': 40}
- solved_directly (NOT FLI1 success): 0
- captured by namespace: {'Finset': 14, 'List': 14, 'Multiset': 4, 'Nat': 4, 'Set': 4}

| seed | theorem | status | quality | prefix | #goals |
|---|---|---|---|---|---|
| FLI0-S01 | `Finset.biUnion_nonempty` | captured | high | `constructor ; intro h` | 2 |
| FLI0-S02 | `Finset.card_le_card` | captured | high | `intro h` | 1 |
| FLI0-S03 | `Finset.card_le_one` | captured | high | `constructor ; intro h` | 2 |
| FLI0-S04 | `Finset.card_le_one_iff` | captured | high | `constructor ; intro h` | 2 |
| FLI0-S08 | `Finset.disjoint_range_addLeftEmbedding` | captured | high | `simp [Finset.disjoint_left]` | 1 |
| FLI0-S09 | `Finset.disjoint_range_addRightEmbedding` | captured | high | `simp [Finset.disjoint_left]` | 1 |
| FLI0-S10 | `Finset.eq_of_subset_of_card_le` | captured | medium | `` |  |
| FLI0-S11 | `Finset.exists_of_one_lt_card_pi` | captured | medium | `` |  |
| FLI0-S12 | `Finset.exists_subset_card_eq` | captured | medium | `` |  |
| FLI0-S13 | `Finset.filter_attach'` | captured | medium | `` |  |
| FLI0-S14 | `Finset.map_eq_of_subset` | captured | medium | `` |  |
| FLI0-S21 | `List.attach_map_coe'` | captured | medium | `` |  |
| FLI0-S22 | `List.attach_map_val` | captured | medium | `` |  |
| FLI0-S23 | `List.bind_congr` | captured | medium | `` |  |
| FLI0-S24 | `List.bind_eq_bind` | captured | medium | `` |  |
| FLI0-S25 | `List.bind_pure_eq_map` | captured | medium | `` |  |
| FLI0-S26 | `List.bind_ret_eq_map` | captured | medium | `` |  |
| FLI0-S27 | `List.count_map_of_injective` | captured | medium | `` |  |
| FLI0-S33 | `List.filterMap_congr` | captured | medium | `` |  |
| FLI0-S15 | `Multiset.forall_mem_map_iff` | captured | high | `constructor ; intro h` | 2 |
| FLI0-S16 | `Multiset.mem_filter` | captured | high | `constructor ; intro h` | 2 |
| FLI0-S17 | `Multiset.mem_filterMap` | captured | high | `constructor ; intro h` | 2 |
| FLI0-S18 | `Multiset.one_le_count_iff_mem` | captured | high | `constructor ; intro h` | 2 |
| FLI0-S37 | `Nat.coprime_add_mul_left_left` | captured | high | `constructor ; intro h` | 2 |
| FLI0-S38 | `Nat.coprime_add_mul_left_right` | captured | high | `constructor ; intro h` | 2 |
| FLI0-S39 | `Nat.coprime_add_mul_right_left` | captured | high | `constructor ; intro h` | 2 |
| FLI0-S40 | `Nat.coprime_add_mul_right_right` | captured | high | `constructor ; intro h` | 2 |
| FLI0-S19 | `Set.InjOn.image_diff_subset` | captured | medium | `` |  |
| FLI0-S20 | `Set.InjOn.image_inter` | captured | medium | `` |  |
| FLI0-S05 | `Finset.card_le_one_iff_subsingleton_coe` | captured | high | `constructor ; intro h` | 2 |
| FLI0-S06 | `Finset.card_le_one_of_subsingleton` | captured | high | `simp [Finset.card_le_one_iff]` | 1 |
| FLI0-S07 | `Finset.card_singleton_inter` | captured | high | `constructor` | 1 |
| FLI0-S35 | `Set.inter_empty_of_inter_sUnion_empty` | captured | high | `ext x ; simp` | 1 |
| FLI0-S36 | `Set.sInter_diff_singleton_univ` | captured | medium | `` |  |
| FLI0-S28 | `List.dedup_append` | captured | medium | `` |  |
| FLI0-S29 | `List.dedup_cons_of_mem` | captured | high | `induction h` | 2 |
| FLI0-S30 | `List.dedup_cons_of_mem'` | captured | medium | `` |  |
| FLI0-S31 | `List.dedup_cons_of_not_mem` | captured | medium | `` |  |
| FLI0-S32 | `List.dedup_cons_of_not_mem'` | captured | medium | `` |  |
| FLI0-S34 | `List.getLast_append'` | captured | medium | `` |  |
