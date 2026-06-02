# FLI1 live rerun plan

- seeds: 40 | by priority: {'high': 29, 'medium': 5, 'low': 6}
- seeds missing file_path: 0 []

| seed | theorem | ns | pattern | prio | #probes |
|---|---|---|---|---|---|
| FLI0-S01 | `Finset.biUnion_nonempty` | Finset | MEMBERSHIP_BRIDGE | high | 4 |
| FLI0-S02 | `Finset.card_le_card` | Finset | SUBSET_BRIDGE | high | 4 |
| FLI0-S03 | `Finset.card_le_one` | Finset | MEMBERSHIP_BRIDGE | high | 4 |
| FLI0-S04 | `Finset.card_le_one_iff` | Finset | MEMBERSHIP_BRIDGE | high | 4 |
| FLI0-S08 | `Finset.disjoint_range_addLeftEmbedding` | Finset | DISJOINT_BRIDGE | high | 4 |
| FLI0-S09 | `Finset.disjoint_range_addRightEmbedding` | Finset | DISJOINT_BRIDGE | high | 4 |
| FLI0-S10 | `Finset.eq_of_subset_of_card_le` | Finset | SUBSET_BRIDGE | high | 4 |
| FLI0-S11 | `Finset.exists_of_one_lt_card_pi` | Finset | SUBSET_BRIDGE | high | 4 |
| FLI0-S12 | `Finset.exists_subset_card_eq` | Finset | SUBSET_BRIDGE | high | 4 |
| FLI0-S13 | `Finset.filter_attach'` | Finset | SUBSET_BRIDGE | high | 4 |
| FLI0-S14 | `Finset.map_eq_of_subset` | Finset | SUBSET_BRIDGE | high | 4 |
| FLI0-S21 | `List.attach_map_coe'` | List | MAP_FILTER_BIND_BRIDGE | high | 5 |
| FLI0-S22 | `List.attach_map_val` | List | MAP_FILTER_BIND_BRIDGE | high | 5 |
| FLI0-S23 | `List.bind_congr` | List | MAP_FILTER_BIND_BRIDGE | high | 4 |
| FLI0-S24 | `List.bind_eq_bind` | List | MAP_FILTER_BIND_BRIDGE | high | 5 |
| FLI0-S25 | `List.bind_pure_eq_map` | List | MAP_FILTER_BIND_BRIDGE | high | 5 |
| FLI0-S26 | `List.bind_ret_eq_map` | List | MAP_FILTER_BIND_BRIDGE | high | 5 |
| FLI0-S27 | `List.count_map_of_injective` | List | MAP_FILTER_BIND_BRIDGE | high | 5 |
| FLI0-S33 | `List.filterMap_congr` | List | MAP_FILTER_BIND_BRIDGE | high | 5 |
| FLI0-S15 | `Multiset.forall_mem_map_iff` | Multiset | MEMBERSHIP_BRIDGE | high | 4 |
| FLI0-S16 | `Multiset.mem_filter` | Multiset | MEMBERSHIP_BRIDGE | high | 4 |
| FLI0-S17 | `Multiset.mem_filterMap` | Multiset | MEMBERSHIP_BRIDGE | high | 4 |
| FLI0-S18 | `Multiset.one_le_count_iff_mem` | Multiset | MEMBERSHIP_BRIDGE | high | 4 |
| FLI0-S37 | `Nat.coprime_add_mul_left_left` | Nat | IFF_SPLIT | high | 4 |
| FLI0-S38 | `Nat.coprime_add_mul_left_right` | Nat | IFF_SPLIT | high | 4 |
| FLI0-S39 | `Nat.coprime_add_mul_right_left` | Nat | IFF_SPLIT | high | 4 |
| FLI0-S40 | `Nat.coprime_add_mul_right_right` | Nat | IFF_SPLIT | high | 4 |
| FLI0-S19 | `Set.InjOn.image_diff_subset` | Set | SUBSET_BRIDGE | high | 4 |
| FLI0-S20 | `Set.InjOn.image_inter` | Set | SUBSET_BRIDGE | high | 4 |
| FLI0-S05 | `Finset.card_le_one_iff_subsingleton_coe` | Finset | SINGLETON_CHARACTERIZATION | medium | 4 |
| FLI0-S06 | `Finset.card_le_one_of_subsingleton` | Finset | SINGLETON_CHARACTERIZATION | medium | 4 |
| FLI0-S07 | `Finset.card_singleton_inter` | Finset | SINGLETON_CHARACTERIZATION | medium | 4 |
| FLI0-S35 | `Set.inter_empty_of_inter_sUnion_empty` | Set | EXTENSIONALITY_NEEDED | medium | 4 |
| FLI0-S36 | `Set.sInter_diff_singleton_univ` | Set | SINGLETON_CHARACTERIZATION | medium | 4 |
| FLI0-S28 | `List.dedup_append` | List | INDUCTION_GENERALIZATION | low | 3 |
| FLI0-S29 | `List.dedup_cons_of_mem` | List | INDUCTION_GENERALIZATION | low | 3 |
| FLI0-S30 | `List.dedup_cons_of_mem'` | List | INDUCTION_GENERALIZATION | low | 3 |
| FLI0-S31 | `List.dedup_cons_of_not_mem` | List | INDUCTION_GENERALIZATION | low | 3 |
| FLI0-S32 | `List.dedup_cons_of_not_mem'` | List | INDUCTION_GENERALIZATION | low | 3 |
| FLI0-S34 | `List.getLast_append'` | List | INDUCTION_GENERALIZATION | low | 3 |
