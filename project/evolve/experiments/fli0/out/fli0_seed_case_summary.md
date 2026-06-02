# FLI0 seed-case selection summary

- **seeds selected: 40** (target 40) from pool 217
- by pattern: {'SUBSET_BRIDGE': 8, 'MAP_FILTER_BIND_BRIDGE': 8, 'MEMBERSHIP_BRIDGE': 7, 'INDUCTION_GENERALIZATION': 6, 'SINGLETON_CHARACTERIZATION': 4, 'IFF_SPLIT': 4, 'DISJOINT_BRIDGE': 2, 'EXTENSIONALITY_NEEDED': 1}
- by namespace: {'Finset': 14, 'List': 14, 'Multiset': 4, 'Set': 4, 'Nat': 4}
- by source stage: {'RC5V3': 19, 'RC5V2': 21} | confidence: {'high': 20, 'medium': 20}
- recommended FLI1 actions: {'generate_subset_bridge': 8, 'generate_map_filter_bind_membership': 8, 'generate_membership_bridge': 7, 'generate_induction_helper': 6, 'generate_singleton_iff': 4, 'generate_iff_split_helper': 4, 'generate_disjoint_membership_bridge': 2, 'generate_ext_membership_bridge': 1}

## Seeds

| id | theorem | ns | pattern | conf | action |
|---|---|---|---|---|---|
| FLI0-S01 | `Finset.biUnion_nonempty` | Finset | MEMBERSHIP_BRIDGE | high | generate_membership_bridge |
| FLI0-S02 | `Finset.card_le_card` | Finset | SUBSET_BRIDGE | high | generate_subset_bridge |
| FLI0-S03 | `Finset.card_le_one` | Finset | MEMBERSHIP_BRIDGE | high | generate_membership_bridge |
| FLI0-S04 | `Finset.card_le_one_iff` | Finset | MEMBERSHIP_BRIDGE | high | generate_membership_bridge |
| FLI0-S05 | `Finset.card_le_one_iff_subsingleton_coe` | Finset | SINGLETON_CHARACTERIZATION | high | generate_singleton_iff |
| FLI0-S06 | `Finset.card_le_one_of_subsingleton` | Finset | SINGLETON_CHARACTERIZATION | high | generate_singleton_iff |
| FLI0-S07 | `Finset.card_singleton_inter` | Finset | SINGLETON_CHARACTERIZATION | high | generate_singleton_iff |
| FLI0-S08 | `Finset.disjoint_range_addLeftEmbedding` | Finset | DISJOINT_BRIDGE | high | generate_disjoint_membership_bridge |
| FLI0-S09 | `Finset.disjoint_range_addRightEmbedding` | Finset | DISJOINT_BRIDGE | high | generate_disjoint_membership_bridge |
| FLI0-S10 | `Finset.eq_of_subset_of_card_le` | Finset | SUBSET_BRIDGE | high | generate_subset_bridge |
| FLI0-S11 | `Finset.exists_of_one_lt_card_pi` | Finset | SUBSET_BRIDGE | high | generate_subset_bridge |
| FLI0-S12 | `Finset.exists_subset_card_eq` | Finset | SUBSET_BRIDGE | high | generate_subset_bridge |
| FLI0-S13 | `Finset.filter_attach'` | Finset | SUBSET_BRIDGE | high | generate_subset_bridge |
| FLI0-S14 | `Finset.map_eq_of_subset` | Finset | SUBSET_BRIDGE | high | generate_subset_bridge |
| FLI0-S15 | `Multiset.forall_mem_map_iff` | Multiset | MEMBERSHIP_BRIDGE | high | generate_membership_bridge |
| FLI0-S16 | `Multiset.mem_filter` | Multiset | MEMBERSHIP_BRIDGE | high | generate_membership_bridge |
| FLI0-S17 | `Multiset.mem_filterMap` | Multiset | MEMBERSHIP_BRIDGE | high | generate_membership_bridge |
| FLI0-S18 | `Multiset.one_le_count_iff_mem` | Multiset | MEMBERSHIP_BRIDGE | high | generate_membership_bridge |
| FLI0-S19 | `Set.InjOn.image_diff_subset` | Set | SUBSET_BRIDGE | high | generate_subset_bridge |
| FLI0-S20 | `Set.InjOn.image_inter` | Set | SUBSET_BRIDGE | high | generate_subset_bridge |
| FLI0-S21 | `List.attach_map_coe'` | List | MAP_FILTER_BIND_BRIDGE | medium | generate_map_filter_bind_membership |
| FLI0-S22 | `List.attach_map_val` | List | MAP_FILTER_BIND_BRIDGE | medium | generate_map_filter_bind_membership |
| FLI0-S23 | `List.bind_congr` | List | MAP_FILTER_BIND_BRIDGE | medium | generate_map_filter_bind_membership |
| FLI0-S24 | `List.bind_eq_bind` | List | MAP_FILTER_BIND_BRIDGE | medium | generate_map_filter_bind_membership |
| FLI0-S25 | `List.bind_pure_eq_map` | List | MAP_FILTER_BIND_BRIDGE | medium | generate_map_filter_bind_membership |
| FLI0-S26 | `List.bind_ret_eq_map` | List | MAP_FILTER_BIND_BRIDGE | medium | generate_map_filter_bind_membership |
| FLI0-S27 | `List.count_map_of_injective` | List | MAP_FILTER_BIND_BRIDGE | medium | generate_map_filter_bind_membership |
| FLI0-S28 | `List.dedup_append` | List | INDUCTION_GENERALIZATION | medium | generate_induction_helper |
| FLI0-S29 | `List.dedup_cons_of_mem` | List | INDUCTION_GENERALIZATION | medium | generate_induction_helper |
| FLI0-S30 | `List.dedup_cons_of_mem'` | List | INDUCTION_GENERALIZATION | medium | generate_induction_helper |
| FLI0-S31 | `List.dedup_cons_of_not_mem` | List | INDUCTION_GENERALIZATION | medium | generate_induction_helper |
| FLI0-S32 | `List.dedup_cons_of_not_mem'` | List | INDUCTION_GENERALIZATION | medium | generate_induction_helper |
| FLI0-S33 | `List.filterMap_congr` | List | MAP_FILTER_BIND_BRIDGE | medium | generate_map_filter_bind_membership |
| FLI0-S34 | `List.getLast_append'` | List | INDUCTION_GENERALIZATION | medium | generate_induction_helper |
| FLI0-S35 | `Set.inter_empty_of_inter_sUnion_empty` | Set | EXTENSIONALITY_NEEDED | medium | generate_ext_membership_bridge |
| FLI0-S36 | `Set.sInter_diff_singleton_univ` | Set | SINGLETON_CHARACTERIZATION | medium | generate_singleton_iff |
| FLI0-S37 | `Nat.coprime_add_mul_left_left` | Nat | IFF_SPLIT | medium | generate_iff_split_helper |
| FLI0-S38 | `Nat.coprime_add_mul_left_right` | Nat | IFF_SPLIT | medium | generate_iff_split_helper |
| FLI0-S39 | `Nat.coprime_add_mul_right_left` | Nat | IFF_SPLIT | medium | generate_iff_split_helper |
| FLI0-S40 | `Nat.coprime_add_mul_right_right` | Nat | IFF_SPLIT | medium | generate_iff_split_helper |
