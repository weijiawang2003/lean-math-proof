# FLI2 rescue attribution summary

- actions classified: 1059
- action classes: {'NO_RESCUE': 880, 'UNKNOWN_NAME_OR_IMPORT_GAP': 101, 'PARTIAL_PROGRESS': 64, 'TRUE_RETRIEVAL_GAP_RESCUE': 11, 'CONTROL_DUPLICATE': 2, 'NEEDS_REVIEW': 1}
- **per-theorem verdict: {'NO_RESCUE': 94, 'UNKNOWN_NAME_OR_IMPORT_GAP': 30, 'PARTIAL_PROGRESS': 30, 'TRUE_RETRIEVAL_GAP_RESCUE': 6, 'CONTROL_DUPLICATE': 1}**
- **TRUE_RETRIEVAL_GAP_RESCUE: 6 theorems (11 actions)**

## True rescues

| theorem | lemma | tactic |
|---|---|---|
| `Finset.card_le_one_iff` | `Finset.card_le_one` | `simp [Finset.card_le_one] <;> aesop` |
| `Finset.mem_filterMap` | `Finset.filterMap` | `simp [Finset.filterMap]` |
| `Finset.mem_filterMap` | `Finset.filterMap` | `simp [Finset.filterMap] <;> aesop` |
| `Finset.card_subtype` | `Finset.subtype` | `simp [Finset.subtype]` |
| `Finset.card_subtype` | `Finset.subtype` | `simp [Finset.subtype] <;> aesop` |
| `Finset.mem_map` | `Finset.map` | `simp [Finset.map]` |
| `Finset.mem_map` | `Finset.map` | `simp [Finset.map] <;> aesop` |
| `Finset.mem_preimage` | `Finset.preimage` | `simp [Finset.preimage]` |
| `Finset.mem_preimage` | `Finset.preimage` | `simp [Finset.preimage] <;> aesop` |
| `List.bidirectionalRec_singleton` | `List.bidirectionalRec` | `simp [List.bidirectionalRec]` |
| `List.bidirectionalRec_singleton` | `List.bidirectionalRec` | `simp [List.bidirectionalRec] <;> aesop` |

- partial-progress theorems: ['Finset.card_filter_le', 'Finset.card_le_one_iff_subsingleton_coe', 'Finset.card_le_one_of_subsingleton', 'Finset.image_val_of_injOn', 'Finset.map_val_val_powersetCard', 'Finset.max_mem_image_coe', 'Finset.max_mem_insert_bot_image_coe', 'Finset.mem_image_const_self', 'Finset.mem_range_iff_mem_finset_range_of_mod_eq', "Finset.mem_range_iff_mem_finset_range_of_mod_eq'", 'Finset.mem_ssubsets', 'Finset.min_mem_insert_top_image_coe', 'Finset.powerset_card_biUnion', 'List.filter_comm', 'List.getLast_filter', 'Multiset.bind_add', 'Multiset.bind_cons', 'Multiset.countP_eq_countP_filter_add', 'Multiset.count_map', 'Multiset.filterMap_some', 'Multiset.filter_add_not', 'Multiset.filter_attach', 'Multiset.map_filterMap_of_inv', 'Multiset.pmap_cons', 'Set.LeftInvOn.image_inter', 'Set.biUnion_diff_biUnion_subset', 'Set.iUnion_iInter_ge_nat_add', 'Set.iUnion_subset_iUnion_const', 'Set.kernImage_preimage_union', 'Set.kernImage_union_preimage']
- control-duplicate theorems: 1
