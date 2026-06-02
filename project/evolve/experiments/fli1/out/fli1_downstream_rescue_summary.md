# FLI1 downstream rescue summary

- tested: 40 | **DOWNSTREAM_RESCUE: 1** (robust 1) | partial: 0 | direct-solve-dup: 1
- histogram: {'NO_RESCUE': 38, 'DOWNSTREAM_RESCUE': 1, 'DIRECT_SOLVE_DUPLICATE': 1}

| candidate | theorem | class | rescue tactic | robust |
|---|---|---|---|---|
| FLI1-L01 | `Finset.biUnion_nonempty` | NO_RESCUE | `` | None |
| FLI1-L02 | `Finset.card_le_card` | NO_RESCUE | `` | None |
| FLI1-L03 | `Finset.card_le_one` | NO_RESCUE | `` | None |
| FLI1-L04 | `Finset.card_le_one_iff` | DOWNSTREAM_RESCUE | `simp [Finset.card_le_one] <;> aesop` | True |
| FLI1-L05 | `Finset.disjoint_range_addLeftEmbedding` | NO_RESCUE | `` | None |
| FLI1-L06 | `Finset.disjoint_range_addRightEmbedding` | NO_RESCUE | `` | None |
| FLI1-L07 | `Finset.eq_of_subset_of_card_le` | NO_RESCUE | `` | None |
| FLI1-L08 | `Finset.exists_of_one_lt_card_pi` | NO_RESCUE | `` | None |
| FLI1-L09 | `Finset.exists_subset_card_eq` | NO_RESCUE | `` | None |
| FLI1-L10 | `Finset.filter_attach'` | NO_RESCUE | `` | None |
| FLI1-L11 | `Finset.map_eq_of_subset` | NO_RESCUE | `` | None |
| FLI1-L12 | `List.attach_map_coe'` | NO_RESCUE | `` | None |
| FLI1-L13 | `List.attach_map_val` | NO_RESCUE | `` | None |
| FLI1-L14 | `List.bind_congr` | NO_RESCUE | `` | None |
| FLI1-L15 | `List.bind_eq_bind` | DIRECT_SOLVE_DUPLICATE | `constructor <;> simp [List.bind_pure_eq_map]` | None |
| FLI1-L16 | `List.bind_pure_eq_map` | NO_RESCUE | `` | None |
| FLI1-L17 | `List.bind_ret_eq_map` | NO_RESCUE | `` | None |
| FLI1-L18 | `List.count_map_of_injective` | NO_RESCUE | `` | None |
| FLI1-L19 | `List.filterMap_congr` | NO_RESCUE | `` | None |
| FLI1-L20 | `Multiset.forall_mem_map_iff` | NO_RESCUE | `` | None |
| FLI1-L21 | `Multiset.mem_filter` | NO_RESCUE | `` | None |
| FLI1-L22 | `Multiset.mem_filterMap` | NO_RESCUE | `` | None |
| FLI1-L23 | `Multiset.one_le_count_iff_mem` | NO_RESCUE | `` | None |
| FLI1-L24 | `Nat.coprime_add_mul_left_left` | NO_RESCUE | `` | None |
| FLI1-L25 | `Nat.coprime_add_mul_left_right` | NO_RESCUE | `` | None |
| FLI1-L26 | `Nat.coprime_add_mul_right_left` | NO_RESCUE | `` | None |
| FLI1-L27 | `Nat.coprime_add_mul_right_right` | NO_RESCUE | `` | None |
| FLI1-L28 | `Set.InjOn.image_diff_subset` | NO_RESCUE | `` | None |
| FLI1-L29 | `Set.InjOn.image_inter` | NO_RESCUE | `` | None |
| FLI1-L30 | `Finset.card_le_one_iff_subsingleton_coe` | NO_RESCUE | `` | None |
| FLI1-L31 | `Finset.card_le_one_of_subsingleton` | NO_RESCUE | `` | None |
| FLI1-L32 | `Finset.card_singleton_inter` | NO_RESCUE | `` | None |
| FLI1-L33 | `Set.inter_empty_of_inter_sUnion_empty` | NO_RESCUE | `` | None |
| FLI1-L34 | `Set.sInter_diff_singleton_univ` | NO_RESCUE | `` | None |
| FLI1-L35 | `List.dedup_append` | NO_RESCUE | `` | None |
| FLI1-L36 | `List.dedup_cons_of_mem` | NO_RESCUE | `` | None |
| FLI1-L37 | `List.dedup_cons_of_mem'` | NO_RESCUE | `` | None |
| FLI1-L38 | `List.dedup_cons_of_not_mem` | NO_RESCUE | `` | None |
| FLI1-L39 | `List.dedup_cons_of_not_mem'` | NO_RESCUE | `` | None |
| FLI1-L40 | `List.getLast_append'` | NO_RESCUE | `` | None |
