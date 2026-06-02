# FLI1 existing-lemma check summary

- candidates: 40 | classes: {'PROBABLY_NEW': 21, 'EXISTS_CLOSE': 15, 'TOO_VAGUE_TO_CHECK': 4}
- **retrieval gaps (bridge exists, search didn't use): 15**
- probably new: 21

| id | seed | class | closest existing | score | retr-gap |
|---|---|---|---|---|---|
| FLI1-L01 | FLI0-S01 | PROBABLY_NEW | `Finset.disjoint_biUnion_left` | 0.308 | False |
| FLI1-L02 | FLI0-S02 | PROBABLY_NEW | `Finset.eq_of_subset_of_card_le` | 0.4 | False |
| FLI1-L03 | FLI0-S03 | EXISTS_CLOSE | `Finset.card_le_one_iff` | 0.5 | True |
| FLI1-L04 | FLI0-S04 | EXISTS_CLOSE | `Finset.card_le_one` | 0.455 | True |
| FLI1-L05 | FLI0-S08 | PROBABLY_NEW | `Finset.range` | 0.167 | False |
| FLI1-L06 | FLI0-S09 | PROBABLY_NEW | `Finset.range` | 0.167 | False |
| FLI1-L07 | FLI0-S10 | EXISTS_CLOSE | `Finset.card_le_card` | 0.5 | True |
| FLI1-L08 | FLI0-S11 | PROBABLY_NEW | `Finset.one_lt_card` | 0.133 | False |
| FLI1-L09 | FLI0-S12 | EXISTS_CLOSE | `Finset.exists_smaller_set` | 0.5 | True |
| FLI1-L10 | FLI0-S13 | PROBABLY_NEW | `Finset.filter_attach` | 0.381 | False |
| FLI1-L11 | FLI0-S14 | PROBABLY_NEW | `Finset.map_perm` | 0.4 | False |
| FLI1-L12 | FLI0-S21 | PROBABLY_NEW | `List.pmap_eq_map_attach` | 0.4 | False |
| FLI1-L13 | FLI0-S22 | EXISTS_CLOSE | `Multiset.attach_map_val` | 0.625 | True |
| FLI1-L14 | FLI0-S23 | EXISTS_CLOSE | `List.bind_eq_nil` | 0.6 | True |
| FLI1-L15 | FLI0-S24 | EXISTS_CLOSE | `List.bind_pure_eq_map` | 0.5 | True |
| FLI1-L16 | FLI0-S25 | EXISTS_CLOSE | `List.bind_ret_eq_map` | 0.556 | True |
| FLI1-L17 | FLI0-S26 | EXISTS_CLOSE | `List.bind_pure_eq_map` | 0.556 | True |
| FLI1-L18 | FLI0-S27 | EXISTS_CLOSE | `List.disjoint_map` | 0.5 | True |
| FLI1-L19 | FLI0-S33 | EXISTS_CLOSE | `List.Nodup.filterMap` | 0.545 | True |
| FLI1-L20 | FLI0-S15 | EXISTS_CLOSE | `Multiset.forall_mem_cons` | 0.455 | True |
| FLI1-L21 | FLI0-S16 | PROBABLY_NEW | `Multiset.filter` | 0.286 | False |
| FLI1-L22 | FLI0-S17 | EXISTS_CLOSE | `Multiset.filterMap_cons_some` | 0.5 | True |
| FLI1-L23 | FLI0-S18 | PROBABLY_NEW | `Multiset.nodup_iff_count_le_one` | 0.308 | False |
| FLI1-L24 | FLI0-S37 | TOO_VAGUE_TO_CHECK | `Nat.coprime_mul_left_add_left` | 0.143 | False |
| FLI1-L25 | FLI0-S38 | TOO_VAGUE_TO_CHECK | `Nat.coprime_add_mul_right_right` | 0.143 | False |
| FLI1-L26 | FLI0-S39 | TOO_VAGUE_TO_CHECK | `Nat.coprime_add_mul_right_right` | 0.143 | False |
| FLI1-L27 | FLI0-S40 | TOO_VAGUE_TO_CHECK | `Nat.coprime_mul_right_add_right` | 0.143 | False |
| FLI1-L28 | FLI0-S19 | PROBABLY_NEW | `Set.diff_subset_diff` | 0.25 | False |
| FLI1-L29 | FLI0-S20 | PROBABLY_NEW | `Set.InjOn.preimage_image_inter` | 0.333 | False |
| FLI1-L30 | FLI0-S05 | PROBABLY_NEW | `Finset.card_le_one_of_subsingleton` | 0.364 | False |
| FLI1-L31 | FLI0-S06 | PROBABLY_NEW | `Finset.card_le_one_iff` | 0.308 | False |
| FLI1-L32 | FLI0-S07 | PROBABLY_NEW | `Finset.card_singleton` | 0.286 | False |
| FLI1-L33 | FLI0-S35 | PROBABLY_NEW | `Set.sUnion_inter_sUnion` | 0.333 | False |
| FLI1-L34 | FLI0-S36 | PROBABLY_NEW | `Set.sInter_eq_univ` | 0.273 | False |
| FLI1-L35 | FLI0-S28 | EXISTS_CLOSE | `List.dedup_idem` | 0.5 | True |
| FLI1-L36 | FLI0-S29 | PROBABLY_NEW | `List.dedup_cons_of_not_mem` | 0.3 | False |
| FLI1-L37 | FLI0-S30 | PROBABLY_NEW | `List.dedup_cons_of_mem` | 0.444 | False |
| FLI1-L38 | FLI0-S31 | PROBABLY_NEW | `List.dedup_cons_of_mem'` | 0.333 | False |
| FLI1-L39 | FLI0-S32 | EXISTS_CLOSE | `List.dedup` | 0.5 | True |
| FLI1-L40 | FLI0-S34 | PROBABLY_NEW | `List.getLast_cons_cons` | 0.429 | False |
