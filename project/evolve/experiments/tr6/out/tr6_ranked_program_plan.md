# TR6 ranked program plan

- ranker: hgb | theorems: 137 | scored 7090 programs
- family histogram: {'d1_simp_lemma': 783, 'd2_simp_aesop': 417, 'd2_simp_simpall': 368, 'd3_simp_try': 167, 'd2_rw_aesop': 192, 'd1_aesop': 97, 'd3_constructor_aesop': 39, 'd1_rw_lemma': 95, 'def_unfold_simp': 76, 'd1_exact': 183, 'd1_simpa_using': 151, 'd1_simpa_lemma': 117, 'd2_rw_simpall': 3, 'd3_ext_simp_aesop': 7, 'd2_ext_aesop': 1, 'd2_ext_simp': 8, 'd1_simp_all': 8, 'd1_tauto': 7, 'd2_constructor_simpa': 2, 'd1_tofinset_simp': 15, 'd3_constructor_simp_all': 4}
- programs per budget: {1: 137, 3: 411, 5: 685, 10: 1370, 20: 2740}

## rank-1 program per theorem (first 25)

| theorem | ns | rank1 family | score | tactic |
|---|---|---|---|---|
| `Set.disjoint_sUnion_left` | Set | d1_simp_lemma | 0.924964 | `simp [Set.disjoint_left]` |
| `Set.disjoint_sUnion_right` | Set | d2_simp_aesop | 0.971078 | `simp [Set.disjoint_sUnion_left] <;> aesop` |
| `Set.injOn_union` | Set | d2_simp_aesop | 0.004986 | `simp [Set.disjoint_left] <;> aesop` |
| `Set.disjoint_iUnion_left` | Set | d2_simp_aesop | 0.844896 | `simp [Set.disjoint_sUnion_left] <;> aesop` |
| `Set.disjoint_iUnion_right` | Set | d2_simp_aesop | 0.919415 | `simp [Set.disjoint_iUnion_left] <;> aesop` |
| `Set.InjOn.mem_image_iff` | Set | d2_simp_aesop | 0.002498 | `simp [Set.InjOn.image_subset_image_iff] <;> a` |
| `Set.mapsTo_sInter` | Set | d2_simp_aesop | 0.001289 | `simp [Set.sInter_eq_univ] <;> aesop` |
| `Set.mapsTo_sUnion` | Set | d1_simp_lemma | 0.019048 | `simp [Set.disjoint_sUnion_right]` |
| `Set.mapsTo'` | Set | d2_simp_aesop | 0.000892 | `simp [Set.mapsTo_image_iff] <;> aesop` |
| `Set.kernImage_preimage_eq_iff` | Set | d2_simp_aesop | 0.024077 | `simp [Set.kernImage_eq_compl] <;> aesop` |
| `Set.InjOn.image_eq_image_iff` | Set | d2_simp_aesop | 0.024086 | `simp [Set.InjOn.image_subset_image_iff] <;> a` |
| `Set.InjOn.image_subset_image_iff` | Set | d2_simp_aesop | 0.023223 | `simp [Set.image_subset_image_iff] <;> aesop` |
| `Set.InjOn.image_ssubset_image_iff` | Set | d1_simp_lemma | 0.002666 | `simp [Set.InjOn.image_subset_image_iff]` |
| `Set.surjOn_iff_exists_map_subtype` | Set | d1_simp_lemma | 0.051553 | `simp [Set.surjective_iff_surjOn_univ]` |
| `Set.biUnion_compl_eq_of_pairwise_disjoint_of_iUnion_eq_univ` | Set | d2_simp_aesop | 0.10772 | `simp [Set.disjoint_left] <;> aesop` |
| `Set.mapsTo_singleton` | Set | d1_simp_lemma | 0.000788 | `simp [Set.mapsTo_univ_iff]` |
| `Set.mapsTo_inter` | Set | d2_simp_aesop | 0.002092 | `simp [Set.MapsTo.inter] <;> aesop` |
| `Set.mapsTo_union` | Set | d2_simp_aesop | 0.00181 | `simp [Set.mapsTo_iUnion] <;> aesop` |
| `Set.mapsTo_range_iff` | Set | d2_simp_aesop | 0.011503 | `simp [Set.maps_range_to] <;> aesop` |
| `Set.MapsTo.mem_iff` | Set | d2_simp_aesop | 0.03127 | `simp [Set.mapsTo_univ_iff] <;> aesop` |
| `Set.bijective_iff_bijective_of_iUnion_eq_univ` | Set | d2_simp_aesop | 0.001305 | `simp [Set.surjective_iff_surjective_of_iUnion` |
| `Finset.card_union_eq_card_add_card` | Finset | d2_simp_aesop | 0.034742 | `simp [Set.disjoint_left] <;> aesop` |
| `Finset.disjoint_biUnion_left` | Finset | d1_simp_lemma | 0.575044 | `simp [Finset.biUnion_nonempty]` |
| `Finset.disjoint_biUnion_right` | Finset | d2_simp_aesop | 0.582481 | `simp [Set.disjoint_left] <;> aesop` |
| `Finset.card_filter_le_iff` | Finset | d1_simp_lemma | 0.016179 | `simp [Multiset.card_filter_le_iff]` |
