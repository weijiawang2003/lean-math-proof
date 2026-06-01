# TR6 ranked program plan

- ranker: hgb | theorems: 149 | scored 7021 programs
- family histogram: {'d1_simp_lemma': 776, 'd2_simp_aesop': 406, 'd2_simp_simpall': 372, 'def_unfold_simp': 92, 'd1_exact': 266, 'd1_simpa_using': 221, 'd1_simpa_lemma': 186, 'd2_rw_aesop': 157, 'd3_simp_try': 152, 'd1_rw_lemma': 114, 'd1_aesop': 93, 'd1_simp_all': 5, 'd1_tauto': 5, 'd3_ext_simp_aesop': 70, 'd2_ext_aesop': 2, 'd2_ext_simp': 58, 'd2_rw_simpall': 3, 'd1_tofinset_simp': 1, 'd3_constructor_aesop': 1}
- programs per budget: {1: 149, 3: 447, 5: 745, 10: 1490, 20: 2980}

## rank-1 program per theorem (first 25)

| theorem | ns | rank1 family | score | tactic |
|---|---|---|---|---|
| `Set.InjOn.image_diff_subset` | Set | d1_simp_lemma | 0.000217 | `simp [Set.InjOn.image_diff]` |
| `Set.InjOn.mem_of_mem_image` | Set | d2_simp_aesop | 0.186001 | `simp [Set.InjOn.image_inter] <;> aesop` |
| `Set.SurjOn.image_invFunOn_image_of_subset` | Set | d1_simp_lemma | 0.000665 | `simp [Set.InjOn.invFunOn_image]` |
| `Set.biInter_subset_of_mem` | Set | d2_simp_aesop | 0.045287 | `simp [Set.biInter_mono] <;> aesop` |
| `Set.compl_range_subset_kernImage` | Set | d2_simp_aesop | 0.089985 | `simp [Set.kernImage_preimage_eq_iff] <;> aeso` |
| `Set.diff_singleton_sSubset` | Set | d1_simp_lemma | 4e-06 | `simp [Set.subset_diff_singleton]` |
| `Set.diff_singleton_subset_iff` | Set | d2_simp_aesop | 1e-06 | `simp [Set.subset_diff_singleton] <;> aesop` |
| `Set.exists_image_eq_injOn_of_subset_range` | Set | d2_simp_aesop | 0.052516 | `simp [Set.subset_range_iff_exists_image_eq] <` |
| `Set.image_iInter` | Set | d2_simp_aesop | 0.001422 | `simp [Set.image_iInter_subset] <;> aesop` |
| `Set.preimage_invFun_of_mem` | Set | d2_simp_aesop | 0.067477 | `simp [Set.nonempty_of_nonempty_preimage] <;> ` |
| `Set.preimage_invFun_of_not_mem` | Set | d1_simp_lemma | 0.044661 | `simp [Set.nonempty_of_nonempty_preimage]` |
| `Set.ssubset_iff_sdiff_singleton` | Set | d1_simp_lemma | 3.6e-05 | `simp [Set.ssubset_univ_iff]` |
| `Set.ssubset_singleton_iff` | Set | d1_simp_lemma | 3e-06 | `simp [Set.ssubset_univ_iff]` |
| `Set.subset_biUnion_of_mem` | Set | d1_simp_lemma | 0.960775 | `simp [Finset.subset_biUnion_of_mem]` |
| `Set.subset_pair_iff_eq` | Set | d1_simp_lemma | 3e-05 | `simp [Set.pair_subset]` |
| `Set.subset_singleton_iff_eq` | Set | d1_simp_lemma | 1e-06 | `simp [Set.subset_compl_singleton_iff]` |
| `Set.BijOn.exists_extend_of_subset` | Set | d2_simp_aesop | 0.065169 | `simp [Set.BijOn.exists_extend] <;> aesop` |
| `Set.BijOn.image_eq` | Set | d2_simp_aesop | 0.000307 | `simp [Set.EqOn.image_eq] <;> aesop` |
| `Set.BijOn.iterate` | Set | d1_simp_lemma | 0.000211 | `simp [Set.bijOn_empty_iff_left]` |
| `Set.BijOn.subset_left` | Set | d1_simp_lemma | 0.002606 | `simp [Set.image2_iInter]` |
| `Set.BijOn.subset_range` | Set | d1_simp_lemma | 0.09585 | `simp [Set.range_const_subset]` |
| `Set.BijOn.subset_right` | Set | d1_simp_lemma | 0.001091 | `simp [Set.BijOn.subset_left]` |
| `Set.EqOn.image_eq` | Set | d2_simp_aesop | 0.00161 | `simp [Set.BijOn.image_eq] <;> aesop` |
| `Set.EqOn.image_eq_self` | Set | d1_simp_lemma | 0.000355 | `simp [Set.eqOn_refl]` |
| `Set.EqOn.inter_preimage_eq` | Set | d2_simp_aesop | 0.000219 | `simp [Set.EqOn.image_eq] <;> aesop` |
