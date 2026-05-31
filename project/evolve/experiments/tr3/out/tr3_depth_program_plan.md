# TR3 depth-program plan

- targets: 92 | total programs: 4377

## Family histogram
- d1_exact: 552
- d1_simpa_using: 552
- d1_simp_lemma: 552
- d1_simpa_lemma: 552
- d1_rw_lemma: 393
- d2_simp_aesop: 276
- d2_simp_simpall: 276
- d2_rw_aesop: 189
- d2_rw_simpall: 189
- d3_simp_try: 184
- d2_constructor_simpa: 120
- d1_aesop: 92
- d1_simp_all: 92
- d1_tauto: 92
- def_unfold_simp: 49
- d3_constructor_aesop: 40
- d3_constructor_simp_all: 40
- d2_ext_simp: 30
- d1_omega: 28
- d1_nlinarith: 28
- d3_ext_simp_aesop: 20
- d1_tofinset_simp: 11
- d2_ext_aesop: 10
- d3_antisymm_aesop: 10

## Sample (first 3 targets)
### Multiset.toFinset_eq_singleton_iff (54 programs, shape ['iff', 'multiset_tofinset'])
- d1 `simp [Multiset.card, Multiset.toFinset, Cycle.toFinset, Finset.card]` [def_unfold_simp]
- d1 `exact Multiset.toFinset_card_eq_card_iff_nodup` [d1_exact]
- d1 `simpa using Multiset.toFinset_card_eq_card_iff_nodup` [d1_simpa_using]
- d1 `simp [Multiset.toFinset_card_eq_card_iff_nodup]` [d1_simp_lemma]
- d1 `simpa [Multiset.toFinset_card_eq_card_iff_nodup]` [d1_simpa_lemma]
- d1 `rw [Multiset.toFinset_card_eq_card_iff_nodup]` [d1_rw_lemma]
- d1 `exact Multiset.singleton_eq_cons_iff` [d1_exact]
- d1 `simpa using Multiset.singleton_eq_cons_iff` [d1_simpa_using]
- d1 `simp [Multiset.singleton_eq_cons_iff]` [d1_simp_lemma]
- d1 `simpa [Multiset.singleton_eq_cons_iff]` [d1_simpa_lemma]

### Set.diff_singleton_subset_iff (47 programs, shape ['iff', 'subset'])
- d1 `simp [AList.insert]` [def_unfold_simp]
- d1 `exact Set.subset_insert_diff_singleton` [d1_exact]
- d1 `simpa using Set.subset_insert_diff_singleton` [d1_simpa_using]
- d1 `simp [Set.subset_insert_diff_singleton]` [d1_simp_lemma]
- d1 `simpa [Set.subset_insert_diff_singleton]` [d1_simpa_lemma]
- d1 `exact Set.subset_diff_singleton` [d1_exact]
- d1 `simpa using Set.subset_diff_singleton` [d1_simpa_using]
- d1 `simp [Set.subset_diff_singleton]` [d1_simp_lemma]
- d1 `simpa [Set.subset_diff_singleton]` [d1_simpa_lemma]
- d1 `exact Set.insert_diff_singleton` [d1_exact]

### Set.ite_eq_of_subset_left (51 programs, shape ['set_eq', 'subset'])
- d1 `simp [Set.ite]` [def_unfold_simp]
- d1 `exact Set.ite_eq_of_subset_right` [d1_exact]
- d1 `simpa using Set.ite_eq_of_subset_right` [d1_simpa_using]
- d1 `simp [Set.ite_eq_of_subset_right]` [d1_simp_lemma]
- d1 `simpa [Set.ite_eq_of_subset_right]` [d1_simpa_lemma]
- d1 `rw [Set.ite_eq_of_subset_right]` [d1_rw_lemma]
- d1 `exact Set.ite_subset_union` [d1_exact]
- d1 `simpa using Set.ite_subset_union` [d1_simpa_using]
- d1 `simp [Set.ite_subset_union]` [d1_simp_lemma]
- d1 `simpa [Set.ite_subset_union]` [d1_simpa_lemma]

