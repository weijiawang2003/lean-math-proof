# TR5 ranked program plan

- ranker model: **hgb** | theorems: 92 | confirmed failures: 92
- total programs scored: 4377
- family histogram: {'d1_simp_lemma': 552, 'd2_simp_simpall': 276, 'd2_simp_aesop': 276, 'def_unfold_simp': 49, 'd3_simp_try': 184, 'd1_aesop': 92, 'd1_rw_lemma': 393, 'd1_exact': 552, 'd2_rw_aesop': 189, 'd1_simpa_using': 552, 'd1_simpa_lemma': 552, 'd2_rw_simpall': 189, 'd1_simp_all': 92, 'd1_tauto': 92, 'd3_constructor_aesop': 40, 'd2_constructor_simpa': 120, 'd3_constructor_simp_all': 40, 'd1_tofinset_simp': 11, 'd2_ext_simp': 30, 'd3_ext_simp_aesop': 20, 'd3_antisymm_aesop': 10, 'd2_ext_aesop': 10, 'd1_omega': 28, 'd1_nlinarith': 28}
- programs to run per budget: {1: 92, 3: 276, 5: 460, 10: 920, 20: 1840}

## Top program per theorem (rank 1)

| theorem | ns | rank1 family | rank1 score | rank1 tactic |
|---|---|---|---|---|
| `Set.antitoneOn_iff_antitone` | Set | def_unfold_simp | 1.0 | `simp [Antitone, AntitoneOn]` |
| `Set.monotoneOn_iff_monotone` | Set | def_unfold_simp | 1.0 | `simp [Monotone, MonotoneOn]` |
| `Set.strictAntiOn_iff_strictAnti` | Set | def_unfold_simp | 1.0 | `simp [StrictAnti, StrictAntiOn]` |
| `Set.strictMonoOn_iff_strictMono` | Set | def_unfold_simp | 1.0 | `simp [StrictMono, StrictMonoOn]` |
| `Finset.mem_disjUnion` | Finset | d1_simp_lemma | 0.999953 | `simp [Finset.coe_disjUnion]` |
| `List.toFinset.ext_iff` | List | d1_simp_lemma | 0.999174 | `simp [Finset.ext_iff]` |
| `Set.disjoint_right` | Set | d2_simp_aesop | 0.998857 | `simp [Set.disjoint_left] <;> aesop` |
| `Set.compl_union_self` | Set | d1_simp_lemma | 0.998714 | `simp [Set.union_eq_compl_compl_inter_compl]` |
| `Set.disjoint_iff_forall_ne` | Set | d2_simp_aesop | 0.998498 | `simp [Set.disjoint_left] <;> aesop` |
| `Set.disjoint_singleton_left` | Set | d1_simp_lemma | 0.998487 | `simp [Set.disjoint_left]` |
| `Set.Nonempty.subset_pair_iff_eq` | Set | d2_simp_aesop | 0.998038 | `simp [Set.subset_pair_iff_eq] <;> aesop` |
| `List.toFinset_eq` | List | d1_simp_lemma | 0.998004 | `simp [Multiset.toFinset_eq]` |
| `Prop.compl_singleton` | Prop | d1_aesop | 0.997885 | `aesop` |
| `Set.pair_eq_pair_iff` | Set | d2_simp_aesop | 0.000359 | `simp [Set.subset_pair_iff_eq] <;> aesop` |
| `Finset.eq_singleton_iff_nonempty_unique_mem` | Finset | def_unfold_simp | 0.000254 | `simp [Finset.Nonempty]` |
| `Multiset.Nodup.toFinset_inj` | Multiset | d2_simp_aesop | 0.000215 | `simp [Multiset.toFinset_eq] <;> aesop` |
| `Set.Nonempty.eq_univ` | Set | d2_simp_aesop | 0.000198 | `simp [Set.nonempty_iff_univ_nonempty] <;> aesop` |
| `List.toFinset_filter` | List | d1_simp_lemma | 0.000117 | `simp [Multiset.toFinset_filter]` |
| `Multiset.toFinset_eq_singleton_iff` | Multiset | def_unfold_simp | 0.000116 | `simp [Multiset.card, Multiset.toFinset, Cycle.toFinset,` |
| `Set.insert_subset_insert_iff` | Set | d2_simp_aesop | 0.000101 | `simp [Set.insert_subset_iff] <;> aesop` |
| `Set.subset_pair_iff_eq` | Set | d2_simp_aesop | 9.6e-05 | `simp [Set.Nonempty.subset_pair_iff_eq] <;> aesop` |
| `Finset.subset_union_elim` | Finset | d1_simp_lemma | 5.9e-05 | `simp [Finset.union_subset_iff]` |
| `Set.ssubset_iff_sdiff_singleton` | Set | d1_simp_lemma | 3.6e-05 | `simp [Set.ssubset_univ_iff]` |
| `Set.powerset_singleton` | Set | d1_aesop | 3.4e-05 | `aesop` |
| `List.toFinset_eq_iff_perm_dedup` | List | d2_simp_aesop | 2.8e-05 | `simp [List.toFinset_eq_of_perm] <;> aesop` |
