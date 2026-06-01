# TR6 ranked program plan

- ranker: hgb | theorems: 90 | scored 4058 programs
- family histogram: {'d2_simp_aesop': 246, 'd2_simp_simpall': 230, 'd1_simp_lemma': 470, 'd3_simp_try': 112, 'd1_exact': 167, 'd1_simpa_using': 137, 'd1_simpa_lemma': 111, 'd1_rw_lemma': 76, 'd1_tofinset_simp': 11, 'd1_aesop': 69, 'd1_simp_all': 2, 'd2_rw_aesop': 97, 'd2_rw_simpall': 2, 'd2_constructor_simpa': 1, 'def_unfold_simp': 21, 'd3_constructor_aesop': 4, 'd3_ext_simp_aesop': 19, 'd2_ext_simp': 13, 'd2_ext_aesop': 1, 'd1_omega': 1, 'd1_nlinarith': 1, 'd1_tauto': 1}
- programs per budget: {1: 90, 3: 270, 5: 450, 10: 900, 20: 1792}

## rank-1 program per theorem (first 25)

| theorem | ns | rank1 family | score | tactic |
|---|---|---|---|---|
| `Finset.biUnion_subset_iff_forall_subset` | Finset | d2_simp_aesop | 0.000774 | `simp [Finset.biUnion_subset] <;> aesop` |
| `Finset.image_subset_iff` | Finset | d2_simp_aesop | 0.000706 | `simp [Finset.subset_image_iff] <;> aesop` |
| `Multiset.Disjoint.symm` | Multiset | d2_simp_aesop | 0.37977 | `simp [Multiset.disjoint_of_le_left] <;> aesop` |
| `Multiset.add_eq_union_right_of_le` | Multiset | d2_simp_aesop | 0.040278 | `simp [Multiset.union_le_union_right] <;> aeso` |
| `Multiset.disjoint_comm` | Multiset | d2_simp_aesop | 0.94886 | `simp [Multiset.erase_comm] <;> aesop` |
| `Nat.sqrt_pos` | Nat | d2_simp_aesop | 0.008874 | `simp [Nat.sqrt_eq] <;> aesop` |
| `Set.disjoint_sUnion_right` | Set | d2_simp_aesop | 0.987881 | `simp [Set.disjoint_sUnion_left] <;> aesop` |
| `Set.mapsTo_singleton` | Set | d1_simp_lemma | 0.000957 | `simp [Set.MapsTo.comp]` |
| `Finset.card_mono` | Finset | d2_simp_aesop | 0.000232 | `simp [Finset.card_map] <;> aesop` |
| `Finset.card_strictMono` | Finset | d2_simp_aesop | 0.002923 | `simp [Finset.strictMono_sym2] <;> aesop` |
| `Finset.comp_inf_eq_inf_comp_of_is_total` | Finset | d2_simp_aesop | 0.023369 | `simp [Finset.comp_inf_eq_inf_comp] <;> aesop` |
| `Finset.comp_sup_eq_sup_comp_of_is_total` | Finset | d2_simp_aesop | 0.049629 | `simp [Finset.comp_sup_eq_sup_comp] <;> aesop` |
| `Finset.fin_mono` | Finset | d1_simp_lemma | 0.000611 | `simp [Finset.sup_mono_fun]` |
| `Finset.image_mono` | Finset | d2_simp_aesop | 0.000106 | `simp [Finset.sup_mono_fun] <;> aesop` |
| `Finset.max'_image` | Finset | d2_simp_aesop | 0.002762 | `simp [Finset.le_max'] <;> aesop` |
| `Finset.mem_fin` | Finset | d1_simp_lemma | 0.348442 | `simp [Finset.orderEmbOfFin_mem]` |
| `Finset.min'_image` | Finset | d2_simp_aesop | 0.002762 | `simp [Finset.le_min'] <;> aesop` |
| `Finset.monotone_preimage` | Finset | d2_simp_aesop | 0.000568 | `simp [Finset.monotone_filter_left] <;> aesop` |
| `Finset.subtype_mono` | Finset | d2_simp_aesop | 0.000187 | `simp [Finset.subtype_map] <;> aesop` |
| `Finset.Nonempty.inf_eq_bot_iff` | Finset | d2_simp_aesop | 0.084024 | `simp [Finset.inf_eq_bot_iff] <;> aesop` |
| `Finset.Nonempty.strong_induction` | Finset | d1_simp_lemma | 0.000462 | `simp [Finset.strongDownwardInduction_eq]` |
| `Finset.Nonempty.sup_eq_top_iff` | Finset | d2_simp_aesop | 0.7873 | `simp [Finset.sup_eq_top_iff] <;> aesop` |
| `Finset.Nontrivial.erase_nonempty` | Finset | d2_simp_aesop | 6e-05 | `simp [Finset.not_nontrivial_empty] <;> aesop` |
| `List.Pairwise.pmap` | List | d2_simp_aesop | 0.017392 | `simp [List.pairwise_pmap] <;> aesop` |
| `List.Pairwise.set_pairwise` | List | d2_simp_aesop | 0.010461 | `simp [List.Pairwise.forall] <;> aesop` |
