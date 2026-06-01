# RC5H dynamic program plan

- theorems: 90 | total programs: 1792
- budgets: {'B5': 5, 'B10': 10, 'B20': 20}
- family histogram: {'d2_simp_aesop': 246, 'd2_simp_simpall': 230, 'd1_simp_lemma': 470, 'd3_simp_try': 112, 'd1_exact': 167, 'd1_simpa_using': 137, 'd1_simpa_lemma': 111, 'd1_rw_lemma': 76, 'd1_tofinset_simp': 11, 'd1_aesop': 69, 'd1_simp_all': 2, 'd2_rw_aesop': 97, 'd2_rw_simpall': 2, 'd2_constructor_simpa': 1, 'def_unfold_simp': 21, 'd3_constructor_aesop': 4, 'd3_ext_simp_aesop': 19, 'd2_ext_simp': 13, 'd2_ext_aesop': 1, 'd1_omega': 1, 'd1_nlinarith': 1, 'd1_tauto': 1}
- namespace histogram: {'Finset': 32, 'Multiset': 19, 'Nat': 23, 'Set': 3, 'List': 13}

| theorem | ns | #programs | top tactic | top score |
|---|---|---|---|---|
| `Finset.biUnion_subset_iff_forall_subset` | Finset | 20 | `simp [Finset.biUnion_subset] <;> aesop` | 0.000774 |
| `Finset.image_subset_iff` | Finset | 20 | `simp [Finset.subset_image_iff] <;> aesop` | 0.000706 |
| `Multiset.Disjoint.symm` | Multiset | 20 | `simp [Multiset.disjoint_of_le_left] <;> aesop` | 0.37977 |
| `Multiset.add_eq_union_right_of_le` | Multiset | 20 | `simp [Multiset.union_le_union_right] <;> aesop` | 0.040278 |
| `Multiset.disjoint_comm` | Multiset | 20 | `simp [Multiset.erase_comm] <;> aesop` | 0.94886 |
| `Nat.sqrt_pos` | Nat | 20 | `simp [Nat.sqrt_eq] <;> aesop` | 0.008874 |
| `Set.disjoint_sUnion_right` | Set | 20 | `simp [Set.disjoint_sUnion_left] <;> aesop` | 0.987881 |
| `Set.mapsTo_singleton` | Set | 20 | `simp [Set.MapsTo.comp]` | 0.000957 |
| `Finset.card_mono` | Finset | 20 | `simp [Finset.card_map] <;> aesop` | 0.000232 |
| `Finset.card_strictMono` | Finset | 20 | `simp [Finset.strictMono_sym2] <;> aesop` | 0.002923 |
| `Finset.comp_inf_eq_inf_comp_of_is_total` | Finset | 20 | `simp [Finset.comp_inf_eq_inf_comp] <;> aesop` | 0.023369 |
| `Finset.comp_sup_eq_sup_comp_of_is_total` | Finset | 20 | `simp [Finset.comp_sup_eq_sup_comp] <;> aesop` | 0.049629 |
| `Finset.fin_mono` | Finset | 20 | `simp [Finset.sup_mono_fun]` | 0.000611 |
| `Finset.image_mono` | Finset | 20 | `simp [Finset.sup_mono_fun] <;> aesop` | 0.000106 |
| `Finset.max'_image` | Finset | 20 | `simp [Finset.le_max'] <;> aesop` | 0.002762 |
| `Finset.mem_fin` | Finset | 20 | `simp [Finset.orderEmbOfFin_mem]` | 0.348442 |
| `Finset.min'_image` | Finset | 20 | `simp [Finset.le_min'] <;> aesop` | 0.002762 |
| `Finset.monotone_preimage` | Finset | 20 | `simp [Finset.monotone_filter_left] <;> aesop` | 0.000568 |
| `Finset.subtype_mono` | Finset | 20 | `simp [Finset.subtype_map] <;> aesop` | 0.000187 |
| `Finset.Nonempty.inf_eq_bot_iff` | Finset | 20 | `simp [Finset.inf_eq_bot_iff] <;> aesop` | 0.084024 |
| `Finset.Nonempty.strong_induction` | Finset | 20 | `simp [Finset.strongDownwardInduction_eq]` | 0.000462 |
| `Finset.Nonempty.sup_eq_top_iff` | Finset | 20 | `simp [Finset.sup_eq_top_iff] <;> aesop` | 0.7873 |
| `Finset.Nontrivial.erase_nonempty` | Finset | 20 | `simp [Finset.not_nontrivial_empty] <;> aesop` | 6e-05 |
| `List.Pairwise.pmap` | List | 20 | `simp [List.pairwise_pmap] <;> aesop` | 0.017392 |
| `List.Pairwise.set_pairwise` | List | 20 | `simp [List.Pairwise.forall] <;> aesop` | 0.010461 |
| `List.Sublist.antisymm` | List | 20 | `simp [List.sublist_of_cons_sublist_cons] <;> aesop` | 0.004611 |
| `List.Sublist.map` | List | 20 | `simp [List.map_pure_sublist_sublists] <;> aesop` | 0.005424 |
| `List.Sublist.tail` | List | 20 | `simp [List.tail_sublist] <;> aesop` | 0.034038 |
| `List.append_cons_inj_of_not_mem` | List | 20 | `simp [List.length_injective_iff] <;> aesop` | 0.00876 |
| `List.append_left_eq_self` | List | 20 | `simp [List.self_eq_append_left] <;> aesop` | 0.31081 |
| `List.append_right_eq_self` | List | 20 | `simp [List.self_eq_append_left] <;> aesop` | 0.053912 |
| `List.attach_eq_nil` | List | 20 | `simp [List.takeWhile_eq_nil_iff] <;> aesop` | 0.099824 |
| `Multiset.Nodup.le_nsmul_iff_le` | Multiset | 20 | `simp [Multiset.nodup_iff_le] <;> aesop` | 0.047271 |
| `Multiset.Rel.countP_eq` | Multiset | 20 | `simp [Multiset.countP_eq_countP_filter_add] <;> aesop` | 0.202332 |
| `Multiset.Rel.mono` | Multiset | 20 | `simp [Multiset.rel_bind] <;> aesop` | 0.009406 |
| `Multiset.Subset.refl` | Multiset | 20 | `simp [Finset.Subset.refl]` | 0.049437 |
| `Multiset.Subset.trans` | Multiset | 20 | `simp [Finset.Subset.trans]` | 0.007625 |
| `Multiset.addHom_ext` | Multiset | 20 | `simp [Multiset.ext] <;> aesop` | 0.013513 |
| `Multiset.add_bind` | Multiset | 20 | `simp [Multiset.bind_add] <;> aesop` | 0.013199 |
| `Multiset.add_cons` | Multiset | 20 | `simp [Multiset.cons_add] <;> aesop` | 0.155699 |
| `Multiset.add_product` | Multiset | 20 | `simp [Multiset.product_add] <;> aesop` | 0.214408 |
| `Multiset.add_sigma` | Multiset | 20 | `simp [Multiset.sigma_add] <;> aesop` | 0.161475 |
| `Multiset.add_singleton_eq_iff` | Multiset | 20 | `simp [Multiset.mem_singleton]` | 0.014037 |
| `Multiset.add_union_distrib` | Multiset | 20 | `simp [Multiset.union_le_add]` | 0.962906 |
| `Multiset.attach_bind_coe` | Multiset | 20 | `simp [Multiset.coe_bind] <;> aesop` | 0.086828 |
| `Multiset.attach_cons` | Multiset | 20 | `simp [Multiset.mem_cons_of_mem] <;> aesop` | 0.012646 |
| `Multiset.attach_map_val` | Multiset | 20 | `simp [Finset.attach_map_val] <;> aesop` | 0.272806 |
| `Multiset.attach_map_val'` | Multiset | 20 | `simp [Multiset.erase_attach_map_val] <;> aesop` | 0.07703 |
| `Nat.add_div_of_dvd_left` | Nat | 20 | `simp [Nat.dvd_add_left] <;> aesop` | 0.002245 |
| `Nat.add_div_of_dvd_right` | Nat | 20 | `simp [Nat.add_div_of_dvd_left] <;> aesop` | 0.001186 |
| `Nat.add_le_mul` | Nat | 20 | `simp [Nat.add_sub_one_le_mul] <;> aesop` | 0.001422 |
| `Nat.clog_anti_left` | Nat | 20 | `simp [Nat.clog_of_left_le_one] <;> aesop` | 0.0025 |
| `Nat.clog_eq_one` | Nat | 20 | `simp [Nat.clog_one_left] <;> aesop` | 0.000358 |
| `Nat.clog_mono_right` | Nat | 20 | `simp [Nat.clog_one_left]` | 0.000134 |
| `Nat.clog_of_left_le_one` | Nat | 20 | `simp [Nat.clog_one_left] <;> aesop` | 0.005727 |
| `Set.BijOn.congr` | Set | 20 | `simp [Set.InjOn.congr] <;> aesop` | 0.000242 |
| `Finset.disjoint_insert_right` | Finset | 20 | `simp [Finset.disjoint_erase_insert] <;> aesop` | 0.01725 |
| `Finset.mem_insert` | Finset | 20 | `simp [Finset.mem_insert_self] <;> simp_all` | 0.053599 |
| `Finset.mem_singleton` | Finset | 20 | `simp [Finset.map_singleton]` | 0.002454 |
| `Nat.AM_GM` | Nat | 12 | `simp [Nat.Upto] <;> aesop` | 0.002535 |
| `Nat.div_ne_zero_iff` | Nat | 20 | `simp [Nat.one_le_iff_ne_zero] <;> aesop` | 0.001436 |
| `Nat.dvd_right_iff_eq` | Nat | 20 | `simp [Nat.dvd_add_right] <;> aesop` | 0.001382 |
| `Nat.dvd_left_iff_eq` | Nat | 20 | `simp [Nat.dvd_add_left] <;> aesop` | 0.006403 |
| `Nat.eq_div_of_mul_eq_left` | Nat | 20 | `simp [Nat.eq_mul_of_div_eq_left] <;> aesop` | 0.00728 |
| `Nat.eq_mul_of_div_eq_left` | Nat | 20 | `simp [Nat.eq_div_of_mul_eq_left] <;> aesop` | 0.00728 |
| `Nat.find_eq_zero` | Nat | 20 | `simp [Nat.nth_zero_of_exists] <;> aesop` | 0.000168 |
| `Nat.forall_lt_succ` | Nat | 20 | `simp [Nat.one_lt_succ_succ] <;> aesop` | 0.000375 |
| `Nat.le_of_mul_le_mul_right` | Nat | 20 | `simp [Nat.eq_zero_of_mul_le] <;> aesop` | 0.000604 |
| `Nat.mod_mul_mod` | Nat | 20 | `simp [Nat.ModEq.mul] <;> aesop` | 0.000473 |
| `Nat.mod_eq_iff_lt` | Nat | 20 | `simp [Nat.mod_succ_eq_iff_lt] <;> aesop` | 0.003151 |
| `Nat.one_le_div_iff` | Nat | 20 | `simp [Nat.one_le_bit0_iff]` | 6.7e-05 |
| `Nat.one_le_pow` | Nat | 20 | `simp [Nat.one_lt_pow] <;> aesop` | 0.000192 |
| `Nat.sqrt_lt'` | Nat | 20 | `simp [Nat.sqrt_eq] <;> aesop` | 0.003103 |
| `Nat.zero_eq_mul` | Nat | 20 | `simp [Nat.dist_eq_zero] <;> aesop` | 0.0016 |
| `Finset.card_union_eq_card_add_card` | Finset | 20 | `simp [Finset.card_sdiff_add_card]` | 0.051173 |
| `Finset.codisjoint_inf_left` | Finset | 20 | `simp [Finset.inf_sup_distrib_left] <;> aesop` | 0.012638 |
| `Finset.codisjoint_inf_right` | Finset | 20 | `simp [Finset.codisjoint_inf_left] <;> aesop` | 0.026398 |
| `Finset.disjoint_filter_filter'` | Finset | 20 | `simp [Finset.map_filter'] <;> aesop` | 5e-06 |
| `Finset.disjoint_image` | Finset | 20 | `simp [Finset.mem_image_of_mem] <;> aesop` | 0.060304 |
| `Finset.disjoint_map` | Finset | 20 | `simp [Finset.disjoint_map_inl_map_inr] <;> aesop` | 0.275304 |
| `Finset.disjoint_sup_left` | Finset | 20 | `simp [Finset.inf_sup_distrib_left] <;> aesop` | 0.561354 |
| `Finset.disjoint_sup_right` | Finset | 20 | `simp [Finset.inf_sup_distrib_right] <;> aesop` | 0.223106 |
| `Finset.pairwise_cons'` | Finset | 20 | `aesop` | 8e-06 |
| `Finset.powerset_card_disjiUnion` | Finset | 20 | `simp [Finset.coe_disjiUnion] <;> aesop` | 0.994334 |
| `List.disjoint_map` | List | 20 | `simp [List.map_eq_map] <;> aesop` | 0.078832 |
| `List.disjoint_pmap` | List | 20 | `simp [List.mem_pmap] <;> aesop` | 0.122181 |
| `Finset.disjoint_biUnion_left` | Finset | 20 | `simp [Finset.biUnion_image_sup_left]` | 0.805976 |
| `Finset.disjoint_biUnion_right` | Finset | 20 | `simp [Finset.biUnion_image_right] <;> aesop` | 0.081391 |
| `List.perm_of_nodup_nodup_toFinset_eq` | List | 20 | `simp [List.count_eq_one_of_mem]` | 8.5e-05 |
| `List.toFinset.ext_iff` | List | 20 | `simp [List.ext_get_iff] <;> simp_all` | 0.805228 |
