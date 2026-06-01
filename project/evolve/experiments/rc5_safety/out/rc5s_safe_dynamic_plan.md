# RC5S safe dynamic plan

- theorems with programs: 89 | gated out: 22
- B5 programs: 444 | B10 reserve: 384 | rejected at generation: 0
- **off-policy in final plan: 0** (must be 0)
- B5 pattern histogram: {'simp_L_aesop': 168, 'simp_L': 242, 'rw_L_aesop': 4, 'exact_L': 16, 'simpa_using_L': 12, 'ext_simp_L': 1, 'simpa_L': 1}

| theorem | ns | category | #programs | top tactic |
|---|---|---|---|---|
| `Finset.Nonempty.inf_eq_bot_iff` | Finset | prior_stall_cases | 8 | `simp [Finset.inf_eq_bot_iff] <;> aesop` |
| `Finset.Nonempty.strong_induction` | Finset | prior_stall_cases | 10 | `simp [Finset.strongDownwardInduction_eq]` |
| `Finset.Nonempty.sup_eq_top_iff` | Finset | prior_stall_cases | 9 | `simp [Finset.sup_eq_top_iff] <;> aesop` |
| `Finset.Nontrivial.erase_nonempty` | Finset | prior_stall_cases | 10 | `simp [Finset.not_nontrivial_empty] <;> aesop` |
| `Finset.biUnion_subset_iff_forall_subset` | Finset | true_winners | 10 | `simp [Finset.biUnion_subset] <;> aesop` |
| `Finset.card_mono` | Finset | prior_stall_cases | 10 | `simp [Finset.card_map] <;> aesop` |
| `Finset.card_strictMono` | Finset | prior_stall_cases | 10 | `simp [Finset.strictMono_sym2] <;> aesop` |
| `Finset.card_union_eq_card_add_card` | Finset | prior_stall_cases | 8 | `simp [Finset.card_sdiff_add_card]` |
| `Finset.codisjoint_inf_left` | Finset | prior_stall_cases | 9 | `simp [Finset.inf_sup_distrib_left] <;> aesop` |
| `Finset.codisjoint_inf_right` | Finset | prior_stall_cases | 9 | `simp [Finset.codisjoint_inf_left] <;> aesop` |
| `Finset.comp_inf_eq_inf_comp_of_is_total` | Finset | prior_stall_cases | 9 | `simp [Finset.comp_inf_eq_inf_comp] <;> aesop` |
| `Finset.comp_sup_eq_sup_comp_of_is_total` | Finset | prior_stall_cases | 9 | `simp [Finset.comp_sup_eq_sup_comp] <;> aesop` |
| `Finset.disjoint_biUnion_left` | Finset | prior_stall_cases | 7 | `simp [Finset.biUnion_image_sup_left]` |
| `Finset.disjoint_biUnion_right` | Finset | prior_stall_cases | 8 | `simp [Finset.biUnion_image_right] <;> aesop` |
| `Finset.disjoint_filter_filter'` | Finset | prior_stall_cases | 10 | `simp [Finset.map_filter'] <;> aesop` |
| `Finset.disjoint_image` | Finset | prior_stall_cases | 9 | `simp [Finset.mem_image_of_mem] <;> aesop` |
| `Finset.disjoint_insert_right` | Finset | prior_stall_cases | 10 | `simp [Finset.disjoint_erase_insert] <;> aesop` |
| `Finset.disjoint_map` | Finset | prior_stall_cases | 10 | `simp [Finset.disjoint_map_inl_map_inr] <;> aesop` |
| `Finset.disjoint_sup_left` | Finset | prior_stall_cases | 9 | `simp [Finset.inf_sup_distrib_left] <;> aesop` |
| `Finset.disjoint_sup_right` | Finset | prior_stall_cases | 9 | `simp [Finset.inf_sup_distrib_right] <;> aesop` |
| `Finset.fin_mono` | Finset | prior_stall_cases | 9 | `simp [Finset.sup_mono_fun]` |
| `Finset.image_mono` | Finset | prior_stall_cases | 10 | `simp [Finset.sup_mono_fun] <;> aesop` |
| `Finset.image_subset_iff` | Finset | true_winners | 10 | `simp [Finset.subset_image_iff] <;> aesop` |
| `Finset.max'_image` | Finset | prior_stall_cases | 10 | `simp [Finset.le_max'] <;> aesop` |
| `Finset.mem_fin` | Finset | prior_stall_cases | 7 | `simp [Finset.orderEmbOfFin_mem]` |
| `Finset.mem_insert` | Finset | prior_stall_cases | 10 | `simp [Finset.mem_insert_self] <;> aesop` |
| `Finset.mem_singleton` | Finset | off_policy_cases | 10 | `simp [Finset.map_singleton]` |
| `Finset.min'_image` | Finset | prior_stall_cases | 10 | `simp [Finset.le_min'] <;> aesop` |
| `Finset.monotone_preimage` | Finset | prior_stall_cases | 10 | `simp [Finset.monotone_filter_left] <;> aesop` |
| `Finset.pairwise_cons'` | Finset | off_policy_cases | 9 | `simp [Finset.pairwiseDisjoint_slice] <;> aesop` |
| `Finset.powerset_card_disjiUnion` | Finset | prior_stall_cases | 9 | `simp [Finset.coe_disjiUnion] <;> aesop` |
| `Finset.subtype_mono` | Finset | prior_stall_cases | 10 | `simp [Finset.subtype_map] <;> aesop` |
| `List.Pairwise.pmap` | List | prior_stall_cases | 9 | `simp [List.pairwise_pmap] <;> aesop` |
| `List.Pairwise.set_pairwise` | List | prior_stall_cases | 10 | `simp [List.Pairwise.forall] <;> aesop` |
| `List.Sublist.antisymm` | List | prior_stall_cases | 10 | `simp [List.sublist_of_cons_sublist_cons] <;> aesop` |
| `List.Sublist.map` | List | prior_stall_cases | 10 | `simp [List.map_pure_sublist_sublists] <;> aesop` |
| `List.Sublist.tail` | List | prior_stall_cases | 10 | `simp [List.tail_sublist] <;> aesop` |
| `List.append_cons_inj_of_not_mem` | List | prior_stall_cases | 9 | `simp [List.length_injective_iff] <;> aesop` |
| `List.append_left_eq_self` | List | prior_stall_cases | 9 | `simp [List.self_eq_append_left] <;> aesop` |
| `List.append_right_eq_self` | List | prior_stall_cases | 9 | `simp [List.self_eq_append_left] <;> aesop` |
| `List.attach_eq_nil` | List | prior_stall_cases | 9 | `simp [List.takeWhile_eq_nil_iff] <;> aesop` |
| `List.disjoint_map` | List | prior_stall_cases | 8 | `simp [List.map_eq_map] <;> aesop` |
| `List.disjoint_pmap` | List | prior_stall_cases | 9 | `simp [List.mem_pmap] <;> aesop` |
| `List.perm_of_nodup_nodup_toFinset_eq` | List | prior_stall_cases | 9 | `simp [List.count_eq_one_of_mem]` |
| `List.toFinset.ext_iff` | List | prior_stall_cases | 8 | `simp [List.ext_get_iff] <;> aesop` |
| `Multiset.Disjoint.symm` | Multiset | prior_stall_cases | 10 | `simp [Multiset.disjoint_of_le_left] <;> aesop` |
| `Multiset.Nodup.le_nsmul_iff_le` | Multiset | prior_stall_cases | 9 | `simp [Multiset.nodup_iff_le] <;> aesop` |
| `Multiset.Rel.countP_eq` | Multiset | prior_stall_cases | 9 | `simp [Multiset.countP_eq_countP_filter_add] <;> aesop` |
| `Multiset.Rel.mono` | Multiset | prior_stall_cases | 10 | `simp [Multiset.rel_bind] <;> aesop` |
| `Multiset.Subset.refl` | Multiset | prior_stall_cases | 9 | `simp [Finset.Subset.refl]` |
| `Multiset.Subset.trans` | Multiset | prior_stall_cases | 10 | `simp [Finset.Subset.trans]` |
| `Multiset.addHom_ext` | Multiset | prior_stall_cases | 10 | `simp [Multiset.ext] <;> aesop` |
| `Multiset.add_bind` | Multiset | true_winners | 10 | `simp [Multiset.bind_add] <;> aesop` |
| `Multiset.add_cons` | Multiset | prior_stall_cases | 9 | `simp [Multiset.cons_add] <;> aesop` |
| `Multiset.add_eq_union_right_of_le` | Multiset | prior_stall_cases | 10 | `simp [Multiset.union_le_union_right] <;> aesop` |
| `Multiset.add_product` | Multiset | prior_stall_cases | 9 | `simp [Multiset.product_add] <;> aesop` |
| `Multiset.add_sigma` | Multiset | prior_stall_cases | 10 | `simp [Multiset.sigma_add] <;> aesop` |
| `Multiset.add_singleton_eq_iff` | Multiset | prior_stall_cases | 8 | `simp [Multiset.mem_singleton]` |
| `Multiset.add_union_distrib` | Multiset | prior_stall_cases | 9 | `simp [Multiset.union_le_add]` |
| `Multiset.attach_bind_coe` | Multiset | prior_stall_cases | 10 | `simp [Multiset.coe_bind] <;> aesop` |
| `Multiset.attach_cons` | Multiset | prior_stall_cases | 10 | `simp [Multiset.mem_cons_of_mem] <;> aesop` |
| `Multiset.attach_map_val` | Multiset | prior_stall_cases | 8 | `simp [Finset.attach_map_val] <;> aesop` |
| `Multiset.attach_map_val'` | Multiset | prior_stall_cases | 8 | `simp [Multiset.erase_attach_map_val] <;> aesop` |
| `Multiset.disjoint_comm` | Multiset | prior_stall_cases | 10 | `simp [Multiset.erase_comm] <;> aesop` |
| `Nat.AM_GM` | Nat | prior_stall_cases | 4 | `simp [Nat.Upto]` |
| `Nat.add_div_of_dvd_left` | Nat | prior_stall_cases | 10 | `simp [Nat.dvd_add_self_left]` |
| `Nat.add_div_of_dvd_right` | Nat | prior_stall_cases | 10 | `simp [Nat.add_div_of_dvd_left]` |
| `Nat.add_le_mul` | Nat | prior_stall_cases | 10 | `simp [Nat.le_mul_self]` |
| `Nat.clog_anti_left` | Nat | prior_stall_cases | 9 | `simp [Nat.clog_zero_left]` |
| `Nat.clog_eq_one` | Nat | prior_stall_cases | 10 | `simp [Nat.clog_of_left_le_one]` |
| `Nat.clog_mono_right` | Nat | prior_stall_cases | 10 | `simp [Nat.clog_one_left]` |
| `Nat.clog_of_left_le_one` | Nat | prior_stall_cases | 8 | `simp [Nat.le_pow_clog]` |
| `Nat.div_ne_zero_iff` | Nat | prior_stall_cases | 10 | `simp [Nat.le_iff_ne_zero_of_dvd]` |
| `Nat.dvd_left_iff_eq` | Nat | prior_stall_cases | 10 | `simp [Nat.dvd_add_self_left]` |
| `Nat.dvd_right_iff_eq` | Nat | prior_stall_cases | 10 | `simp [Nat.dvd_add_self_right]` |
| `Nat.eq_div_of_mul_eq_left` | Nat | prior_stall_cases | 8 | `simp [Nat.eq_mul_of_div_eq_left]` |
| `Nat.eq_mul_of_div_eq_left` | Nat | prior_stall_cases | 8 | `simp [Nat.eq_div_of_mul_eq_left]` |
| `Nat.find_eq_zero` | Nat | prior_stall_cases | 10 | `simp [Nat.find_add]` |
| `Nat.forall_lt_succ` | Nat | prior_stall_cases | 10 | `simp [Nat.exists_lt_succ]` |
| `Nat.le_of_mul_le_mul_right` | Nat | prior_stall_cases | 10 | `simp [Nat.mul_self_le_mul_self]` |
| `Nat.mod_eq_iff_lt` | Nat | prior_stall_cases | 10 | `simp [Nat.modEq_iff_dvd]` |
| `Nat.mod_mul_mod` | Nat | prior_stall_cases | 10 | `simp [Nat.mul_add_mod_of_lt]` |
| `Nat.one_le_pow` | Nat | prior_stall_cases | 10 | `simp [Nat.pow_le_choose]` |
| `Nat.sqrt_lt'` | Nat | prior_stall_cases | 10 | `simp [Nat.sqrt_eq]` |
| `Nat.sqrt_pos` | Nat | prior_stall_cases | 10 | `simp [Nat.eq_sqrt]` |
| `Nat.zero_eq_mul` | Nat | prior_stall_cases | 10 | `simp [Nat.xor_eq_zero]` |
| `Set.BijOn.congr` | Set | prior_stall_cases | 9 | `simp [Set.InjOn.congr] <;> aesop` |
| `Set.disjoint_sUnion_right` | Set | prior_stall_cases | 8 | `simp [Set.disjoint_sUnion_left] <;> aesop` |
| `Set.mapsTo_singleton` | Set | prior_stall_cases | 10 | `simp [Set.MapsTo.comp]` |
