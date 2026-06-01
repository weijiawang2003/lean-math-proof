# RC5H hybrid attribution

- dynamic attempts: 90 | classifications: {'NO_DYNAMIC_WIN': 87, 'TRUE_HYBRID_DELTA': 3}
- **TRUE_HYBRID_DELTA: 3** ['Finset.biUnion_subset_iff_forall_subset', 'Finset.image_subset_iff', 'Multiset.add_bind']
- dynamic wins total: 3 | source-specific: 0
- true delta by namespace: {'Finset': 2, 'Multiset': 1} | by family: {'d2_simp_aesop': 1, 'd1_simp_lemma': 1, 'def_unfold_simp': 1} | by budget: {'5': 3}

| theorem | ns | rc2 | rc4 | dyn | class | winning program |
|---|---|---|---|---|---|---|
| `Finset.biUnion_subset_iff_forall_subset` | Finset | F | F | S | TRUE_HYBRID_DELTA | `simp [Finset.biUnion_subset] <;> aesop` |
| `Finset.image_subset_iff` | Finset | F | F | S | TRUE_HYBRID_DELTA | `simp [Finset.subset_iff]` |
| `Multiset.add_bind` | Multiset | F | F | S | TRUE_HYBRID_DELTA | `simp [Multiset.bind]` |
| `Finset.Nonempty.inf_eq_bot_iff` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.Nonempty.strong_induction` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.Nonempty.sup_eq_top_iff` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.Nontrivial.erase_nonempty` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.card_mono` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.card_strictMono` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.card_union_eq_card_add_card` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.codisjoint_inf_left` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.codisjoint_inf_right` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.comp_inf_eq_inf_comp_of_is_total` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.comp_sup_eq_sup_comp_of_is_total` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.disjoint_biUnion_left` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.disjoint_biUnion_right` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.disjoint_filter_filter'` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.disjoint_image` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.disjoint_insert_right` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.disjoint_map` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.disjoint_sup_left` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.disjoint_sup_right` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.fin_mono` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.image_mono` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.max'_image` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.mem_fin` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.mem_insert` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.mem_singleton` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.min'_image` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.monotone_preimage` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.pairwise_cons'` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.powerset_card_disjiUnion` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Finset.subtype_mono` | Finset | F | F | F | NO_DYNAMIC_WIN | `` |
| `List.Pairwise.pmap` | List | F | F | F | NO_DYNAMIC_WIN | `` |
| `List.Pairwise.set_pairwise` | List | F | F | F | NO_DYNAMIC_WIN | `` |
| `List.Sublist.antisymm` | List | F | F | F | NO_DYNAMIC_WIN | `` |
| `List.Sublist.map` | List | F | F | F | NO_DYNAMIC_WIN | `` |
| `List.Sublist.tail` | List | F | F | F | NO_DYNAMIC_WIN | `` |
| `List.append_cons_inj_of_not_mem` | List | F | F | F | NO_DYNAMIC_WIN | `` |
| `List.append_left_eq_self` | List | F | F | F | NO_DYNAMIC_WIN | `` |
| `List.append_right_eq_self` | List | F | F | F | NO_DYNAMIC_WIN | `` |
| `List.attach_eq_nil` | List | F | F | F | NO_DYNAMIC_WIN | `` |
| `List.disjoint_map` | List | F | F | F | NO_DYNAMIC_WIN | `` |
| `List.disjoint_pmap` | List | F | F | F | NO_DYNAMIC_WIN | `` |
| `List.perm_of_nodup_nodup_toFinset_eq` | List | F | F | F | NO_DYNAMIC_WIN | `` |
| `List.toFinset.ext_iff` | List | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.Disjoint.symm` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.Nodup.le_nsmul_iff_le` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.Rel.countP_eq` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.Rel.mono` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.Subset.refl` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.Subset.trans` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.addHom_ext` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.add_cons` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.add_eq_union_right_of_le` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.add_product` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.add_sigma` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.add_singleton_eq_iff` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.add_union_distrib` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.attach_bind_coe` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.attach_cons` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.attach_map_val` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.attach_map_val'` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Multiset.disjoint_comm` | Multiset | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.AM_GM` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.add_div_of_dvd_left` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.add_div_of_dvd_right` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.add_le_mul` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.clog_anti_left` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.clog_eq_one` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.clog_mono_right` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.clog_of_left_le_one` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.div_ne_zero_iff` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.dvd_left_iff_eq` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.dvd_right_iff_eq` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.eq_div_of_mul_eq_left` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.eq_mul_of_div_eq_left` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.find_eq_zero` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.forall_lt_succ` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.le_of_mul_le_mul_right` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.mod_eq_iff_lt` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.mod_mul_mod` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.one_le_div_iff` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.one_le_pow` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.sqrt_lt'` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.sqrt_pos` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Nat.zero_eq_mul` | Nat | F | F | F | NO_DYNAMIC_WIN | `` |
| `Set.BijOn.congr` | Set | F | F | F | NO_DYNAMIC_WIN | `` |
| `Set.disjoint_sUnion_right` | Set | F | F | F | NO_DYNAMIC_WIN | `` |
| `Set.mapsTo_singleton` | Set | F | F | F | NO_DYNAMIC_WIN | `` |
