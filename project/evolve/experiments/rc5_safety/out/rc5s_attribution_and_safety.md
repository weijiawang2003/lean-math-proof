# RC5S attribution & safety

- classification: {'SAFE_TRUE_DYNAMIC_WIN': 3, 'NO_WIN_SAFE_BUDGET': 86}
- **recovered prior wins: 3/3** ['Finset.biUnion_subset_iff_forall_subset', 'Finset.image_subset_iff', 'Multiset.add_bind']
- lost prior wins: 0 []
- new safe wins: 0 []
- bounded timeouts: 0
- off-policy blocked (pre-execution): 73 | unsafe quarantined: 366

| theorem | category | class | success | killed | wall(s) |
|---|---|---|---|---|---|
| `Finset.Nonempty.inf_eq_bot_iff` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 6.68 |
| `Finset.Nonempty.strong_induction` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 4.41 |
| `Finset.Nonempty.sup_eq_top_iff` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 6.09 |
| `Finset.Nontrivial.erase_nonempty` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 8.66 |
| `Finset.card_mono` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 2.46 |
| `Finset.card_strictMono` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 3.05 |
| `Finset.card_union_eq_card_add_card` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 3.51 |
| `Finset.codisjoint_inf_left` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 5.62 |
| `Finset.codisjoint_inf_right` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 5.57 |
| `Finset.comp_inf_eq_inf_comp_of_is_total` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 6.17 |
| `Finset.comp_sup_eq_sup_comp_of_is_total` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 5.92 |
| `Finset.disjoint_biUnion_left` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 2.88 |
| `Finset.disjoint_biUnion_right` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 2.86 |
| `Finset.disjoint_filter_filter'` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 10.5 |
| `Finset.disjoint_image` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 4.0 |
| `Finset.disjoint_insert_right` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 7.4 |
| `Finset.disjoint_map` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 2.82 |
| `Finset.disjoint_sup_left` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 5.57 |
| `Finset.disjoint_sup_right` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 5.65 |
| `Finset.fin_mono` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 4.12 |
| `Finset.image_mono` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 3.48 |
| `Finset.max'_image` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 8.91 |
| `Finset.mem_fin` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 4.17 |
| `Finset.mem_insert` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 6.99 |
| `Finset.mem_singleton` | off_policy_cases | NO_WIN_SAFE_BUDGET | False | False | 6.05 |
| `Finset.min'_image` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 9.01 |
| `Finset.monotone_preimage` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 2.43 |
| `Finset.pairwise_cons'` | off_policy_cases | NO_WIN_SAFE_BUDGET | False | False | 12.15 |
| `Finset.powerset_card_disjiUnion` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 3.16 |
| `Finset.subtype_mono` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 4.02 |
| `List.Pairwise.pmap` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 2.41 |
| `List.Pairwise.set_pairwise` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 2.18 |
| `List.Sublist.antisymm` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 7.19 |
| `List.Sublist.map` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 10.44 |
| `List.Sublist.tail` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 7.02 |
| `List.append_cons_inj_of_not_mem` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 8.79 |
| `List.append_left_eq_self` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 5.78 |
| `List.append_right_eq_self` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 5.85 |
| `List.attach_eq_nil` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 9.81 |
| `List.disjoint_map` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 11.84 |
| `List.disjoint_pmap` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 12.35 |
| `List.perm_of_nodup_nodup_toFinset_eq` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 11.79 |
| `List.toFinset.ext_iff` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 11.76 |
| `Multiset.Disjoint.symm` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 11.42 |
| `Multiset.Nodup.le_nsmul_iff_le` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 2.54 |
| `Multiset.Rel.countP_eq` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 11.4 |
| `Multiset.Rel.mono` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 10.99 |
| `Multiset.Subset.refl` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 5.62 |
| `Multiset.Subset.trans` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 5.62 |
| `Multiset.addHom_ext` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 10.76 |
| `Multiset.add_cons` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 6.29 |
| `Multiset.add_eq_union_right_of_le` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 11.86 |
| `Multiset.add_product` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 3.0 |
| `Multiset.add_sigma` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 3.15 |
| `Multiset.add_singleton_eq_iff` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 7.1 |
| `Multiset.add_union_distrib` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 8.62 |
| `Multiset.attach_bind_coe` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 2.79 |
| `Multiset.attach_cons` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 7.99 |
| `Multiset.attach_map_val` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 7.92 |
| `Multiset.attach_map_val'` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 7.89 |
| `Multiset.disjoint_comm` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 11.33 |
| `Nat.AM_GM` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 5.77 |
| `Nat.add_div_of_dvd_left` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 3.51 |
| `Nat.add_div_of_dvd_right` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 3.39 |
| `Nat.add_le_mul` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 4.4 |
| `Nat.clog_anti_left` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 2.78 |
| `Nat.clog_eq_one` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 2.66 |
| `Nat.clog_mono_right` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 2.78 |
| `Nat.clog_of_left_le_one` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 2.58 |
| `Nat.div_ne_zero_iff` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 5.21 |
| `Nat.dvd_left_iff_eq` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 5.83 |
| `Nat.dvd_right_iff_eq` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 5.61 |
| `Nat.eq_div_of_mul_eq_left` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 4.6 |
| `Nat.eq_mul_of_div_eq_left` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 4.4 |
| `Nat.find_eq_zero` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 6.18 |
| `Nat.forall_lt_succ` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 3.45 |
| `Nat.le_of_mul_le_mul_right` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 4.16 |
| `Nat.mod_eq_iff_lt` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 5.21 |
| `Nat.mod_mul_mod` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 5.24 |
| `Nat.one_le_pow` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 4.61 |
| `Nat.sqrt_lt'` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 6.03 |
| `Nat.sqrt_pos` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 6.11 |
| `Nat.zero_eq_mul` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 4.08 |
| `Set.BijOn.congr` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 4.88 |
| `Set.disjoint_sUnion_right` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 9.36 |
| `Set.mapsTo_singleton` | prior_stall_cases | NO_WIN_SAFE_BUDGET | False | False | 4.13 |
| `Finset.biUnion_subset_iff_forall_subset` | true_winners | SAFE_TRUE_DYNAMIC_WIN | True | False | 2.45 |
| `Finset.image_subset_iff` | true_winners | SAFE_TRUE_DYNAMIC_WIN | True | False | 3.57 |
| `Multiset.add_bind` | true_winners | SAFE_TRUE_DYNAMIC_WIN | True | False | 2.26 |
