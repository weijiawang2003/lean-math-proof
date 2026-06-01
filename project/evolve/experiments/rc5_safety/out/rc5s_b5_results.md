# RC5S safe B5 results

- theorems: 89 | successes: **3** | killed_by_timeout (bounded): 0
- **no global stalls: True** | max wall: 12.3s (cap 60s) | off-policy: 0
- **RC5H winners reproduced: 3/3** ['Finset.biUnion_subset_iff_forall_subset', 'Finset.image_subset_iff', 'Multiset.add_bind']
- unknown-name: 28 | first-success ranks: {'1': 1, '2': 1, '5': 1}

| theorem | success | rank | wall(s) | killed | winning tactic |
|---|---|---|---|---|---|
| `Finset.biUnion_subset_iff_forall_subset` | True | 1 | 2.45 | False | `simp [Finset.biUnion_subset] <;> aesop` |
| `Finset.image_subset_iff` | True | 5 | 3.57 | False | `simp [Finset.subset_iff]` |
| `Multiset.add_bind` | True | 2 | 2.26 | False | `simp [Multiset.bind]` |
| `Finset.Nonempty.inf_eq_bot_iff` | False | None | 6.68 | False | `` |
| `Finset.Nonempty.strong_induction` | False | None | 4.41 | False | `` |
| `Finset.Nonempty.sup_eq_top_iff` | False | None | 6.09 | False | `` |
| `Finset.Nontrivial.erase_nonempty` | False | None | 8.66 | False | `` |
| `Finset.card_mono` | False | None | 2.46 | False | `` |
| `Finset.card_strictMono` | False | None | 3.05 | False | `` |
| `Finset.card_union_eq_card_add_card` | False | None | 3.51 | False | `` |
| `Finset.codisjoint_inf_left` | False | None | 5.62 | False | `` |
| `Finset.codisjoint_inf_right` | False | None | 5.57 | False | `` |
| `Finset.comp_inf_eq_inf_comp_of_is_total` | False | None | 6.17 | False | `` |
| `Finset.comp_sup_eq_sup_comp_of_is_total` | False | None | 5.92 | False | `` |
| `Finset.disjoint_biUnion_left` | False | None | 2.88 | False | `` |
| `Finset.disjoint_biUnion_right` | False | None | 2.86 | False | `` |
| `Finset.disjoint_filter_filter'` | False | None | 10.5 | False | `` |
| `Finset.disjoint_image` | False | None | 4.0 | False | `` |
| `Finset.disjoint_insert_right` | False | None | 7.4 | False | `` |
| `Finset.disjoint_map` | False | None | 2.82 | False | `` |
| `Finset.disjoint_sup_left` | False | None | 5.57 | False | `` |
| `Finset.disjoint_sup_right` | False | None | 5.65 | False | `` |
| `Finset.fin_mono` | False | None | 4.12 | False | `` |
| `Finset.image_mono` | False | None | 3.48 | False | `` |
| `Finset.max'_image` | False | None | 8.91 | False | `` |
| `Finset.mem_fin` | False | None | 4.17 | False | `` |
| `Finset.mem_insert` | False | None | 6.99 | False | `` |
| `Finset.mem_singleton` | False | None | 6.05 | False | `` |
| `Finset.min'_image` | False | None | 9.01 | False | `` |
| `Finset.monotone_preimage` | False | None | 2.43 | False | `` |
| `Finset.pairwise_cons'` | False | None | 12.15 | False | `` |
| `Finset.powerset_card_disjiUnion` | False | None | 3.16 | False | `` |
| `Finset.subtype_mono` | False | None | 4.02 | False | `` |
| `List.Pairwise.pmap` | False | None | 2.41 | False | `` |
| `List.Pairwise.set_pairwise` | False | None | 2.18 | False | `` |
| `List.Sublist.antisymm` | False | None | 7.19 | False | `` |
| `List.Sublist.map` | False | None | 10.44 | False | `` |
| `List.Sublist.tail` | False | None | 7.02 | False | `` |
| `List.append_cons_inj_of_not_mem` | False | None | 8.79 | False | `` |
| `List.append_left_eq_self` | False | None | 5.78 | False | `` |
| `List.append_right_eq_self` | False | None | 5.85 | False | `` |
| `List.attach_eq_nil` | False | None | 9.81 | False | `` |
| `List.disjoint_map` | False | None | 11.84 | False | `` |
| `List.disjoint_pmap` | False | None | 12.35 | False | `` |
| `List.perm_of_nodup_nodup_toFinset_eq` | False | None | 11.79 | False | `` |
| `List.toFinset.ext_iff` | False | None | 11.76 | False | `` |
| `Multiset.Disjoint.symm` | False | None | 11.42 | False | `` |
| `Multiset.Nodup.le_nsmul_iff_le` | False | None | 2.54 | False | `` |
| `Multiset.Rel.countP_eq` | False | None | 11.4 | False | `` |
| `Multiset.Rel.mono` | False | None | 10.99 | False | `` |
| `Multiset.Subset.refl` | False | None | 5.62 | False | `` |
| `Multiset.Subset.trans` | False | None | 5.62 | False | `` |
| `Multiset.addHom_ext` | False | None | 10.76 | False | `` |
| `Multiset.add_cons` | False | None | 6.29 | False | `` |
| `Multiset.add_eq_union_right_of_le` | False | None | 11.86 | False | `` |
| `Multiset.add_product` | False | None | 3.0 | False | `` |
| `Multiset.add_sigma` | False | None | 3.15 | False | `` |
| `Multiset.add_singleton_eq_iff` | False | None | 7.1 | False | `` |
| `Multiset.add_union_distrib` | False | None | 8.62 | False | `` |
| `Multiset.attach_bind_coe` | False | None | 2.79 | False | `` |
| `Multiset.attach_cons` | False | None | 7.99 | False | `` |
| `Multiset.attach_map_val` | False | None | 7.92 | False | `` |
| `Multiset.attach_map_val'` | False | None | 7.89 | False | `` |
| `Multiset.disjoint_comm` | False | None | 11.33 | False | `` |
| `Nat.AM_GM` | False | None | 5.77 | False | `` |
| `Nat.add_div_of_dvd_left` | False | None | 3.51 | False | `` |
| `Nat.add_div_of_dvd_right` | False | None | 3.39 | False | `` |
| `Nat.add_le_mul` | False | None | 4.4 | False | `` |
| `Nat.clog_anti_left` | False | None | 2.78 | False | `` |
| `Nat.clog_eq_one` | False | None | 2.66 | False | `` |
| `Nat.clog_mono_right` | False | None | 2.78 | False | `` |
| `Nat.clog_of_left_le_one` | False | None | 2.58 | False | `` |
| `Nat.div_ne_zero_iff` | False | None | 5.21 | False | `` |
| `Nat.dvd_left_iff_eq` | False | None | 5.83 | False | `` |
| `Nat.dvd_right_iff_eq` | False | None | 5.61 | False | `` |
| `Nat.eq_div_of_mul_eq_left` | False | None | 4.6 | False | `` |
| `Nat.eq_mul_of_div_eq_left` | False | None | 4.4 | False | `` |
| `Nat.find_eq_zero` | False | None | 6.18 | False | `` |
| `Nat.forall_lt_succ` | False | None | 3.45 | False | `` |
| `Nat.le_of_mul_le_mul_right` | False | None | 4.16 | False | `` |
| `Nat.mod_eq_iff_lt` | False | None | 5.21 | False | `` |
| `Nat.mod_mul_mod` | False | None | 5.24 | False | `` |
| `Nat.one_le_pow` | False | None | 4.61 | False | `` |
| `Nat.sqrt_lt'` | False | None | 6.03 | False | `` |
| `Nat.sqrt_pos` | False | None | 6.11 | False | `` |
| `Nat.zero_eq_mul` | False | None | 4.08 | False | `` |
| `Set.BijOn.congr` | False | None | 4.88 | False | `` |
| `Set.disjoint_sUnion_right` | False | None | 9.36 | False | `` |
| `Set.mapsTo_singleton` | False | None | 4.13 | False | `` |
