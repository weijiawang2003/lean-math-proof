# TR5 B5 live results

- theorems: 92 | live: 91 | successes: **12** | setup errors: 1
- first-success rank histogram: {1: 12}

| theorem | ns | attempted | success | first_rank | winning tactic |
|---|---|---|---|---|---|
| `List.toFinset.ext_iff` | List | 1 | True | 1 | `simp [Finset.ext_iff]` |
| `List.toFinset_eq` | List | 1 | True | 1 | `simp [Multiset.toFinset_eq]` |
| `Prop.compl_singleton` | Prop | 1 | True | 1 | `aesop` |
| `Set.Nonempty.subset_pair_iff_eq` | Set | 1 | True | 1 | `simp [Set.subset_pair_iff_eq] <;> aesop` |
| `Set.antitoneOn_iff_antitone` | Set | 1 | True | 1 | `simp [Antitone, AntitoneOn]` |
| `Set.compl_union_self` | Set | 1 | True | 1 | `simp [Set.union_eq_compl_compl_inter_compl]` |
| `Set.disjoint_iff_forall_ne` | Set | 1 | True | 1 | `simp [Set.disjoint_left] <;> aesop` |
| `Set.disjoint_right` | Set | 1 | True | 1 | `simp [Set.disjoint_left] <;> aesop` |
| `Set.disjoint_singleton_left` | Set | 1 | True | 1 | `simp [Set.disjoint_left]` |
| `Set.monotoneOn_iff_monotone` | Set | 1 | True | 1 | `simp [Monotone, MonotoneOn]` |
| `Set.strictAntiOn_iff_strictAnti` | Set | 1 | True | 1 | `simp [StrictAnti, StrictAntiOn]` |
| `Set.strictMonoOn_iff_strictMono` | Set | 1 | True | 1 | `simp [StrictMono, StrictMonoOn]` |
| `Eq.subset` | Eq | 5 | False | None | `` |
| `Finset.Nonempty.cons_induction` | Finset | 5 | False | None | `` |
| `Finset.Nontrivial.exists_cons_eq` | Finset | 5 | False | None | `` |
| `Finset.Nontrivial.sdiff_singleton_nonempty` | Finset | 5 | False | None | `` |
| `Finset.cons_induction` | Finset | 5 | False | None | `` |
| `Finset.disjoint_filter_filter'` | Finset | 5 | False | None | `` |
| `Finset.eq_singleton_iff_nonempty_unique_mem` | Finset | 5 | False | None | `` |
| `Finset.erase_inj` | Finset | 5 | False | None | `` |
| `Finset.erase_nonempty` | Finset | 5 | False | None | `` |
| `Finset.induction_on_union` | Finset | 5 | False | None | `` |
| `Finset.insert_erase` | Finset | 5 | False | None | `` |
| `Finset.inter_subset_inter` | Finset | 5 | False | None | `` |
| `Finset.mem_disjUnion` | Finset | 5 | False | None | `` |
| `Finset.pairwise_cons'` | Finset | 5 | False | None | `` |
| `Finset.range_filter_eq` | Finset | 5 | False | None | `` |
| `Finset.sizeOf_lt_sizeOf_of_mem` | Finset | 5 | False | None | `` |
| `Finset.ssubset_iff_exists_cons_subset` | Finset | 5 | False | None | `` |
| `Finset.ssubset_iff_exists_subset_erase` | Finset | 5 | False | None | `` |
| `Finset.subset_union_elim` | Finset | 5 | False | None | `` |
| `Function.Injective.nonempty_apply_iff` | Function | 5 | False | None | `` |
| `List.perm_of_nodup_nodup_toFinset_eq` | List | 5 | False | None | `` |
| `List.toFinset_eq_empty_iff` | List | 5 | False | None | `` |
| `List.toFinset_eq_iff_perm_dedup` | List | 5 | False | None | `` |
| `List.toFinset_filter` | List | 5 | False | None | `` |
| `List.toFinset_nonempty_iff` | List | 5 | False | None | `` |
| `List.toFinset_surj_on` | List | 5 | False | None | `` |
| `Multiset.Nodup.toFinset_inj` | Multiset | 5 | False | None | `` |
| `Multiset.toFinset_eq_singleton_iff` | Multiset | 5 | False | None | `` |
| `Multiset.toFinset_ssubset` | Multiset | 5 | False | None | `` |
| `Multiset.toFinset_subset` | Multiset | 5 | False | None | `` |
| `Nat.add_sub_one_le_mul` | Nat | 5 | False | None | `` |
| `Nat.diag_induction` | Nat | 5 | False | None | `` |
| `Nat.div_div_div_eq_div` | Nat | 5 | False | None | `` |
| `Nat.div_eq_iff_eq_of_dvd_dvd` | Nat | 5 | False | None | `` |
| `Nat.div_eq_self` | Nat | 5 | False | None | `` |
| `Nat.div_eq_sub_mod_div` | Nat | 5 | False | None | `` |
| `Nat.div_le_of_le_mul'` | Nat | 5 | False | None | `` |
| `Nat.div_le_self'` | Nat | 5 | False | None | `` |
| `Nat.div_mul_div_comm` | Nat | 5 | False | None | `` |
| `Nat.div_mul_div_le` | Nat | 5 | False | None | `` |
| `Nat.div_mul_div_le_div` | Nat | 5 | False | None | `` |
| `Nat.div_pow` | Nat | 5 | False | None | `` |
| `Nat.dvd_sub'` | Nat | 5 | False | None | `` |
| `Nat.eq_of_dvd_of_lt_two_mul` | Nat | 5 | False | None | `` |
| `Nat.findGreatest_eq_iff` | Nat | 5 | False | None | `` |
| `Nat.findGreatest_mono_left` | Nat | 5 | False | None | `` |
| `Nat.findGreatest_mono_right` | Nat | 5 | False | None | `` |
| `Nat.findGreatest_spec` | Nat | 5 | False | None | `` |
| `Nat.find_add` | Nat | 5 | False | None | `` |
| `Nat.find_eq_iff` | Nat | 5 | False | None | `` |
| `Nat.leRecOn_injective` | Nat | 5 | False | None | `` |
| `Nat.leRecOn_surjective` | Nat | 5 | False | None | `` |
| `Nat.le_induction` | Nat | 5 | False | None | `` |
| `Nat.not_dvd_of_pos_of_lt` | Nat | 5 | False | None | `` |
| `Nat.not_two_dvd_bit1` | Nat | 5 | False | None | `` |
| `Nat.one_lt_mul_iff` | Nat | 5 | False | None | `` |
| `Nat.sqrt.iter_sq_le` | Nat | 5 | False | None | `` |
| `Nat.sqrt.lt_iter_succ_sq` | Nat | 5 | False | None | `` |
| `Set.Nonempty.eq_univ` | Set | 5 | False | None | `` |
| `Set.diff_singleton_sSubset` | Set | 5 | False | None | `` |
| `Set.diff_singleton_subset_iff` | Set | 5 | False | None | `` |
| `Set.eq_of_inclusion_surjective` | Set | 5 | False | None | `` |
| `Set.insert_subset_insert_iff` | Set | 5 | False | None | `` |
| `Set.ite_eq_of_subset_left` | Set | 5 | False | None | `` |
| `Set.ite_eq_of_subset_right` | Set | 5 | False | None | `` |
| `Set.ite_inter_of_inter_eq` | Set | 5 | False | None | `` |
| `Set.nonempty_compl_of_nontrivial` | Set | 5 | False | None | `` |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le` | Set | 5 | False | None | `` |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` | Set | 5 | False | None | `` |
| `Set.pair_eq_pair_iff` | Set | 5 | False | None | `` |
| `Set.pairwiseDisjoint_filter` | Set | 5 | False | None | `` |
| `Set.powerset_singleton` | Set | 5 | False | None | `` |
| `Set.ssubset_iff_insert` | Set | 5 | False | None | `` |
| `Set.ssubset_iff_sdiff_singleton` | Set | 5 | False | None | `` |
| `Set.ssubset_singleton_iff` | Set | 0 | False | None | `` |
| `Set.subset_insert_iff` | Set | 5 | False | None | `` |
| `Set.subset_ite` | Set | 5 | False | None | `` |
| `Set.subset_pair_iff_eq` | Set | 5 | False | None | `` |
| `Set.subset_singleton_iff_eq` | Set | 5 | False | None | `` |
| `Set.union_empty_iff` | Set | 5 | False | None | `` |
