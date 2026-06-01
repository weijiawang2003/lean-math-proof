# TR3 depth-program results

- theorems: 92 | live: 92
- pre-attribution: {'candidate_win': 12, 'no_win': 79, 'baseline_duplicate': 1, 'needs_review': 0}

| target | live | progs | wins | control_wins | best |
|---|---|---|---|---|---|
| `Multiset.toFinset_eq_singleton_iff` | True | 54/54 | 0 | 0 | `` |
| `Set.diff_singleton_subset_iff` | True | 47/47 | 0 | 0 | `` |
| `Set.ite_eq_of_subset_left` | True | 51/51 | 0 | 0 | `` |
| `Set.ite_eq_of_subset_right` | True | 51/51 | 0 | 0 | `` |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le` | True | 53/53 | 0 | 0 | `` |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` | True | 53/53 | 0 | 0 | `` |
| `Set.pair_eq_pair_iff` | True | 52/52 | 0 | 0 | `` |
| `Set.ssubset_iff_insert` | True | 50/50 | 0 | 0 | `` |
| `Set.ssubset_iff_sdiff_singleton` | True | 51/51 | 0 | 0 | `` |
| `Set.ssubset_singleton_iff` | True | 51/51 | 0 | 0 | `` |
| `Set.subset_insert_iff` | True | 49/49 | 0 | 0 | `` |
| `Set.subset_ite` | True | 47/47 | 0 | 0 | `` |
| `Set.subset_pair_iff_eq` | True | 52/52 | 0 | 0 | `` |
| `Set.subset_singleton_iff_eq` | True | 52/52 | 0 | 0 | `` |
| `Set.union_empty_iff` | True | 52/52 | 0 | 0 | `` |
| `Set.Nonempty.subset_pair_iff_eq` | True | 36/52 | 1 | 0 | `simp [Set.subset_pair_iff_eq] <;> aesop` |
| `Set.antitoneOn_iff_antitone` | True | 1/49 | 1 | 0 | `simp [Antitone, AntitoneOn]` |
| `Set.monotoneOn_iff_monotone` | True | 1/48 | 1 | 0 | `simp [Monotone, MonotoneOn]` |
| `Set.strictAntiOn_iff_strictAnti` | True | 1/53 | 1 | 0 | `simp [StrictAnti, StrictAntiOn]` |
| `Set.strictMonoOn_iff_strictMono` | True | 1/45 | 1 | 0 | `simp [StrictMono, StrictMonoOn]` |
| `Eq.subset` | True | 44/44 | 0 | 0 | `` |
| `Function.Injective.nonempty_apply_iff` | True | 44/44 | 0 | 0 | `` |
| `Prop.compl_singleton` | True | 31/47 | 1 | 1 | `aesop` |
| `Set.eq_of_inclusion_surjective` | True | 55/55 | 0 | 0 | `` |
| `Set.ite_inter_of_inter_eq` | True | 54/54 | 0 | 0 | `` |
| `Set.pairwiseDisjoint_filter` | True | 35/35 | 0 | 0 | `` |
| `Set.powerset_singleton` | True | 48/48 | 0 | 0 | `` |
| `Finset.Nonempty.cons_induction` | True | 36/36 | 0 | 0 | `` |
| `Finset.Nontrivial.exists_cons_eq` | True | 40/40 | 0 | 0 | `` |
| `Finset.Nontrivial.sdiff_singleton_nonempty` | True | 41/41 | 0 | 0 | `` |
| `Finset.cons_induction` | True | 36/36 | 0 | 0 | `` |
| `Finset.disjoint_filter_filter'` | True | 40/40 | 0 | 0 | `` |
| `Finset.eq_singleton_iff_nonempty_unique_mem` | True | 52/52 | 0 | 0 | `` |
| `Finset.erase_inj` | True | 52/52 | 0 | 0 | `` |
| `Finset.erase_nonempty` | True | 50/50 | 0 | 0 | `` |
| `Finset.induction_on_union` | True | 40/40 | 0 | 0 | `` |
| `Finset.insert_erase` | True | 45/45 | 0 | 0 | `` |
| `Finset.inter_subset_inter` | True | 35/35 | 0 | 0 | `` |
| `Finset.mem_disjUnion` | True | 1/52 | 1 | 0 | `simp [Finset.disjUnion]` |
| `Finset.pairwise_cons'` | True | 39/39 | 0 | 0 | `` |
| `Finset.range_filter_eq` | True | 48/48 | 0 | 0 | `` |
| `Finset.sizeOf_lt_sizeOf_of_mem` | True | 41/41 | 0 | 0 | `` |
| `Finset.ssubset_iff_exists_cons_subset` | True | 51/51 | 0 | 0 | `` |
| `Finset.ssubset_iff_exists_subset_erase` | True | 52/52 | 0 | 0 | `` |
| `Finset.subset_union_elim` | True | 45/45 | 0 | 0 | `` |
| `List.perm_of_nodup_nodup_toFinset_eq` | True | 55/55 | 0 | 0 | `` |
| `List.toFinset.ext_iff` | True | 8/52 | 1 | 0 | `simp [Finset.ext_iff]` |
| `List.toFinset_eq` | True | 4/56 | 1 | 0 | `simp [Multiset.toFinset_eq]` |
| `List.toFinset_eq_empty_iff` | True | 54/54 | 0 | 0 | `` |
| `List.toFinset_eq_iff_perm_dedup` | True | 54/54 | 0 | 0 | `` |
| `List.toFinset_filter` | True | 55/55 | 0 | 0 | `` |
| `List.toFinset_nonempty_iff` | True | 54/54 | 0 | 0 | `` |
| `List.toFinset_surj_on` | True | 49/49 | 0 | 0 | `` |
| `Multiset.Nodup.toFinset_inj` | True | 47/47 | 0 | 0 | `` |
| `Multiset.toFinset_ssubset` | True | 53/53 | 0 | 0 | `` |
| `Multiset.toFinset_subset` | True | 54/54 | 0 | 0 | `` |
| `Nat.add_sub_one_le_mul` | True | 49/49 | 0 | 0 | `` |
| `Nat.diag_induction` | True | 49/49 | 0 | 0 | `` |
| `Nat.div_div_div_eq_div` | True | 48/48 | 0 | 0 | `` |
| `Nat.div_eq_iff_eq_of_dvd_dvd` | True | 54/54 | 0 | 0 | `` |
| `Nat.div_eq_self` | True | 54/54 | 0 | 0 | `` |
| `Nat.div_eq_sub_mod_div` | True | 45/45 | 0 | 0 | `` |
| `Nat.div_le_of_le_mul'` | True | 44/44 | 0 | 0 | `` |
| `Nat.div_le_self'` | True | 40/40 | 0 | 0 | `` |
| `Nat.div_mul_div_comm` | True | 41/41 | 0 | 0 | `` |
| `Nat.div_mul_div_le` | True | 38/38 | 0 | 0 | `` |
| `Nat.div_mul_div_le_div` | True | 40/40 | 0 | 0 | `` |
| `Nat.div_pow` | True | 42/42 | 0 | 0 | `` |
| `Nat.dvd_sub'` | True | 42/42 | 0 | 0 | `` |
| `Nat.eq_of_dvd_of_lt_two_mul` | True | 41/41 | 0 | 0 | `` |
| `Nat.findGreatest_eq_iff` | True | 55/55 | 0 | 0 | `` |
| `Nat.findGreatest_mono_left` | True | 40/40 | 0 | 0 | `` |
| `Nat.findGreatest_mono_right` | True | 41/41 | 0 | 0 | `` |
| `Nat.findGreatest_spec` | True | 44/44 | 0 | 0 | `` |
| `Nat.find_add` | True | 47/47 | 0 | 0 | `` |
| `Nat.find_eq_iff` | True | 55/55 | 0 | 0 | `` |
| `Nat.leRecOn_injective` | True | 47/47 | 0 | 0 | `` |
| `Nat.leRecOn_surjective` | True | 47/47 | 0 | 0 | `` |
| `Nat.le_induction` | True | 45/45 | 0 | 0 | `` |
| `Nat.not_dvd_of_pos_of_lt` | True | 39/39 | 0 | 0 | `` |
| `Nat.not_two_dvd_bit1` | True | 46/46 | 0 | 0 | `` |
| `Nat.one_lt_mul_iff` | True | 54/54 | 0 | 0 | `` |
| `Nat.sqrt.iter_sq_le` | True | 38/38 | 0 | 0 | `` |
| `Nat.sqrt.lt_iter_succ_sq` | True | 38/38 | 0 | 0 | `` |
| `Set.Nonempty.eq_univ` | True | 41/41 | 0 | 0 | `` |
| `Set.compl_union_self` | True | 23/54 | 1 | 0 | `simp [Set.union_eq_compl_compl_inter_compl]` |
| `Set.diff_singleton_sSubset` | True | 48/48 | 0 | 0 | `` |
| `Set.disjoint_iff_forall_ne` | True | 42/53 | 1 | 0 | `simp [Set.disjoint_left] <;> aesop` |
| `Set.disjoint_right` | True | 46/52 | 1 | 0 | `simp [Set.disjoint_left] <;> aesop` |
| `Set.disjoint_singleton_left` | True | 19/52 | 1 | 0 | `simp [Set.disjoint_left]` |
| `Set.insert_subset_insert_iff` | True | 52/52 | 0 | 0 | `` |
| `Set.nonempty_compl_of_nontrivial` | True | 45/45 | 0 | 0 | `` |
