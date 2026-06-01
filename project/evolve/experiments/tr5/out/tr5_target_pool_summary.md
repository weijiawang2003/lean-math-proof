# TR5 target pool

- **92 targets** (deduped by full_name)
- by category: {'A_tr3_winner': 13, 'D_rc4c_d2aesop': 54, 'E_high_confidence': 8, 'F_high_uncertainty': 5, 'G_underrep_namespace': 12}
- by namespace: {'Finset': 18, 'List': 8, 'Prop': 1, 'Set': 31, 'Eq': 1, 'Multiset': 4, 'Nat': 28, 'Function': 1}
- by known RC2 status: {'failed': 92}
- known winners (TR3/RC4A): 13
- RC4B (Set.disjoint_left) targets: 3 → ['Set.disjoint_iff_forall_ne', 'Set.disjoint_right', 'Set.disjoint_singleton_left']
- RC4C (d2_simp_aesop) targets: 57 → ['Set.Nonempty.subset_pair_iff_eq', 'Set.disjoint_iff_forall_ne', 'Set.disjoint_right', 'Eq.subset', 'Finset.Nonempty.cons_induction', 'Finset.Nontrivial.exists_cons_eq', 'Finset.Nontrivial.sdiff_singleton_nonempty', 'Finset.cons_induction', "Finset.disjoint_filter_filter'", 'Finset.eq_singleton_iff_nonempty_unique_mem', 'Finset.erase_inj', 'Finset.erase_nonempty', 'Finset.inter_subset_inter', "Finset.pairwise_cons'", 'Finset.range_filter_eq', 'Finset.ssubset_iff_exists_cons_subset', 'Finset.ssubset_iff_exists_subset_erase', 'Finset.subset_union_elim', 'Multiset.Nodup.toFinset_inj', 'Multiset.toFinset_eq_singleton_iff', 'Multiset.toFinset_ssubset', 'Multiset.toFinset_subset', 'Nat.div_eq_self', 'Nat.div_eq_sub_mod_div', "Nat.div_le_of_le_mul'", "Nat.div_le_self'", 'Nat.div_mul_div_comm', 'Nat.div_mul_div_le', 'Nat.div_mul_div_le_div', 'Nat.div_pow', "Nat.dvd_sub'", 'Nat.eq_of_dvd_of_lt_two_mul', 'Nat.findGreatest_eq_iff', 'Nat.findGreatest_mono_left', 'Nat.findGreatest_spec', 'Nat.find_add', 'Nat.find_eq_iff', 'Nat.leRecOn_surjective', 'Nat.not_dvd_of_pos_of_lt', 'Nat.not_two_dvd_bit1', 'Nat.one_lt_mul_iff', 'Nat.sqrt.lt_iter_succ_sq', 'Set.Nonempty.eq_univ', 'Set.diff_singleton_subset_iff', 'Set.ite_eq_of_subset_left', 'Set.ite_eq_of_subset_right', 'Set.ite_inter_of_inter_eq', 'Set.nonempty_compl_of_nontrivial', 'Set.pair_eq_pair_iff', 'Set.pairwiseDisjoint_filter', 'Set.powerset_singleton', 'Set.ssubset_iff_insert', 'Set.ssubset_iff_sdiff_singleton', 'Set.subset_insert_iff', 'Set.subset_ite', 'Set.subset_pair_iff_eq', 'Set.subset_singleton_iff_eq']

## Top 20 by priority

| full_name | ns | category | rc2 | tr3 | ranker | tags |
|---|---|---|---|---|---|---|
| `Finset.mem_disjUnion` | Finset | A_tr3_winner | failed | WIN:def_unfold_simp | 0.0 | rc4a_def_unfold |
| `List.toFinset.ext_iff` | List | A_tr3_winner | failed | WIN:d1_simp_lemma | 0.0 |  |
| `List.toFinset_eq` | List | A_tr3_winner | failed | WIN:d1_simp_lemma | 0.0 |  |
| `Prop.compl_singleton` | Prop | A_tr3_winner | failed | WIN:d1_aesop | 0.0 |  |
| `Set.Nonempty.subset_pair_iff_eq` | Set | A_tr3_winner | failed | WIN:d2_simp_aesop | 0.0 | rc4c_d2_simp_aesop |
| `Set.antitoneOn_iff_antitone` | Set | A_tr3_winner | failed | WIN:def_unfold_simp | 0.9834 | rc4a_def_unfold |
| `Set.compl_union_self` | Set | A_tr3_winner | failed | WIN:d1_simp_lemma | 0.0 |  |
| `Set.disjoint_iff_forall_ne` | Set | A_tr3_winner | failed | WIN:d2_simp_aesop | 0.0911 | rc4b_set_disjoint_left,rc4c_d2_simp_aesop |
| `Set.disjoint_right` | Set | A_tr3_winner | failed | WIN:d2_simp_aesop | 0.1465 | rc4b_set_disjoint_left,rc4c_d2_simp_aesop |
| `Set.disjoint_singleton_left` | Set | A_tr3_winner | failed | WIN:d1_simp_lemma | 0.0044 | rc4b_set_disjoint_left |
| `Set.monotoneOn_iff_monotone` | Set | A_tr3_winner | failed | WIN:def_unfold_simp | 0.9574 | rc4a_def_unfold |
| `Set.strictAntiOn_iff_strictAnti` | Set | A_tr3_winner | failed | WIN:def_unfold_simp | 0.9983 | rc4a_def_unfold |
| `Set.strictMonoOn_iff_strictMono` | Set | A_tr3_winner | failed | WIN:def_unfold_simp | 0.995 | rc4a_def_unfold |
| `Eq.subset` | Eq | D_rc4c_d2aesop | failed | PROOF_DEPTH_GAP | 0.0003 | rc4c_d2_simp_aesop |
| `Finset.Nonempty.cons_induction` | Finset | D_rc4c_d2aesop | failed | PROOF_DEPTH_GAP | 0.0012 | rc4c_d2_simp_aesop |
| `Finset.Nontrivial.exists_cons_eq` | Finset | D_rc4c_d2aesop | failed | PROOF_DEPTH_GAP | 0.0003 | rc4c_d2_simp_aesop |
| `Finset.Nontrivial.sdiff_singleton_nonempty` | Finset | D_rc4c_d2aesop | failed | PROOF_DEPTH_GAP | 0.0036 | rc4c_d2_simp_aesop |
| `Finset.cons_induction` | Finset | D_rc4c_d2aesop | failed | PROOF_DEPTH_GAP | 0.0 | rc4c_d2_simp_aesop |
| `Finset.disjoint_filter_filter'` | Finset | D_rc4c_d2aesop | failed | PROOF_DEPTH_GAP | 0.0 | rc4c_d2_simp_aesop |
| `Finset.eq_singleton_iff_nonempty_unique_mem` | Finset | D_rc4c_d2aesop | failed | PROOF_DEPTH_GAP | 0.0028 | rc4c_d2_simp_aesop |
