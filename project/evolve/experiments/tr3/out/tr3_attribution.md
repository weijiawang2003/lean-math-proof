# TR3 attribution

- targets: 92
- classification: {'PROOF_DEPTH_GAP': 79, 'TRUE_RETRIEVAL_DEPTH_DELTA': 3, 'TRUE_RETRIEVAL_ONLY_DELTA': 9, 'BASELINE_DUPLICATE': 1}
- **TRUE_DELTA total: 12** by class {'TRUE_RETRIEVAL_ONLY_DELTA': 9, 'TRUE_RETRIEVAL_DEPTH_DELTA': 3, 'TRUE_DEPTH_ONLY_DELTA': 0}
- every win over literal RC2: True

| target | class | program | depth | lemmas |
|---|---|---|---|---|
| `Set.Nonempty.subset_pair_iff_eq` | TRUE_RETRIEVAL_DEPTH_DELTA | `simp [Set.subset_pair_iff_eq] <;> aesop` | 2 | ['Set.subset_pair_iff_eq'] |
| `Set.antitoneOn_iff_antitone` | TRUE_RETRIEVAL_ONLY_DELTA | `simp [Antitone, AntitoneOn]` | 1 | ['Antitone', 'AntitoneOn'] |
| `Set.monotoneOn_iff_monotone` | TRUE_RETRIEVAL_ONLY_DELTA | `simp [Monotone, MonotoneOn]` | 1 | ['Monotone', 'MonotoneOn'] |
| `Set.strictAntiOn_iff_strictAnti` | TRUE_RETRIEVAL_ONLY_DELTA | `simp [StrictAnti, StrictAntiOn]` | 1 | ['StrictAnti', 'StrictAntiOn'] |
| `Set.strictMonoOn_iff_strictMono` | TRUE_RETRIEVAL_ONLY_DELTA | `simp [StrictMono, StrictMonoOn]` | 1 | ['StrictMono', 'StrictMonoOn'] |
| `Finset.mem_disjUnion` | TRUE_RETRIEVAL_ONLY_DELTA | `simp [Finset.disjUnion]` | 1 | ['Finset.disjUnion'] |
| `List.toFinset.ext_iff` | TRUE_RETRIEVAL_ONLY_DELTA | `simp [Finset.ext_iff]` | 1 | ['Finset.ext_iff'] |
| `List.toFinset_eq` | TRUE_RETRIEVAL_ONLY_DELTA | `simp [Multiset.toFinset_eq]` | 1 | ['Multiset.toFinset_eq'] |
| `Set.compl_union_self` | TRUE_RETRIEVAL_ONLY_DELTA | `simp [Set.union_eq_compl_compl_inter_compl]` | 1 | ['Set.union_eq_compl_compl_inter_compl'] |
| `Set.disjoint_iff_forall_ne` | TRUE_RETRIEVAL_DEPTH_DELTA | `simp [Set.disjoint_left] <;> aesop` | 2 | ['Set.disjoint_left'] |
| `Set.disjoint_right` | TRUE_RETRIEVAL_DEPTH_DELTA | `simp [Set.disjoint_left] <;> aesop` | 2 | ['Set.disjoint_left'] |
| `Set.disjoint_singleton_left` | TRUE_RETRIEVAL_ONLY_DELTA | `simp [Set.disjoint_left]` | 1 | ['Set.disjoint_left'] |

### Non-credited
| target | class | evidence |
|---|---|---|
| `Multiset.toFinset_eq_singleton_iff` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.diff_singleton_subset_iff` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.ite_eq_of_subset_left` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.ite_eq_of_subset_right` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.pair_eq_pair_iff` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.ssubset_iff_insert` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.ssubset_iff_sdiff_singleton` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.ssubset_singleton_iff` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.subset_insert_iff` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.subset_ite` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.subset_pair_iff_eq` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.subset_singleton_iff_eq` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.union_empty_iff` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Eq.subset` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Function.Injective.nonempty_apply_iff` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Prop.compl_singleton` | BASELINE_DUPLICATE | bare control solves: ['aesop'] |
| `Set.eq_of_inclusion_surjective` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.ite_inter_of_inter_eq` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.pairwiseDisjoint_filter` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.powerset_singleton` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.Nonempty.cons_induction` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.Nontrivial.exists_cons_eq` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.Nontrivial.sdiff_singleton_nonempty` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.cons_induction` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.disjoint_filter_filter'` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.eq_singleton_iff_nonempty_unique_mem` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.erase_inj` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.erase_nonempty` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.induction_on_union` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.insert_erase` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.inter_subset_inter` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.pairwise_cons'` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.range_filter_eq` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.sizeOf_lt_sizeOf_of_mem` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.ssubset_iff_exists_cons_subset` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.ssubset_iff_exists_subset_erase` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Finset.subset_union_elim` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `List.perm_of_nodup_nodup_toFinset_eq` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `List.toFinset_eq_empty_iff` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `List.toFinset_eq_iff_perm_dedup` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `List.toFinset_filter` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `List.toFinset_nonempty_iff` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `List.toFinset_surj_on` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Multiset.Nodup.toFinset_inj` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Multiset.toFinset_ssubset` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Multiset.toFinset_subset` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.add_sub_one_le_mul` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.diag_induction` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.div_div_div_eq_div` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.div_eq_iff_eq_of_dvd_dvd` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.div_eq_self` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.div_eq_sub_mod_div` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.div_le_of_le_mul'` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.div_le_self'` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.div_mul_div_comm` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.div_mul_div_le` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.div_mul_div_le_div` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.div_pow` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.dvd_sub'` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.eq_of_dvd_of_lt_two_mul` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.findGreatest_eq_iff` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.findGreatest_mono_left` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.findGreatest_mono_right` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.findGreatest_spec` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.find_add` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.find_eq_iff` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.leRecOn_injective` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.leRecOn_surjective` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.le_induction` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.not_dvd_of_pos_of_lt` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.not_two_dvd_bit1` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.one_lt_mul_iff` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.sqrt.iter_sq_le` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Nat.sqrt.lt_iter_succ_sq` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.Nonempty.eq_univ` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.diff_singleton_sSubset` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.insert_subset_insert_iff` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
| `Set.nonempty_compl_of_nontrivial` | PROOF_DEPTH_GAP | retrieval found lemmas but no depth<=3 program closed it |
