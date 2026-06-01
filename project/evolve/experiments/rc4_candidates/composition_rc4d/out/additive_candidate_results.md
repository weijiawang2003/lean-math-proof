# RC4D additive composition evaluation (ordered)

- raw delta over RC2: **24**
- delta by component: {'RC4A': 5, 'RC4B': 16, 'RC4C_residue': 3}
- overlap eliminated (RC4C→RC4B): 9 ['Multiset.disjoint_add_left', 'Multiset.disjoint_add_right', 'Multiset.disjoint_cons_left', 'Multiset.disjoint_iff_ne', 'Multiset.disjoint_right', 'Multiset.disjoint_singleton', 'Multiset.disjoint_union_left', 'Multiset.singleton_disjoint', 'Multiset.zero_disjoint']
- off-gate emissions: 0 | regressions: 0
- emitted&solved 24 / emitted&failed 23

| theorem | sets | ns | rc2 | comps_fire | win_comp | win_tac | new |
|---|---|---|---|---|---|---|---|
| `Finset.mem_disjUnion` | rc4a_known_wins | Finset | F | RC4A | RC4A | `simp [Finset.disjUnion]` | True |
| `List.Forall.imp` | rc4c_residue_known_wins | List | F | RC4C_residue | RC4C_residue | `simp [List.forall_iff_forall_mem] <;> aesop` | True |
| `List.forall_map_iff` | composition_fresh_holdout | List | F | RC4C_residue | RC4C_residue | `simp [List.forall_iff_forall_mem]` | True |
| `Multiset.disjoint_add_left` | rc4b_known_wins,rc4c_residue_known_wins,component_overlap_controls | Multiset | F | RC4B,RC4C_residue | RC4B | `simp [Multiset.disjoint_left] <;> aesop` | True |
| `Multiset.disjoint_add_right` | rc4b_known_wins,rc4c_residue_known_wins,component_overlap_controls | Multiset | F | RC4B,RC4C_residue | RC4B | `simp [Multiset.disjoint_left] <;> aesop` | True |
| `Multiset.disjoint_cons_left` | rc4b_known_wins | Multiset | F | RC4B,RC4C_residue | RC4B | `simp [Multiset.disjoint_left]` | True |
| `Multiset.disjoint_iff_ne` | rc4b_known_wins,rc4c_residue_known_wins,component_overlap_controls | Multiset | F | RC4B,RC4C_residue | RC4B | `simp [Multiset.disjoint_left] <;> aesop` | True |
| `Multiset.disjoint_right` | rc4b_known_wins | Multiset | F | RC4B,RC4C_residue | RC4B | `simp [Multiset.disjoint_left] <;> aesop` | True |
| `Multiset.disjoint_singleton` | rc4b_known_wins | Multiset | F | RC4B,RC4C_residue | RC4B | `simp [Multiset.disjoint_left] <;> aesop` | True |
| `Multiset.disjoint_union_left` | rc4b_known_wins,rc4c_residue_known_wins,component_overlap_controls | Multiset | F | RC4B,RC4C_residue | RC4B | `simp [Multiset.disjoint_left] <;> aesop` | True |
| `Multiset.singleton_disjoint` | rc4b_known_wins,rc4c_residue_known_wins,component_overlap_controls | Multiset | F | RC4B,RC4C_residue | RC4B | `simp [Multiset.disjoint_left]` | True |
| `Multiset.zero_disjoint` | rc4b_known_wins | Multiset | F | RC4B,RC4C_residue | RC4B | `simp [Multiset.disjoint_left]` | True |
| `Set.Nonempty.subset_pair_iff_eq` | rc4c_residue_known_wins | Set | F | RC4C_residue | RC4C_residue | `simp [Set.subset_pair_iff_eq] <;> aesop` | True |
| `Set.antitoneOn_iff_antitone` | rc4a_known_wins | Set | F | RC4A | RC4A | `simp [Antitone, AntitoneOn]` | True |
| `Set.disjoint_iUnion_left` | rc4b_known_wins | Set | F | RC4B | RC4B | `simp [Set.disjoint_left] <;> aesop` | True |
| `Set.disjoint_iUnion_right` | rc4b_known_wins | Set | F | RC4B | RC4B | `simp [Set.disjoint_left] <;> aesop` | True |
| `Set.disjoint_iff_forall_ne` | rc4b_known_wins | Set | F | RC4B | RC4B | `simp [Set.disjoint_left] <;> aesop` | True |
| `Set.disjoint_right` | rc4b_known_wins | Set | F | RC4B | RC4B | `simp [Set.disjoint_left] <;> aesop` | True |
| `Set.disjoint_sUnion_left` | rc4b_known_wins | Set | F | RC4B | RC4B | `simp [Set.disjoint_left] <;> aesop` | True |
| `Set.disjoint_sUnion_right` | rc4b_known_wins | Set | F | RC4B | RC4B | `simp [Set.disjoint_left] <;> aesop` | True |
| `Set.disjoint_singleton_left` | rc4b_known_wins | Set | F | RC4B | RC4B | `simp [Set.disjoint_left]` | True |
| `Set.monotoneOn_iff_monotone` | rc4a_known_wins | Set | F | RC4A | RC4A | `simp [Monotone, MonotoneOn]` | True |
| `Set.strictAntiOn_iff_strictAnti` | rc4a_known_wins | Set | F | RC4A | RC4A | `simp [StrictAnti, StrictAntiOn]` | True |
| `Set.strictMonoOn_iff_strictMono` | rc4a_known_wins | Set | F | RC4A | RC4A | `simp [StrictMono, StrictMonoOn]` | True |
| `Finset.coe_disjUnion` | composition_fresh_holdout | Finset | S | RC4A |  | `` | False |
| `Finset.disjUnion_eq_union` | composition_fresh_holdout | Finset | S | RC4A |  | `` | False |
| `Finset.disjUnion_singleton` | composition_fresh_holdout | Finset | S | RC4A |  | `` | False |
| `Finset.disjiUnion_cons` | composition_fresh_holdout | Finset | S | RC4A |  | `` | False |
| `Finset.filter_cons` | composition_fresh_holdout | Finset | S | RC4A |  | `` | False |
| `List.filterMap_eq_map_iff_forall_eq_some` | composition_fresh_holdout | List | F | RC4C_residue |  | `` | False |
| `List.forall_cons` | composition_fresh_holdout | List | F | RC4C_residue |  | `` | False |
| `List.forall_iff_forall_mem` | composition_fresh_holdout | List | F | RC4C_residue |  | `` | False |
| `Multiset.Disjoint.symm` | composition_fresh_holdout | Multiset | F | RC4B,RC4C_residue |  | `` | False |
| `Multiset.add_eq_union_iff_disjoint` | composition_fresh_holdout | Multiset | F | RC4B,RC4C_residue |  | `` | False |
| `Multiset.add_eq_union_left_of_le` | composition_fresh_holdout | Multiset | F | RC4B,RC4C_residue |  | `` | False |
| `Multiset.add_eq_union_right_of_le` | composition_fresh_holdout | Multiset | F | RC4B,RC4C_residue |  | `` | False |
| `Multiset.coe_disjoint` | composition_fresh_holdout | Multiset | F | RC4B,RC4C_residue |  | `` | False |
| `Multiset.disjoint_comm` | composition_fresh_holdout | Multiset | F | RC4B,RC4C_residue |  | `` | False |
| `Multiset.disjoint_cons_right` | composition_fresh_holdout | Multiset | S | RC4B,RC4C_residue | RC4C_residue | `simp [Multiset.disjoint_right]` | False |
| `Multiset.disjoint_left` | composition_fresh_holdout | Multiset | F | RC4B,RC4C_residue |  | `` | False |
| `Multiset.disjoint_map_map` | composition_fresh_holdout | Multiset | S | RC4B,RC4C_residue | RC4B | `simp [Multiset.disjoint_left] <;> aesop` | False |
| `Multiset.disjoint_of_subset_left` | composition_fresh_holdout | Multiset | F | RC4B,RC4C_residue |  | `` | False |
| `Multiset.disjoint_of_subset_right` | composition_fresh_holdout | Multiset | F | RC4B,RC4C_residue |  | `` | False |
| `Multiset.disjoint_toFinset` | composition_fresh_holdout | Multiset | S | RC4B,RC4C_residue |  | `` | False |
| `Multiset.disjoint_union_right` | composition_fresh_holdout | Multiset | S | RC4B,RC4C_residue | RC4B | `simp [Multiset.disjoint_left] <;> aesop` | False |
| `Multiset.inter_eq_zero_iff_disjoint` | composition_fresh_holdout | Multiset | F | RC4B,RC4C_residue |  | `` | False |
| `Multiset.nodup_bind` | composition_fresh_holdout | Multiset | F | RC4B,RC4C_residue |  | `` | False |
| `Set._root_.Disjoint.image` | composition_fresh_holdout | Set | F | RC4B |  | `` | False |
| `Set.biUnion_compl_eq_of_pairwise_disjoint_of_iUnion_eq_univ` | composition_fresh_holdout | Set | F | RC4B,RC4C_residue |  | `` | False |
| `Set.disjoint_iUnion` | composition_fresh_holdout | Set | F | RC4B |  | `` | False |
| `Set.disjoint_singleton` | composition_fresh_holdout | Set | S | RC4B | RC4B | `simp [Set.disjoint_left]` | False |
| `Set.injOn_union` | composition_fresh_holdout | Set | F | RC4B |  | `` | False |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le` | composition_fresh_holdout | Set | F | RC4A |  | `` | False |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` | composition_fresh_holdout | Set | F | RC4A |  | `` | False |
| `Set.pairwiseDisjoint_filter` | composition_fresh_holdout | Set | F | RC4B,RC4C_residue |  | `` | False |
| `Set.sigmaToiUnion_bijective` | composition_fresh_holdout | Set | F | RC4B,RC4C_residue |  | `` | False |
| `Set.sigmaToiUnion_injective` | composition_fresh_holdout | Set | F | RC4B,RC4C_residue |  | `` | False |
