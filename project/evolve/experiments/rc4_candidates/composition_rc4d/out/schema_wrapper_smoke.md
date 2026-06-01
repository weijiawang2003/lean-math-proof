# RC4D schema-native wrapper smoke

- wrapper: `project/evolve/experiments/rc4_candidates/composition_rc4d/rc4d_candidate_wrapper.json`
- known wins solved by wrapper: **22/23**
- wrapper new wins: 22 by_comp={'RC4A': 5, 'RC4B': 4}
- additive wins missed by wrapper: 1 ['Set.disjoint_sUnion_right']
- regressions: 0 []
- verdict: **SCHEMA_REPRODUCES**

| theorem | sets | rc2 | add_comp | wrapper | win_tac |
|---|---|---|---|---|---|
| `Finset.disjoint_insert_right` | canonical_smoke | F |  | F | `` |
| `Finset.mem_insert` | canonical_smoke | F |  | F | `` |
| `Finset.mem_singleton` | canonical_smoke | F |  | F | `` |
| `Nat.add_eq_left` | canonical_smoke | S |  | S | `exact ⟨fun h => by omega, fun h => by omega⟩` |
| `Nat.add_eq_max_iff` | canonical_smoke | S |  | S | `exact ⟨fun h => by omega, fun h => by omega⟩` |
| `Nat.add_eq_min_iff` | canonical_smoke | S |  | S | `exact ⟨fun h => by omega, fun h => by omega⟩` |
| `Nat.add_eq_one_iff` | canonical_smoke | S |  | S | `exact ⟨fun h => by omega, fun h => by omega⟩` |
| `Nat.add_eq_right` | canonical_smoke | S |  | S | `exact ⟨fun h => by omega, fun h => by omega⟩` |
| `Nat.add_mod_eq_add_mod_left` | canonical_smoke | S |  | S | `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]` |
| `Nat.add_mod_eq_add_mod_right` | canonical_smoke | S |  | S | `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]` |
| `Nat.add_mod_eq_ite` | canonical_smoke | S |  | S | `split_ifs <;> omega` |
| `Nat.div_le_div_right` | canonical_smoke | S |  | S | `by_cases hc : c = 0 <;> [simp [hc]; exact (Nat.le_div_iff_mul_le' (Nat.pos_of_ne_zero hc)).2 (Nat.le_trans (Nat.div_mul_le_self _ _) h)]` |
| `Nat.div_lt_iff_lt_mul'` | canonical_smoke | S |  | S | `simp_all` |
| `Nat.div_lt_one_iff` | canonical_smoke | S |  | S | `rw [Nat.div_lt_iff_lt_mul hb, Nat.one_mul]` |
| `Nat.half_le_of_sub_le_half` | canonical_smoke | S |  | S | `omega` |
| `Nat.le_and_le_add_one_iff` | canonical_smoke | S |  | S | `exact ⟨fun h => by omega, fun h => by omega⟩` |
| `Nat.le_or_le_of_add_eq_add_pred` | canonical_smoke | S |  | S | `omega` |
| `Nat.mul_add_mod'` | canonical_smoke | S |  | S | `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]` |
| `Set.empty_subset` | canonical_smoke | S |  | S | `simp [Set.subset_def]` |
| `Set.empty_union` | canonical_smoke | S |  | S | `simp [Set.ext_iff]` |
| `Set.inter_comm` | canonical_smoke | S |  | S | `aesop` |
| `Set.inter_univ` | canonical_smoke | S |  | S | `simp [Set.subset_def]` |
| `Set.ite_univ` | canonical_smoke | S |  | S | `simp [Set.ite]` |
| `Set.mem_inter_iff` | canonical_smoke | S |  | S | `tauto` |
| `Set.mem_union` | canonical_smoke | S |  | S | `aesop` |
| `Set.subset_univ` | canonical_smoke | S |  | S | `simp [Set.subset_def]` |
| `Set.union_comm` | canonical_smoke | S |  | S | `aesop` |
| `Set.union_empty` | canonical_smoke | S |  | S | `simp [Set.ext_iff]` |
| `Set.univ_inter` | canonical_smoke | S |  | S | `simp [Set.subset_def]` |
| `Nat.Coprime.dvd_mul_left` | namespace_negative_controls | F |  | F | `` |
| `Nat.Coprime.dvd_mul_right` | namespace_negative_controls | F |  | F | `` |
| `Nat.Coprime.eq_of_mul_eq_zero` | namespace_negative_controls | S |  | S | `aesop` |
| `Nat.Coprime.lcm_eq_mul` | namespace_negative_controls | F |  | F | `` |
| `Nat.ModEq.add_le_of_lt` | namespace_negative_controls | F |  | F | `` |
| `Nat.ModEq.cancel_left_of_coprime` | namespace_negative_controls | F |  | F | `` |
| `Nat.ModEq.cancel_right_of_coprime` | namespace_negative_controls | F |  | F | `` |
| `Nat.ModEq.comm` | namespace_negative_controls | F |  | F | `` |
| `Nat.ModEq.dvd_iff` | namespace_negative_controls | F |  | F | `` |
| `Nat.ModEq.eq_of_abs_lt` | namespace_negative_controls | F |  | F | `` |
| `Nat.ModEq.eq_of_lt_of_lt` | namespace_negative_controls | F |  | F | `` |
| `Nat.ModEq.gcd_eq` | namespace_negative_controls | F |  | F | `` |
| `Nat.ModEq.le_of_lt_add` | namespace_negative_controls | F |  | F | `` |
| `Nat.ModEq.mul_left_cancel_iff'` | namespace_negative_controls | F |  | F | `` |
| `Nat.ModEq.mul_right_cancel_iff'` | namespace_negative_controls | F |  | F | `` |
| `Nat.add_def` | namespace_negative_controls | S |  | S | `simp [List.length_cons]` |
| `Nat.add_div` | namespace_negative_controls | F |  | F | `` |
| `Nat.add_div_eq_of_add_mod_lt` | namespace_negative_controls | F |  | F | `` |
| `Nat.add_div_eq_of_le_mod_add_mod` | namespace_negative_controls | F |  | F | `` |
| `Nat.AM_GM` | namespace_negative_controls,canonical_smoke | F |  | F | `` |
| `Finset.card_union_eq_card_add_card` | negative_controls | F |  | F | `` |
| `Finset.codisjoint_inf_left` | negative_controls | F |  | F | `` |
| `Finset.codisjoint_inf_right` | negative_controls | F |  | F | `` |
| `Finset.disjiUnion_filter_eq` | negative_controls | S |  | S | `aesop` |
| `Finset.disjiUnion_filter_eq_of_maps_to` | negative_controls | S |  | S | `aesop` |
| `Finset.disjiUnion_map` | negative_controls | S |  | S | `aesop` |
| `Finset.disjoint_biUnion_left` | negative_controls | F |  | F | `` |
| `Finset.disjoint_biUnion_right` | negative_controls | F |  | F | `` |
| `Finset.disjoint_filter_filter'` | negative_controls | F |  | F | `` |
| `Finset.disjoint_image` | negative_controls | F |  | F | `` |
| `Finset.disjoint_map` | negative_controls | F |  | F | `` |
| `Finset.disjoint_sup_left` | negative_controls | F |  | F | `` |
| `Finset.disjoint_sup_right` | negative_controls | F |  | F | `` |
| `Finset.pairwise_cons'` | negative_controls | F |  | F | `` |
| `Finset.powerset_card_disjiUnion` | negative_controls | F |  | F | `` |
| `List.disjoint_map` | negative_controls | F |  | F | `` |
| `List.disjoint_pmap` | negative_controls | F |  | F | `` |
| `List.mem_pair` | negative_controls | S |  | S | `aesop` |
| `List.perm_of_nodup_nodup_toFinset_eq` | negative_controls | F |  | F | `` |
| `List.toFinset.ext_iff` | negative_controls | F |  | F | `` |
| `List.toFinset_eq` | negative_controls | F |  | F | `` |
| `List.toFinset_eq_empty_iff` | negative_controls | F |  | F | `` |
| `List.toFinset_eq_iff_perm_dedup` | negative_controls | F |  | F | `` |
| `List.toFinset_filter` | negative_controls | F |  | F | `` |
| `Finset.mem_disjUnion` | rc4a_known_wins | F | RC4A | S | `simp [Finset.disjUnion]` |
| `Set.antitoneOn_iff_antitone` | rc4a_known_wins | F | RC4A | S | `simp [Antitone, AntitoneOn]` |
| `Set.monotoneOn_iff_monotone` | rc4a_known_wins | F | RC4A | S | `simp [Monotone, MonotoneOn]` |
| `Set.strictAntiOn_iff_strictAnti` | rc4a_known_wins | F | RC4A | S | `simp [StrictAnti, StrictAntiOn]` |
| `Set.strictMonoOn_iff_strictMono` | rc4a_known_wins | F | RC4A | S | `simp [StrictMono, StrictMonoOn]` |
| `Multiset.disjoint_cons_left` | rc4b_known_wins | F | RC4B | S | `simp [Multiset.disjoint_left]` |
| `Multiset.disjoint_right` | rc4b_known_wins | F | RC4B | S | `aesop` |
| `Multiset.disjoint_singleton` | rc4b_known_wins | F | RC4B | S | `aesop` |
| `Multiset.zero_disjoint` | rc4b_known_wins | F | RC4B | S | `simp [Multiset.disjoint_left]` |
| `Set.disjoint_iUnion_left` | rc4b_known_wins | F | RC4B | S | `aesop` |
| `Set.disjoint_iUnion_right` | rc4b_known_wins | F | RC4B | S | `aesop` |
| `Set.disjoint_iff_forall_ne` | rc4b_known_wins | F | RC4B | S | `aesop` |
| `Set.disjoint_right` | rc4b_known_wins | F | RC4B | S | `aesop` |
| `Set.disjoint_sUnion_left` | rc4b_known_wins | F | RC4B | S | `aesop` |
| `Set.disjoint_sUnion_right` | rc4b_known_wins | F | RC4B | F | `` |
| `Set.disjoint_singleton_left` | rc4b_known_wins | F | RC4B | S | `simp [Set.disjoint_left]` |
| `Multiset.disjoint_add_left` | rc4b_known_wins,rc4c_residue_known_wins,component_overlap_controls | F | RC4B | S | `aesop` |
| `Multiset.disjoint_add_right` | rc4b_known_wins,rc4c_residue_known_wins,component_overlap_controls | F | RC4B | S | `aesop` |
| `Multiset.disjoint_iff_ne` | rc4b_known_wins,rc4c_residue_known_wins,component_overlap_controls | F | RC4B | S | `aesop` |
| `Multiset.disjoint_union_left` | rc4b_known_wins,rc4c_residue_known_wins,component_overlap_controls | F | RC4B | S | `aesop` |
| `Multiset.singleton_disjoint` | rc4b_known_wins,rc4c_residue_known_wins,component_overlap_controls | F | RC4B | S | `simp [Multiset.disjoint_left]` |
| `List.Forall.imp` | rc4c_residue_known_wins | F | RC4C_residue | S | `aesop` |
| `Set.Nonempty.subset_pair_iff_eq` | rc4c_residue_known_wins | F | RC4C_residue | S | `aesop` |
