# RC4C schema-native wrapper smoke

- wrapper: `project/evolve/experiments/rc4_candidates/d2_simp_aesop/rc4c_candidate_wrapper.json`
- known wins solved by wrapper: **0/12**
- regressions on negative controls: 0 []
- new wins observed: 0 []

| theorem | sets | rc2 | wrapper | win_tac |
|---|---|---|---|---|
| `Finset.biUnion_subset_iff_forall_subset` | known_wins_all | F | F | `` |
| `List.Forall.imp` | known_wins_all | F | F | `` |
| `Multiset.disjoint_add_left` | known_wins_all | F | F | `` |
| `Multiset.disjoint_add_right` | known_wins_all | F | F | `` |
| `Multiset.disjoint_cons_left` | known_wins_all | F | F | `` |
| `Set.Nonempty.subset_pair_iff_eq` | known_wins_all | F | F | `` |
| `Set.disjoint_iUnion_left` | known_wins_all | F | F | `` |
| `Set.disjoint_iUnion_right` | known_wins_all | F | F | `` |
| `Set.disjoint_iff_forall_ne` | known_wins_all | F | F | `` |
| `Set.disjoint_right` | known_wins_all | F | F | `` |
| `Set.disjoint_sUnion_left` | known_wins_all | F | F | `` |
| `Set.disjoint_sUnion_right` | known_wins_all | F | F | `` |
| `Nat.AM_GM` | namespace_negative_controls | F | F | `` |
| `Nat.Coprime.dvd_mul_left` | namespace_negative_controls | F | F | `` |
| `Nat.Coprime.dvd_mul_right` | namespace_negative_controls | F | F | `` |
| `Nat.Coprime.eq_of_mul_eq_zero` | namespace_negative_controls | S | S | `aesop` |
| `Nat.Coprime.lcm_eq_mul` | namespace_negative_controls | F | F | `` |
| `Nat.ModEq.add_le_of_lt` | namespace_negative_controls | F | F | `` |
| `Nat.ModEq.cancel_left_of_coprime` | namespace_negative_controls | F | F | `` |
| `Nat.ModEq.cancel_right_of_coprime` | namespace_negative_controls | F | F | `` |
| `Nat.ModEq.comm` | namespace_negative_controls | F | F | `` |
| `Nat.ModEq.dvd_iff` | namespace_negative_controls | F | F | `` |
| `Nat.ModEq.eq_of_abs_lt` | namespace_negative_controls | F | F | `` |
| `Nat.ModEq.eq_of_lt_of_lt` | namespace_negative_controls | F | F | `` |
| `Nat.ModEq.gcd_eq` | namespace_negative_controls | F | F | `` |
| `Nat.ModEq.le_of_lt_add` | namespace_negative_controls | F | F | `` |
| `Nat.ModEq.mul_left_cancel_iff'` | namespace_negative_controls | F | F | `` |
| `Nat.ModEq.mul_right_cancel_iff'` | namespace_negative_controls | F | F | `` |
| `Nat.add_def` | namespace_negative_controls | S | S | `simp [List.length_cons]` |
| `Nat.add_div` | namespace_negative_controls | F | F | `` |
| `Nat.add_div_eq_of_add_mod_lt` | namespace_negative_controls | F | F | `` |
| `Nat.add_div_eq_of_le_mod_add_mod` | namespace_negative_controls | F | F | `` |
| `Finset.card_union_eq_card_add_card` | negative_controls | F | F | `` |
| `Finset.codisjoint_inf_left` | negative_controls | F | F | `` |
| `Finset.codisjoint_inf_right` | negative_controls | F | F | `` |
| `Finset.disjiUnion_cons` | negative_controls | S | S | `aesop` |
| `Finset.disjiUnion_filter_eq` | negative_controls | S | S | `aesop` |
| `Finset.disjiUnion_filter_eq_of_maps_to` | negative_controls | S | S | `aesop` |
| `Finset.disjiUnion_map` | negative_controls | S | S | `aesop` |
| `Finset.disjoint_filter_filter'` | negative_controls | F | F | `` |
| `Finset.disjoint_image` | negative_controls | F | F | `` |
| `Finset.disjoint_map` | negative_controls | F | F | `` |
| `Finset.disjoint_sup_left` | negative_controls | F | F | `` |
| `Finset.disjoint_sup_right` | negative_controls | F | F | `` |
| `Finset.filter_cons` | negative_controls | S | S | `aesop` |
| `Finset.pairwise_cons'` | negative_controls | F | F | `` |
| `Finset.powerset_card_disjiUnion` | negative_controls | F | F | `` |
| `List.disjoint_map` | negative_controls | F | F | `` |
| `List.disjoint_pmap` | negative_controls | F | F | `` |
| `List.mem_pair` | negative_controls | S | S | `aesop` |
