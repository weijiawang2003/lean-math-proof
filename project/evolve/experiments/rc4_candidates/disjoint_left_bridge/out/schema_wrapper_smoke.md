# RC4B schema-native wrapper smoke

- wrapper: `project/evolve/experiments/rc4_candidates/disjoint_left_bridge/rc4b_candidate_wrapper.json`
- known wins solved by wrapper: **10/11**
- regressions on negative controls: 0 []
- new wins observed: 10 ['Set.disjoint_singleton_left', 'Set.disjoint_right', 'Set.disjoint_iUnion_right', 'Multiset.singleton_disjoint', 'Multiset.disjoint_cons_left', 'Set.disjoint_sUnion_left', 'Multiset.disjoint_add_left', 'Set.disjoint_iff_forall_ne', 'Set.disjoint_iUnion_left', 'Multiset.zero_disjoint']

| theorem | sets | rc2 | wrapper | win_tac |
|---|---|---|---|---|
| `Finset.card_union_eq_card_add_card` | disjoint_negative_controls | F | F | `` |
| `Finset.codisjoint_inf_left` | disjoint_negative_controls | F | F | `` |
| `Finset.codisjoint_inf_right` | disjoint_negative_controls | F | F | `` |
| `Finset.disjiUnion_cons` | disjoint_negative_controls | S | S | `aesop` |
| `Finset.disjiUnion_filter_eq` | disjoint_negative_controls | S | S | `aesop` |
| `Finset.disjiUnion_filter_eq_of_maps_to` | disjoint_negative_controls | S | S | `aesop` |
| `Finset.disjiUnion_map` | disjoint_negative_controls | S | S | `aesop` |
| `Finset.disjoint_biUnion_left` | disjoint_negative_controls | F | F | `` |
| `Finset.disjoint_biUnion_right` | disjoint_negative_controls | F | F | `` |
| `Finset.disjoint_filter_filter'` | disjoint_negative_controls | F | F | `` |
| `Finset.disjoint_image` | disjoint_negative_controls | F | F | `` |
| `Finset.disjoint_map` | disjoint_negative_controls | F | F | `` |
| `Finset.disjoint_sup_left` | disjoint_negative_controls | F | F | `` |
| `Finset.disjoint_sup_right` | disjoint_negative_controls | F | F | `` |
| `Finset.filter_cons` | disjoint_negative_controls | S | S | `aesop` |
| `Multiset.disjoint_add_left` | known_wins | F | S | `aesop` |
| `Multiset.disjoint_cons_left` | known_wins | F | S | `simp [Multiset.disjoint_left]` |
| `Multiset.singleton_disjoint` | known_wins | F | S | `simp [Multiset.disjoint_left]` |
| `Multiset.zero_disjoint` | known_wins | F | S | `simp [Multiset.disjoint_left]` |
| `Set.disjoint_iUnion_left` | known_wins | F | S | `aesop` |
| `Set.disjoint_iUnion_right` | known_wins | F | S | `aesop` |
| `Set.disjoint_iff_forall_ne` | known_wins | F | S | `aesop` |
| `Set.disjoint_right` | known_wins | F | S | `aesop` |
| `Set.disjoint_sUnion_left` | known_wins | F | S | `aesop` |
| `Set.disjoint_sUnion_right` | known_wins | F | F | `` |
| `Set.disjoint_singleton_left` | known_wins | F | S | `simp [Set.disjoint_left]` |
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
