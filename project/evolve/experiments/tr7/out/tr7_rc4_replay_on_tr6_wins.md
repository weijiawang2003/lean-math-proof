# TR7 RC4 replay on TR6 fresh wins

- wins replayed: 18 | classification: {'RC4_MISSES_GATE': 4, 'RC4_REPRODUCES_TR6_WIN': 10, 'RC4_ACTION_FAILS_BUT_TR6_PROGRAM_WORKS': 4}
- RC4 reproduces: **10/18**

| theorem | ns | rc4_wrapper | gate | tr6_prog_works | rc4_action_solves | class |
|---|---|---|---|---|---|---|
| `Finset.biUnion_subset_iff_forall_subset` | Finset | failed | False | True | None | RC4_MISSES_GATE |
| `Finset.image_subset_iff` | Finset | failed | False | True | None | RC4_MISSES_GATE |
| `List.Forall.imp` | List | solved | True | True | True | RC4_REPRODUCES_TR6_WIN |
| `Multiset.Disjoint.symm` | Multiset | failed | True | True | False | RC4_ACTION_FAILS_BUT_TR6_PROGRAM_WORKS |
| `Multiset.add_eq_union_right_of_le` | Multiset | failed | True | True | False | RC4_ACTION_FAILS_BUT_TR6_PROGRAM_WORKS |
| `Multiset.disjoint_add_left` | Multiset | solved | True | True | True | RC4_REPRODUCES_TR6_WIN |
| `Multiset.disjoint_add_right` | Multiset | solved | True | True | True | RC4_REPRODUCES_TR6_WIN |
| `Multiset.disjoint_comm` | Multiset | failed | True | True | False | RC4_ACTION_FAILS_BUT_TR6_PROGRAM_WORKS |
| `Multiset.disjoint_cons_left` | Multiset | solved | True | True | True | RC4_REPRODUCES_TR6_WIN |
| `Multiset.disjoint_right` | Multiset | solved | True | True | True | RC4_REPRODUCES_TR6_WIN |
| `Multiset.singleton_disjoint` | Multiset | solved | True | True | True | RC4_REPRODUCES_TR6_WIN |
| `Multiset.zero_disjoint` | Multiset | solved | True | True | True | RC4_REPRODUCES_TR6_WIN |
| `Nat.sqrt_pos` | Nat | failed | False | True | None | RC4_MISSES_GATE |
| `Set.disjoint_iUnion_left` | Set | solved | True | True | True | RC4_REPRODUCES_TR6_WIN |
| `Set.disjoint_iUnion_right` | Set | solved | True | True | True | RC4_REPRODUCES_TR6_WIN |
| `Set.disjoint_sUnion_left` | Set | solved | True | True | True | RC4_REPRODUCES_TR6_WIN |
| `Set.disjoint_sUnion_right` | Set | failed | True | True | True | RC4_ACTION_FAILS_BUT_TR6_PROGRAM_WORKS |
| `Set.mapsTo_singleton` | Set | failed | False | True | None | RC4_MISSES_GATE |
