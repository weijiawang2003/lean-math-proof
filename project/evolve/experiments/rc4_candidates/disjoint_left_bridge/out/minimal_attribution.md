# RC4B minimal attribution

- new wins examined: 16
- classifications: {'TRUE_DISJOINT_LEFT_BRIDGE_WIN': 16}
- **TRUE_DISJOINT_LEFT_BRIDGE_WIN: 16** ['Multiset.disjoint_add_left', 'Multiset.disjoint_cons_left', 'Multiset.singleton_disjoint', 'Multiset.zero_disjoint', 'Set.disjoint_iUnion_left', 'Set.disjoint_iUnion_right', 'Set.disjoint_iff_forall_ne', 'Set.disjoint_right', 'Set.disjoint_sUnion_left', 'Set.disjoint_sUnion_right', 'Set.disjoint_singleton_left', 'Multiset.disjoint_add_right', 'Multiset.disjoint_iff_ne', 'Multiset.disjoint_right', 'Multiset.disjoint_singleton', 'Multiset.disjoint_union_left']
- Set true wins: 7 | Multiset true wins: 9
- fresh-holdout true wins: 5 | known-reproduction true wins: 11

| theorem | ns | bare_solved | bridge_solved | class |
|---|---|---|---|---|
| `Multiset.disjoint_add_left` | Multiset | [] | ['simp [Multiset.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| `Multiset.disjoint_cons_left` | Multiset | [] | ['simp [Multiset.disjoint_left]', 'simp [Multiset.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| `Multiset.singleton_disjoint` | Multiset | [] | ['simp [Multiset.disjoint_left]', 'simp [Multiset.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| `Multiset.zero_disjoint` | Multiset | [] | ['simp [Multiset.disjoint_left]', 'simp [Multiset.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| `Set.disjoint_iUnion_left` | Set | [] | ['simp [Set.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| `Set.disjoint_iUnion_right` | Set | [] | ['simp [Set.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| `Set.disjoint_iff_forall_ne` | Set | [] | ['simp [Set.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| `Set.disjoint_right` | Set | [] | ['simp [Set.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| `Set.disjoint_sUnion_left` | Set | [] | ['simp [Set.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| `Set.disjoint_sUnion_right` | Set | [] | ['simp [Set.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| `Set.disjoint_singleton_left` | Set | [] | ['simp [Set.disjoint_left]', 'simp [Set.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| `Multiset.disjoint_add_right` | Multiset | [] | ['simp [Multiset.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| `Multiset.disjoint_iff_ne` | Multiset | [] | ['simp [Multiset.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| `Multiset.disjoint_right` | Multiset | [] | ['simp [Multiset.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| `Multiset.disjoint_singleton` | Multiset | [] | ['simp [Multiset.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| `Multiset.disjoint_union_left` | Multiset | [] | ['simp [Multiset.disjoint_left] <;> aesop'] | TRUE_DISJOINT_LEFT_BRIDGE_WIN |
