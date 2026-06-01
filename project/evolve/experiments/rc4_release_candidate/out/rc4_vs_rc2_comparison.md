# RC4 vs RC2 comparison

- RC2 solved: **78** | RC4 solved: **100** | raw delta: **22**
- new wins: **22** | regressions: **0** | net delta: **22**
- new wins by component: {'RC4A': 5, 'RC4C_residue': 2, 'RC4B': 15}
- new wins by namespace: {'Finset': 1, 'List': 1, 'Multiset': 9, 'Set': 11}
- known-win reproductions: 22 | fresh new wins: 0 []
- regressions: []
- classification: {'BOTH_FAILED': 159, 'BOTH_SOLVED': 78, 'FLAKE': 12, 'RC4_NEW_WIN': 22}

## RC4 new wins

| theorem | ns | set | component | rc4_tactic |
|---|---|---|---|---|
| `Finset.mem_disjUnion` | Finset | known | RC4A | `simp [Finset.disjUnion]` |
| `List.Forall.imp` | List | known | RC4C_residue | `aesop` |
| `Multiset.disjoint_add_left` | Multiset | known | RC4B | `aesop` |
| `Multiset.disjoint_add_right` | Multiset | known | RC4B | `aesop` |
| `Multiset.disjoint_cons_left` | Multiset | known | RC4B | `simp [Multiset.disjoint_left]` |
| `Multiset.disjoint_iff_ne` | Multiset | known | RC4B | `aesop` |
| `Multiset.disjoint_right` | Multiset | known | RC4B | `aesop` |
| `Multiset.disjoint_singleton` | Multiset | known | RC4B | `aesop` |
| `Multiset.disjoint_union_left` | Multiset | known | RC4B | `aesop` |
| `Multiset.singleton_disjoint` | Multiset | known | RC4B | `simp [Multiset.disjoint_left]` |
| `Multiset.zero_disjoint` | Multiset | known | RC4B | `simp [Multiset.disjoint_left]` |
| `Set.Nonempty.subset_pair_iff_eq` | Set | known | RC4C_residue | `aesop` |
| `Set.antitoneOn_iff_antitone` | Set | known | RC4A | `simp [Antitone, AntitoneOn]` |
| `Set.disjoint_iUnion_left` | Set | known | RC4B | `aesop` |
| `Set.disjoint_iUnion_right` | Set | known | RC4B | `aesop` |
| `Set.disjoint_iff_forall_ne` | Set | known | RC4B | `aesop` |
| `Set.disjoint_right` | Set | known | RC4B | `aesop` |
| `Set.disjoint_sUnion_left` | Set | known | RC4B | `aesop` |
| `Set.disjoint_singleton_left` | Set | known | RC4B | `simp [Set.disjoint_left]` |
| `Set.monotoneOn_iff_monotone` | Set | known | RC4A | `simp [Monotone, MonotoneOn]` |
| `Set.strictAntiOn_iff_strictAnti` | Set | known | RC4A | `simp [StrictAnti, StrictAntiOn]` |
| `Set.strictMonoOn_iff_strictMono` | Set | known | RC4A | `simp [StrictMono, StrictMonoOn]` |
