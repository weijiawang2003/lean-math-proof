# RC4D minimal attribution

- new wins examined: 24
- classifications: {'TRUE_RC4A_WIN': 5, 'TRUE_RC4B_WIN': 16, 'TRUE_RC4C_RESIDUE_WIN': 2, 'SIMP_ONLY_DUPLICATE': 1}
- **credited delta total: 23**
- credited by component: {'RC4A': 5, 'RC4B': 16, 'RC4C_residue': 2}
- credited fresh: 0 []
- credited reproductions: 23
- overlap removed (RC4C_residue→RC4B): 9 ['Multiset.disjoint_add_left', 'Multiset.disjoint_add_right', 'Multiset.disjoint_cons_left', 'Multiset.disjoint_iff_ne', 'Multiset.disjoint_right', 'Multiset.disjoint_singleton', 'Multiset.disjoint_union_left', 'Multiset.singleton_disjoint', 'Multiset.zero_disjoint']

| theorem | ns | win_comp | win_tac | bare | simp[L] | genuine_d2 | class |
|---|---|---|---|---|---|---|---|
| `Finset.mem_disjUnion` | Finset | RC4A | `simp [Finset.disjUnion]` | [] | True | False | TRUE_RC4A_WIN |
| `Set.antitoneOn_iff_antitone` | Set | RC4A | `simp [Antitone, AntitoneOn]` | [] | True | False | TRUE_RC4A_WIN |
| `Set.monotoneOn_iff_monotone` | Set | RC4A | `simp [Monotone, MonotoneOn]` | [] | True | False | TRUE_RC4A_WIN |
| `Set.strictAntiOn_iff_strictAnti` | Set | RC4A | `simp [StrictAnti, StrictAntiOn]` | [] | True | False | TRUE_RC4A_WIN |
| `Set.strictMonoOn_iff_strictMono` | Set | RC4A | `simp [StrictMono, StrictMonoOn]` | [] | True | False | TRUE_RC4A_WIN |
| `Multiset.disjoint_add_left` | Multiset | RC4B | `simp [Multiset.disjoint_left] <;> aesop` | [] | False | True | TRUE_RC4B_WIN |
| `Multiset.disjoint_add_right` | Multiset | RC4B | `simp [Multiset.disjoint_left] <;> aesop` | [] | False | True | TRUE_RC4B_WIN |
| `Multiset.disjoint_cons_left` | Multiset | RC4B | `simp [Multiset.disjoint_left]` | [] | True | False | TRUE_RC4B_WIN |
| `Multiset.disjoint_iff_ne` | Multiset | RC4B | `simp [Multiset.disjoint_left] <;> aesop` | [] | False | True | TRUE_RC4B_WIN |
| `Multiset.disjoint_right` | Multiset | RC4B | `simp [Multiset.disjoint_left] <;> aesop` | [] | False | True | TRUE_RC4B_WIN |
| `Multiset.disjoint_singleton` | Multiset | RC4B | `simp [Multiset.disjoint_left] <;> aesop` | [] | False | True | TRUE_RC4B_WIN |
| `Multiset.disjoint_union_left` | Multiset | RC4B | `simp [Multiset.disjoint_left] <;> aesop` | [] | False | True | TRUE_RC4B_WIN |
| `Multiset.singleton_disjoint` | Multiset | RC4B | `simp [Multiset.disjoint_left]` | [] | True | False | TRUE_RC4B_WIN |
| `Multiset.zero_disjoint` | Multiset | RC4B | `simp [Multiset.disjoint_left]` | [] | True | False | TRUE_RC4B_WIN |
| `Set.disjoint_iUnion_left` | Set | RC4B | `simp [Set.disjoint_left] <;> aesop` | [] | False | True | TRUE_RC4B_WIN |
| `Set.disjoint_iUnion_right` | Set | RC4B | `simp [Set.disjoint_left] <;> aesop` | [] | False | True | TRUE_RC4B_WIN |
| `Set.disjoint_iff_forall_ne` | Set | RC4B | `simp [Set.disjoint_left] <;> aesop` | [] | False | True | TRUE_RC4B_WIN |
| `Set.disjoint_right` | Set | RC4B | `simp [Set.disjoint_left] <;> aesop` | [] | False | True | TRUE_RC4B_WIN |
| `Set.disjoint_sUnion_left` | Set | RC4B | `simp [Set.disjoint_left] <;> aesop` | [] | False | True | TRUE_RC4B_WIN |
| `Set.disjoint_sUnion_right` | Set | RC4B | `simp [Set.disjoint_left] <;> aesop` | [] | False | True | TRUE_RC4B_WIN |
| `Set.disjoint_singleton_left` | Set | RC4B | `simp [Set.disjoint_left]` | [] | True | False | TRUE_RC4B_WIN |
| `List.Forall.imp` | List | RC4C_residue | `simp [List.forall_iff_forall_mem] <;> aesop` | [] | False | True | TRUE_RC4C_RESIDUE_WIN |
| `Set.Nonempty.subset_pair_iff_eq` | Set | RC4C_residue | `simp [Set.subset_pair_iff_eq] <;> aesop` | [] | False | True | TRUE_RC4C_RESIDUE_WIN |
| `List.forall_map_iff` | List | RC4C_residue | `simp [List.forall_iff_forall_mem]` | [] | True | False | SIMP_ONLY_DUPLICATE |
