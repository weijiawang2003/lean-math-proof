# RC4C minimal attribution

- new wins examined: 19
- classifications: {'SIMP_ONLY_DUPLICATE': 4, 'TRUE_D2_SIMP_AESOP_WIN': 7, 'TRUE_D2_SIMP_AESOP_OVERLAP_RC4B': 8}
- **TRUE_D2_SIMP_AESOP_WIN (pure RC4C): 7** ['List.Forall.imp', 'Multiset.disjoint_add_right', 'Set.Nonempty.subset_pair_iff_eq', 'Multiset.disjoint_add_left', 'Multiset.disjoint_iff_ne', 'Multiset.disjoint_union_left', 'Multiset.singleton_disjoint']
- TRUE_D2_SIMP_AESOP_OVERLAP_RC4B: 8 ['Multiset.disjoint_right', 'Multiset.disjoint_singleton', 'Set.disjoint_iUnion_left', 'Set.disjoint_iUnion_right', 'Set.disjoint_iff_forall_ne', 'Set.disjoint_right', 'Set.disjoint_sUnion_left', 'Set.disjoint_sUnion_right']
- SIMP_ONLY_DUPLICATE: 4 ['Finset.biUnion_subset_iff_forall_subset', 'List.forall_map_iff', 'Multiset.disjoint_cons_left', 'Multiset.zero_disjoint']
- pure RC4C by namespace: {'List': 1, 'Multiset': 5, 'Set': 1} | fresh: 3 repro: 4

| theorem | ns | bare | genuine_d2(non) | genuine_d2(ovl) | simp_only | class |
|---|---|---|---|---|---|---|
| `Finset.biUnion_subset_iff_forall_subset` | Finset | [] | [] | [] | ['Finset.biUnion_subset'] | SIMP_ONLY_DUPLICATE |
| `List.Forall.imp` | List | [] | ['List.forall_iff_forall_mem'] | [] | [] | TRUE_D2_SIMP_AESOP_WIN |
| `Multiset.disjoint_add_right` | Multiset | [] | ['Multiset.disjoint_right'] | ['Multiset.disjoint_left'] | [] | TRUE_D2_SIMP_AESOP_WIN |
| `Set.Nonempty.subset_pair_iff_eq` | Set | [] | ['Set.subset_pair_iff_eq'] | [] | [] | TRUE_D2_SIMP_AESOP_WIN |
| `Multiset.disjoint_add_left` | Multiset | [] | ['Multiset.disjoint_right'] | ['Multiset.disjoint_left'] | [] | TRUE_D2_SIMP_AESOP_WIN |
| `Multiset.disjoint_cons_left` | Multiset | [] | [] | [] | ['Multiset.disjoint_left'] | SIMP_ONLY_DUPLICATE |
| `Set.disjoint_iUnion_left` | Set | [] | [] | ['Set.disjoint_left'] | [] | TRUE_D2_SIMP_AESOP_OVERLAP_RC4B |
| `Set.disjoint_iUnion_right` | Set | [] | [] | ['Set.disjoint_left'] | [] | TRUE_D2_SIMP_AESOP_OVERLAP_RC4B |
| `Set.disjoint_iff_forall_ne` | Set | [] | [] | ['Set.disjoint_left'] | [] | TRUE_D2_SIMP_AESOP_OVERLAP_RC4B |
| `Set.disjoint_right` | Set | [] | [] | ['Set.disjoint_left'] | [] | TRUE_D2_SIMP_AESOP_OVERLAP_RC4B |
| `Set.disjoint_sUnion_left` | Set | [] | [] | ['Set.disjoint_left'] | [] | TRUE_D2_SIMP_AESOP_OVERLAP_RC4B |
| `Set.disjoint_sUnion_right` | Set | [] | [] | ['Set.disjoint_left'] | [] | TRUE_D2_SIMP_AESOP_OVERLAP_RC4B |
| `List.forall_map_iff` | List | [] | [] | [] | ['List.forall_iff_forall_mem'] | SIMP_ONLY_DUPLICATE |
| `Multiset.disjoint_iff_ne` | Multiset | [] | ['Multiset.disjoint_right'] | ['Multiset.disjoint_left'] | [] | TRUE_D2_SIMP_AESOP_WIN |
| `Multiset.disjoint_right` | Multiset | [] | [] | ['Multiset.disjoint_left'] | [] | TRUE_D2_SIMP_AESOP_OVERLAP_RC4B |
| `Multiset.disjoint_singleton` | Multiset | [] | [] | ['Multiset.disjoint_left'] | ['Multiset.disjoint_right'] | TRUE_D2_SIMP_AESOP_OVERLAP_RC4B |
| `Multiset.disjoint_union_left` | Multiset | [] | ['Multiset.disjoint_right'] | ['Multiset.disjoint_left'] | [] | TRUE_D2_SIMP_AESOP_WIN |
| `Multiset.singleton_disjoint` | Multiset | [] | ['Multiset.disjoint_right'] | [] | ['Multiset.disjoint_left'] | TRUE_D2_SIMP_AESOP_WIN |
| `Multiset.zero_disjoint` | Multiset | [] | [] | [] | ['Multiset.disjoint_left', 'Multiset.disjoint_right'] | SIMP_ONLY_DUPLICATE |
