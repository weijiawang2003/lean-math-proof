# RC4C — d2_simp_aesop evidence

- known wins (deduped): **12**
- by namespace: {'Finset': 1, 'List': 1, 'Multiset': 3, 'Set': 7}
- by lemma: {'Finset.biUnion_subset': 1, 'List.forall_iff_forall_mem': 1, 'Multiset.disjoint_right': 1, 'Set.subset_pair_iff_eq': 1, 'Multiset.disjoint_left': 2, 'Set.disjoint_left': 6}
- **pure RC4C (non-overlap): 4** ['Finset.biUnion_subset_iff_forall_subset', 'List.Forall.imp', 'Multiset.disjoint_add_right', 'Set.Nonempty.subset_pair_iff_eq']
- overlap with RC4B: 8 ['Multiset.disjoint_add_left', 'Multiset.disjoint_cons_left', 'Set.disjoint_iUnion_left', 'Set.disjoint_iUnion_right', 'Set.disjoint_iff_forall_ne', 'Set.disjoint_right', 'Set.disjoint_sUnion_left', 'Set.disjoint_sUnion_right']
- overlap with RC4A: 0 []
- fresh: 9 | reproduction: 3
- overlap_dominates: **True**
- needs_review (excluded): 0

| theorem | ns | lemma | tactic | source | fresh | overlap | bucket |
|---|---|---|---|---|---|---|---|
| `Finset.biUnion_subset_iff_forall_subset` | Finset | `Finset.biUnion_subset` | `simp [Finset.biUnion_subset] <;> aesop` | TR6 | True | none | A_pure_rc4c |
| `List.Forall.imp` | List | `List.forall_iff_forall_mem` | `simp [List.forall_iff_forall_mem] <;> aesop` | TR6 | True | none | A_pure_rc4c |
| `Multiset.disjoint_add_right` | Multiset | `Multiset.disjoint_right` | `simp [Multiset.disjoint_right] <;> aesop` | TR6 | True | none | A_pure_rc4c |
| `Set.Nonempty.subset_pair_iff_eq` | Set | `Set.subset_pair_iff_eq` | `simp [Set.subset_pair_iff_eq] <;> aesop` | TR3+TR5 | False | none | A_pure_rc4c |
| `Multiset.disjoint_add_left` | Multiset | `Multiset.disjoint_left` | `simp [Multiset.disjoint_left] <;> aesop` | TR6 | True | RC4B | B_overlap_rc4b |
| `Multiset.disjoint_cons_left` | Multiset | `Multiset.disjoint_left` | `simp [Multiset.disjoint_left] <;> aesop` | TR6 | True | RC4B | B_overlap_rc4b |
| `Set.disjoint_iUnion_left` | Set | `Set.disjoint_left` | `simp [Set.disjoint_left] <;> aesop` | TR6 | True | RC4B | B_overlap_rc4b |
| `Set.disjoint_iUnion_right` | Set | `Set.disjoint_left` | `simp [Set.disjoint_left] <;> aesop` | TR6 | True | RC4B | B_overlap_rc4b |
| `Set.disjoint_iff_forall_ne` | Set | `Set.disjoint_left` | `simp [Set.disjoint_left] <;> aesop` | TR3+TR5 | False | RC4B | B_overlap_rc4b |
| `Set.disjoint_right` | Set | `Set.disjoint_left` | `simp [Set.disjoint_left] <;> aesop` | TR3+TR5 | False | RC4B | B_overlap_rc4b |
| `Set.disjoint_sUnion_left` | Set | `Set.disjoint_left` | `simp [Set.disjoint_left] <;> aesop` | TR6 | True | RC4B | B_overlap_rc4b |
| `Set.disjoint_sUnion_right` | Set | `Set.disjoint_left` | `simp [Set.disjoint_left] <;> aesop` | TR6 | True | RC4B | B_overlap_rc4b |
