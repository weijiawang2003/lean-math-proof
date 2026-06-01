# TR7 missing allowlist analysis

- TR6 fresh wins: 18 | recommendations: {'NEED_MORE_EVIDENCE': 3, 'ALREADY_IN_ALLOWLIST': 10, 'KEEP_DYNAMIC_ONLY': 5}
- already in allowlist: 10
- ADD_TO_STATIC_ALLOWLIST: 0 []
- KEEP_DYNAMIC_ONLY: 5
- NEED_MORE_EVIDENCE: 3 ['Finset.biUnion_subset', 'Finset.subset_iff', 'Set.MapsTo']
- missing lemmas by family: {'subset': 2, 'none_tauto': 3, 'add_eq_union': 1, 'other': 1, 'def_unfold': 1}

| theorem | lemma | family | in_wrapper | win_occ | parametric | recommendation |
|---|---|---|---|---|---|---|
| `List.Forall.imp` | `List.forall_iff_forall_mem` | forall_mem | True | 1 | False | ALREADY_IN_ALLOWLIST |
| `Multiset.disjoint_add_left` | `Multiset.disjoint_left` | disjoint_left | True | 4 | True | ALREADY_IN_ALLOWLIST |
| `Multiset.disjoint_add_right` | `Multiset.disjoint_right` | disjoint_right | True | 1 | False | ALREADY_IN_ALLOWLIST |
| `Multiset.disjoint_cons_left` | `Multiset.disjoint_left` | disjoint_left | True | 4 | True | ALREADY_IN_ALLOWLIST |
| `Multiset.singleton_disjoint` | `Multiset.disjoint_left` | disjoint_left | True | 4 | True | ALREADY_IN_ALLOWLIST |
| `Multiset.zero_disjoint` | `Multiset.disjoint_left` | disjoint_left | True | 4 | True | ALREADY_IN_ALLOWLIST |
| `Set.disjoint_iUnion_left` | `Set.disjoint_left` | disjoint_left | True | 4 | True | ALREADY_IN_ALLOWLIST |
| `Set.disjoint_iUnion_right` | `Set.disjoint_left` | disjoint_left | True | 4 | True | ALREADY_IN_ALLOWLIST |
| `Set.disjoint_sUnion_left` | `Set.disjoint_left` | disjoint_left | True | 4 | True | ALREADY_IN_ALLOWLIST |
| `Set.disjoint_sUnion_right` | `Set.disjoint_left` | disjoint_left | True | 4 | True | ALREADY_IN_ALLOWLIST |
| `Multiset.Disjoint.symm` | `None` | none_tauto | False | 3 | False | KEEP_DYNAMIC_ONLY |
| `Multiset.add_eq_union_right_of_le` | `Multiset.add_eq_union_left_of_le` | add_eq_union | False | 1 | False | KEEP_DYNAMIC_ONLY |
| `Multiset.disjoint_comm` | `None` | none_tauto | False | 3 | False | KEEP_DYNAMIC_ONLY |
| `Multiset.disjoint_right` | `None` | none_tauto | False | 3 | False | KEEP_DYNAMIC_ONLY |
| `Nat.sqrt_pos` | `Nat.le_sqrt` | other | False | 1 | False | KEEP_DYNAMIC_ONLY |
| `Finset.biUnion_subset_iff_forall_subset` | `Finset.biUnion_subset` | subset | False | 1 | False | NEED_MORE_EVIDENCE |
| `Finset.image_subset_iff` | `Finset.subset_iff` | subset | False | 1 | False | NEED_MORE_EVIDENCE |
| `Set.mapsTo_singleton` | `Set.MapsTo` | def_unfold | False | 1 | False | NEED_MORE_EVIDENCE |
