# TR7 static coverage audit (core diagnostic)

- TR6 fresh wins audited: 18
- classification: {'RC4C_RESIDUE_EXCLUDED': 1, 'ALLOWLIST_MISS': 3, 'STATIC_COVERED_AND_SHOULD_SOLVE': 10, 'DYNAMIC_RETRIEVAL_REQUIRED': 3, 'WRAPPER_REPRESENTATION_MISS': 1}

- **RC4 static would cover: 10/18**
- missing due to allowlist (incl. RC4C-excluded): 4
- missing due to gate: 0
- require dynamic retrieval: 3
- wrapper-representation miss: 1

## Per-win classification

| theorem | ns | lemma | family | gate | in_allow | rc4 | class |
|---|---|---|---|---|---|---|---|
| `Finset.image_subset_iff` | Finset | `Finset.subset_iff` | d1_simp_lemma | False | False | not_applicable | ALLOWLIST_MISS |
| `Nat.sqrt_pos` | Nat | `Nat.le_sqrt` | d1_exact | False | False | not_applicable | ALLOWLIST_MISS |
| `Set.mapsTo_singleton` | Set | `Set.MapsTo` | def_unfold_simp | False | False | not_applicable | ALLOWLIST_MISS |
| `Multiset.Disjoint.symm` | Multiset | `None` | d1_tauto | True | False | not_applicable | DYNAMIC_RETRIEVAL_REQUIRED |
| `Multiset.add_eq_union_right_of_le` | Multiset | `Multiset.add_eq_union_left_of_le` | d2_rw_aesop | True | False | not_applicable | DYNAMIC_RETRIEVAL_REQUIRED |
| `Multiset.disjoint_comm` | Multiset | `None` | d1_tauto | True | False | not_applicable | DYNAMIC_RETRIEVAL_REQUIRED |
| `Finset.biUnion_subset_iff_forall_subset` | Finset | `Finset.biUnion_subset` | d2_simp_aesop | False | False | not_applicable | RC4C_RESIDUE_EXCLUDED |
| `List.Forall.imp` | List | `List.forall_iff_forall_mem` | d2_simp_aesop | True | True | solved | STATIC_COVERED_AND_SHOULD_SOLVE |
| `Multiset.disjoint_add_left` | Multiset | `Multiset.disjoint_left` | d2_simp_aesop | True | True | solved | STATIC_COVERED_AND_SHOULD_SOLVE |
| `Multiset.disjoint_add_right` | Multiset | `Multiset.disjoint_right` | d2_simp_aesop | True | True | solved | STATIC_COVERED_AND_SHOULD_SOLVE |
| `Multiset.disjoint_cons_left` | Multiset | `Multiset.disjoint_left` | d2_simp_aesop | True | True | solved | STATIC_COVERED_AND_SHOULD_SOLVE |
| `Multiset.disjoint_right` | Multiset | `None` | d1_tauto | True | False | solved | STATIC_COVERED_AND_SHOULD_SOLVE |
| `Multiset.singleton_disjoint` | Multiset | `Multiset.disjoint_left` | d1_simp_lemma | True | True | solved | STATIC_COVERED_AND_SHOULD_SOLVE |
| `Multiset.zero_disjoint` | Multiset | `Multiset.disjoint_left` | d1_simp_lemma | True | True | solved | STATIC_COVERED_AND_SHOULD_SOLVE |
| `Set.disjoint_iUnion_left` | Set | `Set.disjoint_left` | d2_simp_aesop | True | True | solved | STATIC_COVERED_AND_SHOULD_SOLVE |
| `Set.disjoint_iUnion_right` | Set | `Set.disjoint_left` | d2_simp_aesop | True | True | solved | STATIC_COVERED_AND_SHOULD_SOLVE |
| `Set.disjoint_sUnion_left` | Set | `Set.disjoint_left` | d2_simp_aesop | True | True | solved | STATIC_COVERED_AND_SHOULD_SOLVE |
| `Set.disjoint_sUnion_right` | Set | `Set.disjoint_left` | d2_simp_aesop | True | True | failed | WRAPPER_REPRESENTATION_MISS |
