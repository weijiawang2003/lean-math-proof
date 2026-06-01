# TR7 dynamic vs static classification

- TR6 fresh wins: 18 | classes: {'STATIC_WRAPPER_COMPATIBLE_WITH_ALLOWLIST_EXPANSION': 3, 'STATIC_WRAPPER_COMPATIBLE_NOW': 10, 'DYNAMIC_RETRIEVAL_PREFERRED': 4, 'STATIC_WRAPPER_COMPATIBLE_WITH_SCHEMA_FIX': 1}
- static-compatible now: 10 | with work (allowlist/gate/schema): 4
- dynamic-only: 4
- **% static-compatible: 78% | % dynamic-only: 22%**
- **recommended RC5 direction: hybrid RC5**

| theorem | ns | static_coverage | dynamic_vs_static |
|---|---|---|---|
| `Multiset.Disjoint.symm` | Multiset | DYNAMIC_RETRIEVAL_REQUIRED | DYNAMIC_RETRIEVAL_PREFERRED |
| `Multiset.add_eq_union_right_of_le` | Multiset | DYNAMIC_RETRIEVAL_REQUIRED | DYNAMIC_RETRIEVAL_PREFERRED |
| `Multiset.disjoint_comm` | Multiset | DYNAMIC_RETRIEVAL_REQUIRED | DYNAMIC_RETRIEVAL_PREFERRED |
| `Nat.sqrt_pos` | Nat | ALLOWLIST_MISS | DYNAMIC_RETRIEVAL_PREFERRED |
| `List.Forall.imp` | List | STATIC_COVERED_AND_SHOULD_SOLVE | STATIC_WRAPPER_COMPATIBLE_NOW |
| `Multiset.disjoint_add_left` | Multiset | STATIC_COVERED_AND_SHOULD_SOLVE | STATIC_WRAPPER_COMPATIBLE_NOW |
| `Multiset.disjoint_add_right` | Multiset | STATIC_COVERED_AND_SHOULD_SOLVE | STATIC_WRAPPER_COMPATIBLE_NOW |
| `Multiset.disjoint_cons_left` | Multiset | STATIC_COVERED_AND_SHOULD_SOLVE | STATIC_WRAPPER_COMPATIBLE_NOW |
| `Multiset.disjoint_right` | Multiset | STATIC_COVERED_AND_SHOULD_SOLVE | STATIC_WRAPPER_COMPATIBLE_NOW |
| `Multiset.singleton_disjoint` | Multiset | STATIC_COVERED_AND_SHOULD_SOLVE | STATIC_WRAPPER_COMPATIBLE_NOW |
| `Multiset.zero_disjoint` | Multiset | STATIC_COVERED_AND_SHOULD_SOLVE | STATIC_WRAPPER_COMPATIBLE_NOW |
| `Set.disjoint_iUnion_left` | Set | STATIC_COVERED_AND_SHOULD_SOLVE | STATIC_WRAPPER_COMPATIBLE_NOW |
| `Set.disjoint_iUnion_right` | Set | STATIC_COVERED_AND_SHOULD_SOLVE | STATIC_WRAPPER_COMPATIBLE_NOW |
| `Set.disjoint_sUnion_left` | Set | STATIC_COVERED_AND_SHOULD_SOLVE | STATIC_WRAPPER_COMPATIBLE_NOW |
| `Finset.biUnion_subset_iff_forall_subset` | Finset | RC4C_RESIDUE_EXCLUDED | STATIC_WRAPPER_COMPATIBLE_WITH_ALLOWLIST_EXPANSION |
| `Finset.image_subset_iff` | Finset | ALLOWLIST_MISS | STATIC_WRAPPER_COMPATIBLE_WITH_ALLOWLIST_EXPANSION |
| `Set.mapsTo_singleton` | Set | ALLOWLIST_MISS | STATIC_WRAPPER_COMPATIBLE_WITH_ALLOWLIST_EXPANSION |
| `Set.disjoint_sUnion_right` | Set | WRAPPER_REPRESENTATION_MISS | STATIC_WRAPPER_COMPATIBLE_WITH_SCHEMA_FIX |
