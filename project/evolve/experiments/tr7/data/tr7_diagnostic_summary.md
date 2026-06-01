# TR7 diagnostic dataset summary

- examples: 18
- static coverage: {'RC4C_RESIDUE_EXCLUDED': 1, 'ALLOWLIST_MISS': 3, 'STATIC_COVERED_AND_SHOULD_SOLVE': 10, 'DYNAMIC_RETRIEVAL_REQUIRED': 3, 'WRAPPER_REPRESENTATION_MISS': 1}
- replay: {'RC4_MISSES_GATE': 4, 'RC4_REPRODUCES_TR6_WIN': 10, 'RC4_ACTION_FAILS_BUT_TR6_PROGRAM_WORKS': 4}
- dynamic/static: {'STATIC_WRAPPER_COMPATIBLE_WITH_ALLOWLIST_EXPANSION': 3, 'STATIC_WRAPPER_COMPATIBLE_NOW': 10, 'DYNAMIC_RETRIEVAL_PREFERRED': 4, 'STATIC_WRAPPER_COMPATIBLE_WITH_SCHEMA_FIX': 1}
- recommended next action: {'TR8: gather recurrence evidence, then add lemma': 3, 'none (already in RC4)': 10, 'RC5H: ranker-guided dynamic retrieval stage': 4, 'RC5H: deploy as bare simp[L] enabling action (RC4B-style)': 1}
- TR6 program reproduces on all wins: True

| theorem | static_coverage | replay | dynamic/static | next action |
|---|---|---|---|---|
| `Multiset.Disjoint.symm` | DYNAMIC_RETRIEVAL_REQUIRED | RC4_ACTION_FAILS_BUT_TR6_PROGRAM_WORKS | DYNAMIC_RETRIEVAL_PREFERRED | RC5H: ranker-guided dynamic retrieval stage |
| `Multiset.add_eq_union_right_of_le` | DYNAMIC_RETRIEVAL_REQUIRED | RC4_ACTION_FAILS_BUT_TR6_PROGRAM_WORKS | DYNAMIC_RETRIEVAL_PREFERRED | RC5H: ranker-guided dynamic retrieval stage |
| `Multiset.disjoint_comm` | DYNAMIC_RETRIEVAL_REQUIRED | RC4_ACTION_FAILS_BUT_TR6_PROGRAM_WORKS | DYNAMIC_RETRIEVAL_PREFERRED | RC5H: ranker-guided dynamic retrieval stage |
| `Nat.sqrt_pos` | ALLOWLIST_MISS | RC4_MISSES_GATE | DYNAMIC_RETRIEVAL_PREFERRED | RC5H: ranker-guided dynamic retrieval stage |
| `List.Forall.imp` | STATIC_COVERED_AND_SHOULD_SOLVE | RC4_REPRODUCES_TR6_WIN | STATIC_WRAPPER_COMPATIBLE_NOW | none (already in RC4) |
| `Multiset.disjoint_add_left` | STATIC_COVERED_AND_SHOULD_SOLVE | RC4_REPRODUCES_TR6_WIN | STATIC_WRAPPER_COMPATIBLE_NOW | none (already in RC4) |
| `Multiset.disjoint_add_right` | STATIC_COVERED_AND_SHOULD_SOLVE | RC4_REPRODUCES_TR6_WIN | STATIC_WRAPPER_COMPATIBLE_NOW | none (already in RC4) |
| `Multiset.disjoint_cons_left` | STATIC_COVERED_AND_SHOULD_SOLVE | RC4_REPRODUCES_TR6_WIN | STATIC_WRAPPER_COMPATIBLE_NOW | none (already in RC4) |
| `Multiset.disjoint_right` | STATIC_COVERED_AND_SHOULD_SOLVE | RC4_REPRODUCES_TR6_WIN | STATIC_WRAPPER_COMPATIBLE_NOW | none (already in RC4) |
| `Multiset.singleton_disjoint` | STATIC_COVERED_AND_SHOULD_SOLVE | RC4_REPRODUCES_TR6_WIN | STATIC_WRAPPER_COMPATIBLE_NOW | none (already in RC4) |
| `Multiset.zero_disjoint` | STATIC_COVERED_AND_SHOULD_SOLVE | RC4_REPRODUCES_TR6_WIN | STATIC_WRAPPER_COMPATIBLE_NOW | none (already in RC4) |
| `Set.disjoint_iUnion_left` | STATIC_COVERED_AND_SHOULD_SOLVE | RC4_REPRODUCES_TR6_WIN | STATIC_WRAPPER_COMPATIBLE_NOW | none (already in RC4) |
| `Set.disjoint_iUnion_right` | STATIC_COVERED_AND_SHOULD_SOLVE | RC4_REPRODUCES_TR6_WIN | STATIC_WRAPPER_COMPATIBLE_NOW | none (already in RC4) |
| `Set.disjoint_sUnion_left` | STATIC_COVERED_AND_SHOULD_SOLVE | RC4_REPRODUCES_TR6_WIN | STATIC_WRAPPER_COMPATIBLE_NOW | none (already in RC4) |
| `Finset.biUnion_subset_iff_forall_subset` | RC4C_RESIDUE_EXCLUDED | RC4_MISSES_GATE | STATIC_WRAPPER_COMPATIBLE_WITH_ALLOWLIST_EXPANSION | TR8: gather recurrence evidence, then add lemma |
| `Finset.image_subset_iff` | ALLOWLIST_MISS | RC4_MISSES_GATE | STATIC_WRAPPER_COMPATIBLE_WITH_ALLOWLIST_EXPANSION | TR8: gather recurrence evidence, then add lemma |
| `Set.mapsTo_singleton` | ALLOWLIST_MISS | RC4_MISSES_GATE | STATIC_WRAPPER_COMPATIBLE_WITH_ALLOWLIST_EXPANSION | TR8: gather recurrence evidence, then add lemma |
| `Set.disjoint_sUnion_right` | WRAPPER_REPRESENTATION_MISS | RC4_ACTION_FAILS_BUT_TR6_PROGRAM_WORKS | STATIC_WRAPPER_COMPATIBLE_WITH_SCHEMA_FIX | RC5H: deploy as bare simp[L] enabling action (RC4B-style) |
