# RC4D component manifest

- ordering: ['RC4A', 'RC4B', 'RC4C_residue']
- total distinct actions: 8

## RC4A — def_unfold_simp (CONFIRMED)
- allowlist: ['Monotone', 'MonotoneOn', 'Antitone', 'AntitoneOn', 'StrictMono', 'StrictMonoOn', 'StrictAnti', 'StrictAntiOn', 'Finset.disjUnion']
- known wins (5): ['Finset.mem_disjUnion', 'Set.antitoneOn_iff_antitone', 'Set.monotoneOn_iff_monotone', 'Set.strictAntiOn_iff_strictAnti', 'Set.strictMonoOn_iff_strictMono']

## RC4B — disjoint_left bridge (CONFIRMED)
- actions: ['SET_DISJOINT_LEFT_SIMP', 'SET_DISJOINT_LEFT_SIMP_AESOP', 'MULTISET_DISJOINT_LEFT_SIMP', 'MULTISET_DISJOINT_LEFT_SIMP_AESOP']
- known wins (16): ['Multiset.disjoint_add_left', 'Multiset.disjoint_add_right', 'Multiset.disjoint_cons_left', 'Multiset.disjoint_iff_ne', 'Multiset.disjoint_right', 'Multiset.disjoint_singleton', 'Multiset.disjoint_union_left', 'Multiset.singleton_disjoint', 'Multiset.zero_disjoint', 'Set.disjoint_iUnion_left', 'Set.disjoint_iUnion_right', 'Set.disjoint_iff_forall_ne', 'Set.disjoint_right', 'Set.disjoint_sUnion_left', 'Set.disjoint_sUnion_right', 'Set.disjoint_singleton_left']
- fresh-holdout wins: ['Multiset.disjoint_add_right', 'Multiset.disjoint_iff_ne', 'Multiset.disjoint_right', 'Multiset.disjoint_singleton', 'Multiset.disjoint_union_left']

## RC4C_residue (de-duplicated)
### Residue lemma decisions

| lemma | action | decision | wins | fresh |
|---|---|---|---|---|
| `Multiset.disjoint_right` | MULTISET_DISJOINT_RIGHT_D2 | **INCLUDE_AS_DEPTH2_SIMP_AESOP** | 5 | 3 |
| `Set.subset_pair_iff_eq` | SET_SUBSET_PAIR_D2 | **INCLUDE_AS_DEPTH2_SIMP_AESOP** | 1 | 0 |
| `List.forall_iff_forall_mem` | LIST_FORALL_D2 | **INCLUDE_AS_DEPTH2_SIMP_AESOP** | 1 | 0 |

### Excluded RC4C actions

| action | lemma | reason |
|---|---|---|
| SET_DISJOINT_LEFT_D2 | `Set.disjoint_left` | EXCLUDE_OVERLAP |
| MULTISET_DISJOINT_LEFT_D2 | `Multiset.disjoint_left` | EXCLUDE_OVERLAP |
| FINSET_BIUNION_SUBSET_D2 | `Finset.biUnion_subset` | EXCLUDE_DUPLICATE |

### Theorem-level overlap (RC4B ∩ RC4C_residue)
- residue theorem wins: ['List.Forall.imp', 'Multiset.disjoint_add_left', 'Multiset.disjoint_add_right', 'Multiset.disjoint_iff_ne', 'Multiset.disjoint_union_left', 'Multiset.singleton_disjoint', 'Set.Nonempty.subset_pair_iff_eq']
- overlap with RC4B (credited to RC4B): ['Multiset.disjoint_add_left', 'Multiset.disjoint_add_right', 'Multiset.disjoint_iff_ne', 'Multiset.disjoint_union_left', 'Multiset.singleton_disjoint']
- **additive over RC4B (RC4C_residue credit): ['List.Forall.imp', 'Set.Nonempty.subset_pair_iff_eq']**

## Expected credited components
- RC4A: 5
- RC4B: 16
- RC4C_residue additive: 2
