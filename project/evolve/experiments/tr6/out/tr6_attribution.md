# TR6 attribution

- searched: 137 | classifications: {'NO_WIN_UNDER_BUDGET': 114, 'FRESH_TRUE_DELTA': 18, 'BASELINE_DUPLICATE': 3, 'NEEDS_REVIEW': 2}
- **FRESH_TRUE_DELTA: 18** | non-Set positives: 13 {'Finset': 2, 'List': 1, 'Multiset': 9, 'Nat': 1}
- RC4A evidence: 1 | RC4B: 8 | RC4C: 9

## Credited fresh wins

| theorem | ns | budget | rank | tags | winning tactic |
|---|---|---|---|---|---|
| `Finset.biUnion_subset_iff_forall_subset` | Finset | 5 | 1 | RC4C | `simp [Finset.biUnion_subset] <;> aesop` |
| `Finset.image_subset_iff` | Finset | 10 | 9 |  | `simp [Finset.subset_iff]` |
| `List.Forall.imp` | List | 5 | 2 | RC4C | `simp [List.forall_iff_forall_mem] <;> aeso` |
| `Multiset.Disjoint.symm` | Multiset | 20 | 20 |  | `tauto` |
| `Multiset.add_eq_union_right_of_le` | Multiset | 20 | 16 |  | `rw [Multiset.add_eq_union_left_of_le] <;> ` |
| `Multiset.disjoint_add_left` | Multiset | 5 | 3 | RC4B,RC4C | `simp [Multiset.disjoint_left] <;> aesop` |
| `Multiset.disjoint_add_right` | Multiset | 20 | 12 | RC4C | `simp [Multiset.disjoint_right] <;> aesop` |
| `Multiset.disjoint_comm` | Multiset | 20 | 16 |  | `tauto` |
| `Multiset.disjoint_cons_left` | Multiset | 5 | 2 | RC4B,RC4C | `simp [Multiset.disjoint_left] <;> aesop` |
| `Multiset.disjoint_right` | Multiset | 20 | 17 |  | `tauto` |
| `Multiset.singleton_disjoint` | Multiset | 5 | 3 | RC4B | `simp [Multiset.disjoint_left]` |
| `Multiset.zero_disjoint` | Multiset | 10 | 7 | RC4B | `simp [Multiset.disjoint_left]` |
| `Nat.sqrt_pos` | Nat | 20 | 16 |  | `exact Nat.le_sqrt` |
| `Set.disjoint_iUnion_left` | Set | 5 | 4 | RC4B,RC4C | `simp [Set.disjoint_left] <;> aesop` |
| `Set.disjoint_iUnion_right` | Set | 5 | 2 | RC4B,RC4C | `simp [Set.disjoint_left] <;> aesop` |
| `Set.disjoint_sUnion_left` | Set | 5 | 5 | RC4B,RC4C | `simp [Set.disjoint_left] <;> aesop` |
| `Set.disjoint_sUnion_right` | Set | 5 | 2 | RC4B,RC4C | `simp [Set.disjoint_left] <;> aesop` |
| `Set.mapsTo_singleton` | Set | 20 | 12 | RC4A | `simp [Set.MapsTo]` |
