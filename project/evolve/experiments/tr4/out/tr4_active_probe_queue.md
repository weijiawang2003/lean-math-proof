# TR4 active probe queue

- theorems: 92 | categories: {'candidate_family_validation': 70, 'high_uncertainty': 5, 'underrepresented_namespace': 4, 'already_has_win': 13}
- (no live probes run; priorities from leakage-free OOF HGB scores)

## Top 15 (open theorems first, by expected value)

| theorem | ns | EV | reason | top program |
|---|---|---|---|---|
| `Set.pair_eq_pair_iff` | Set | 0.994 | candidate_family_validation | `simp [Set.subset_pair_iff_eq] <;> aesop` |
| `List.toFinset_filter` | List | 0.6882 | candidate_family_validation | `simp [Multiset.toFinset_filter]` |
| `Set.Nonempty.eq_univ` | Set | 0.2632 | candidate_family_validation | `simp [Set.empty_ne_univ] <;> aesop` |
| `Finset.sizeOf_lt_sizeOf_of_mem` | Finset | 0.0266 | candidate_family_validation | `simp [Multiset.sizeOf]` |
| `List.toFinset_eq_iff_perm_dedup` | List | 0.0112 | high_uncertainty | `simp [Multiset.toFinset, Multiset.mem_toFinset]` |
| `Set.insert_subset_insert_iff` | Set | 0.0103 | candidate_family_validation | `simp [AList.insert]` |
| `Multiset.Nodup.toFinset_inj` | Multiset | 0.0095 | candidate_family_validation | `simp [Multiset.toFinset_eq] <;> simp_all` |
| `Set.ssubset_iff_sdiff_singleton` | Set | 0.0046 | candidate_family_validation | `simp [Set.ssubset_univ_iff]` |
| `Set.eq_of_inclusion_surjective` | Set | 0.0042 | candidate_family_validation | `ext x <;> simp [Set.inclusion_right]` |
| `Finset.induction_on_union` | Finset | 0.004 | high_uncertainty | `simp [Finset.induction]` |
| `Function.Injective.nonempty_apply_iff` | Function | 0.0036 | underrepresented_namespace | `aesop` |
| `Finset.Nontrivial.sdiff_singleton_nonempty` | Finset | 0.0036 | candidate_family_validation | `aesop` |
| `Finset.eq_singleton_iff_nonempty_unique_mem` | Finset | 0.0028 | candidate_family_validation | `aesop` |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` | Set | 0.0016 | high_uncertainty | `simp [AntitoneOn, MonotoneOn]` |
| `List.toFinset_eq_empty_iff` | List | 0.0015 | high_uncertainty | `simp [Multiset.toFinset, Multiset.mem_toFinset]` |
