# SF5 retrieval probe plan

- targets: 20 | total probes: 800
- limits: ≤10 lemmas, ≤40 probes per target

## Probe families
- exact: 165
- simpa_using: 159
- simp_lemma: 149
- rw_lemma: 122
- def_unfold_simp: 67
- aesop_add_simp: 60
- diagnostic_search: 40
- cluster_simp_only: 19
- cluster_simp: 19

### Multiset.toFinset_eq_singleton_iff (40 probes)
- `exact?` [diag]
- `apply?` [diag]
- `simp [Multiset.toFinset, Finset.card, List.toFinset, Option.toFinset]`
- `simp [Multiset.toFinset]`
- `simp [Finset.card]`
- `simp [List.toFinset]`
- `exact Multiset.toFinset_card_eq_card_iff_nodup`
- `simpa using Multiset.toFinset_card_eq_card_iff_nodup`

### Set.Nonempty.subset_pair_iff_eq (40 probes)
- `exact?` [diag]
- `apply?` [diag]
- `simp only [Set.singleton_subset_singleton, Set.subset_singleton_iff, Set.singleton_subset_iff, Set.subset_singleton_iff_eq, Set.mem_sep_iff, Set.mem_inter_iff]`
- `simp [Set.singleton_subset_singleton, Set.subset_singleton_iff, Set.singleton_subset_iff, Set.subset_singleton_iff_eq]`
- `simp [Set.inclusion]`
- `simp [Set.fintypeSubset]`
- `simp [Set.unionEqSigmaOfDisjoint]`
- `exact Set.subset_pair_iff_eq`

### Set.antitoneOn_iff_antitone (40 probes)
- `exact?` [diag]
- `apply?` [diag]
- `simp only [Set.singleton_subset_singleton, Set.subset_singleton_iff, Set.singleton_subset_iff, Set.subset_singleton_iff_eq, Set.mem_sep_iff, Set.mem_inter_iff]`
- `simp [Set.singleton_subset_singleton, Set.subset_singleton_iff, Set.singleton_subset_iff, Set.subset_singleton_iff_eq]`
- `simp [Antitone, AntitoneOn]`
- `simp [Antitone]`
- `simp [AntitoneOn]`
- `simp [Set.inclusion]`

### Set.diff_singleton_subset_iff (40 probes)
- `exact?` [diag]
- `apply?` [diag]
- `simp only [Set.singleton_subset_singleton, Set.subset_singleton_iff, Set.singleton_subset_iff, Set.subset_singleton_iff_eq, Set.mem_sep_iff, Set.mem_inter_iff]`
- `simp [Set.singleton_subset_singleton, Set.subset_singleton_iff, Set.singleton_subset_iff, Set.subset_singleton_iff_eq]`
- `simp [Set.inclusion]`
- `simp [Set.fintypeSubset]`
- `simp [Set.fintypeInsertOfMem]`
- `exact Set.subset_insert_diff_singleton`

### Set.ite_eq_of_subset_left (40 probes)
- `exact?` [diag]
- `apply?` [diag]
- `simp only [Set.inter_subset_ite, Set.ite_inter, Set.ite, Set.ite_univ, Set.ite_mono, Set.ite_right]`
- `simp [Set.inter_subset_ite, Set.ite_inter, Set.ite, Set.ite_univ]`
- `simp [Set.ite]`
- `simp [Set.unionEqSigmaOfDisjoint]`
- `simp [Set.sigmaToiUnion]`
- `exact Set.ite_left`

### Set.ite_eq_of_subset_right (40 probes)
- `exact?` [diag]
- `apply?` [diag]
- `simp only [Set.inter_subset_ite, Set.ite_inter, Set.ite, Set.ite_univ, Set.ite_mono, Set.ite_right]`
- `simp [Set.inter_subset_ite, Set.ite_inter, Set.ite, Set.ite_univ]`
- `simp [Set.ite]`
- `simp [Set.unionEqSigmaOfDisjoint]`
- `simp [Set.sigmaToiUnion]`
- `exact Set.ite_eq_of_subset_left`

