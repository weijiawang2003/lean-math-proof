# TR5 attribution

- targets: 92 | classifications: {'NO_WIN_UNDER_BUDGET': 79, 'TRUE_RC4A_REPRODUCTION': 5, 'TRUE_RANKER_DELTA': 7, 'BASELINE_DUPLICATE': 1}
- **TRUE_RANKER_DELTA: 7** | TRUE_RC4A_REPRODUCTION: 5 | credited total: 12
- RC4B evidence (Set.disjoint_left): 3 → ['Set.disjoint_iff_forall_ne', 'Set.disjoint_right', 'Set.disjoint_singleton_left']
- RC4C evidence (d2_simp_aesop): 3 → ['Set.Nonempty.subset_pair_iff_eq', 'Set.disjoint_iff_forall_ne', 'Set.disjoint_right']

## Credited wins

| theorem | class | budget | rank | tags | winning tactic |
|---|---|---|---|---|---|
| `Finset.mem_disjUnion` | TRUE_RC4A_REPRODUCTION | 10 | 8 |  | `simp [Finset.disjUnion]` |
| `List.toFinset.ext_iff` | TRUE_RANKER_DELTA | 5 | 1 |  | `simp [Finset.ext_iff]` |
| `List.toFinset_eq` | TRUE_RANKER_DELTA | 5 | 1 |  | `simp [Multiset.toFinset_eq]` |
| `Set.Nonempty.subset_pair_iff_eq` | TRUE_RANKER_DELTA | 5 | 1 | RC4C | `simp [Set.subset_pair_iff_eq] <;> aesop` |
| `Set.antitoneOn_iff_antitone` | TRUE_RC4A_REPRODUCTION | 5 | 1 |  | `simp [Antitone, AntitoneOn]` |
| `Set.compl_union_self` | TRUE_RANKER_DELTA | 5 | 1 |  | `simp [Set.union_eq_compl_compl_inter_compl]` |
| `Set.disjoint_iff_forall_ne` | TRUE_RANKER_DELTA | 5 | 1 | RC4B,RC4C | `simp [Set.disjoint_left] <;> aesop` |
| `Set.disjoint_right` | TRUE_RANKER_DELTA | 5 | 1 | RC4B,RC4C | `simp [Set.disjoint_left] <;> aesop` |
| `Set.disjoint_singleton_left` | TRUE_RANKER_DELTA | 5 | 1 | RC4B | `simp [Set.disjoint_left]` |
| `Set.monotoneOn_iff_monotone` | TRUE_RC4A_REPRODUCTION | 5 | 1 |  | `simp [Monotone, MonotoneOn]` |
| `Set.strictAntiOn_iff_strictAnti` | TRUE_RC4A_REPRODUCTION | 5 | 1 |  | `simp [StrictAnti, StrictAntiOn]` |
| `Set.strictMonoOn_iff_strictMono` | TRUE_RC4A_REPRODUCTION | 5 | 1 |  | `simp [StrictMono, StrictMonoOn]` |
