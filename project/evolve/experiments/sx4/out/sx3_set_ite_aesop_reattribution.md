# SX4 sequence attribution — `SX3_SET_ITE_AESOP`

- sequence: `simp [Set.ite] <;> aesop`
- baseline policy (literal production): **RC2**
- theorems analyzed: **39**
- **credited TRUE_SEQUENCE_DELTA: 0** []
- proxy runner credited: 5 ['Set.ite_compl', 'Set.ite_inter', 'Set.ite_inter_compl_self', 'Set.ite_inter_inter', 'Set.ite_inter_self']
- **over-credit caught: True** (5 theorems proxy-credited but SX4 declines): ['Set.ite_compl', 'Set.ite_inter', 'Set.ite_inter_compl_self', 'Set.ite_inter_inter', 'Set.ite_inter_self']

## Classification histogram

- `FAILED_SEQUENCE`: 22
- `PRODUCTION_SUBSUMED`: 17

## Per-theorem

| theorem | baseline | candidate | proxy credit | SX4 class | credit |
|---|---|---|---|---|---|
| `Bool.and_self` | — | — | — | **FAILED_SEQUENCE** | — |
| `Int.add_mul` | — | — | — | **FAILED_SEQUENCE** | — |
| `List.append_nil` | — | — | — | **FAILED_SEQUENCE** | — |
| `Multiset.cons_inj_left` | — | — | — | **FAILED_SEQUENCE** | — |
| `Multiset.toFinset_eq_singleton_iff` | — | — | — | **FAILED_SEQUENCE** | — |
| `Nat.add_comm` | — | — | — | **FAILED_SEQUENCE** | — |
| `Nat.add_zero` | — | — | — | **FAILED_SEQUENCE** | — |
| `Nat.mul_succ` | — | — | — | **FAILED_SEQUENCE** | — |
| `Nat.succ_le_succ` | — | — | — | **FAILED_SEQUENCE** | — |
| `Nat.zero_add` | — | — | — | **FAILED_SEQUENCE** | — |
| `Set.antitoneOn_iff_antitone` | — | — | — | **FAILED_SEQUENCE** | — |
| `Set.diff_singleton_subset_iff` | — | — | — | **FAILED_SEQUENCE** | — |
| `Set.diff_union_inter` | ✓ | ✓ | — | **PRODUCTION_SUBSUMED** | — |
| `Set.insert_diff_eq_singleton` | ✓ | ✓ | — | **PRODUCTION_SUBSUMED** | — |
| `Set.insert_diff_of_mem` | ✓ | ✓ | — | **PRODUCTION_SUBSUMED** | — |
| `Set.ite_compl` | ✓ | ✓ | ✓ | **PRODUCTION_SUBSUMED** | — |
| `Set.ite_eq_of_subset_left` | — | — | — | **FAILED_SEQUENCE** | — |
| `Set.ite_eq_of_subset_right` | — | — | — | **FAILED_SEQUENCE** | — |
| `Set.ite_inter` | ✓ | ✓ | ✓ | **PRODUCTION_SUBSUMED** | — |
| `Set.ite_inter_compl_self` | ✓ | ✓ | ✓ | **PRODUCTION_SUBSUMED** | — |
| `Set.ite_inter_inter` | ✓ | ✓ | ✓ | **PRODUCTION_SUBSUMED** | — |
| `Set.ite_inter_of_inter_eq` | — | — | — | **FAILED_SEQUENCE** | — |
| `Set.ite_inter_self` | ✓ | ✓ | ✓ | **PRODUCTION_SUBSUMED** | — |
| `Set.ite_univ` | ✓ | ✓ | — | **PRODUCTION_SUBSUMED** | — |
| `Set.mem_dite` | ✓ | ✓ | — | **PRODUCTION_SUBSUMED** | — |
| `Set.mem_dite_empty_left` | ✓ | ✓ | — | **PRODUCTION_SUBSUMED** | — |
| `Set.mem_dite_empty_right` | ✓ | ✓ | — | **PRODUCTION_SUBSUMED** | — |
| `Set.mem_dite_univ_left` | ✓ | ✓ | — | **PRODUCTION_SUBSUMED** | — |
| `Set.mem_dite_univ_right` | ✓ | ✓ | — | **PRODUCTION_SUBSUMED** | — |
| `Set.mem_ite_empty_left` | ✓ | ✓ | — | **PRODUCTION_SUBSUMED** | — |
| `Set.mem_ite_empty_right` | ✓ | ✓ | — | **PRODUCTION_SUBSUMED** | — |
| `Set.pair_diff_left` | ✓ | ✓ | — | **PRODUCTION_SUBSUMED** | — |
| `Set.pair_eq_pair_iff` | — | — | — | **FAILED_SEQUENCE** | — |
| `Set.powerset_singleton` | — | — | — | **FAILED_SEQUENCE** | — |
| `Set.ssubset_singleton_iff` | — | — | — | **FAILED_SEQUENCE** | — |
| `Set.subset_insert_iff` | — | — | — | **FAILED_SEQUENCE** | — |
| `Set.subset_ite` | — | — | — | **FAILED_SEQUENCE** | — |
| `Set.subset_singleton_iff_eq` | — | — | — | **FAILED_SEQUENCE** | — |
| `Set.union_empty_iff` | — | — | — | **FAILED_SEQUENCE** | — |

## Subsumption evidence (PRODUCTION_SUBSUMED)

- `Set.diff_union_inter`: equivalent_sequence_observed=**False** (conf=full, path=['simp [Set.ext_iff]', 'tauto'])
- `Set.insert_diff_eq_singleton`: equivalent_sequence_observed=**False** (conf=full, path=['aesop'])
- `Set.insert_diff_of_mem`: equivalent_sequence_observed=**False** (conf=full, path=['aesop'])
- `Set.ite_compl`: equivalent_sequence_observed=**True** (conf=full, path=['simp [Set.ite]', 'aesop'])
- `Set.ite_inter`: equivalent_sequence_observed=**True** (conf=full, path=['simp [Set.ite]', 'aesop'])
- `Set.ite_inter_compl_self`: equivalent_sequence_observed=**True** (conf=full, path=['simp [Set.ite]', 'aesop'])
- `Set.ite_inter_inter`: equivalent_sequence_observed=**True** (conf=full, path=['simp [Set.ite]', 'aesop'])
- `Set.ite_inter_self`: equivalent_sequence_observed=**True** (conf=full, path=['simp [Set.ite]', 'aesop'])
- `Set.ite_univ`: equivalent_sequence_observed=**False** (conf=full, path=['simp [Set.ite]'])
- `Set.mem_dite`: equivalent_sequence_observed=**False** (conf=full, path=['aesop'])
- `Set.mem_dite_empty_left`: equivalent_sequence_observed=**False** (conf=full, path=['aesop'])
- `Set.mem_dite_empty_right`: equivalent_sequence_observed=**False** (conf=full, path=['aesop'])
- `Set.mem_dite_univ_left`: equivalent_sequence_observed=**False** (conf=full, path=['aesop'])
- `Set.mem_dite_univ_right`: equivalent_sequence_observed=**False** (conf=full, path=['aesop'])
- `Set.mem_ite_empty_left`: equivalent_sequence_observed=**False** (conf=full, path=['aesop'])
- `Set.mem_ite_empty_right`: equivalent_sequence_observed=**False** (conf=full, path=['aesop'])
- `Set.pair_diff_left`: equivalent_sequence_observed=**False** (conf=full, path=['aesop'])