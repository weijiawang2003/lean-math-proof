# SX3 Minimal-Sufficient Attribution

- inputs: `sx3_deferred_results.json`, `sx3_fresh_holdout_results.json`, `sx3_set_cluster_results.json`, `sx3_negative_control_results.json`, `sx3_canonical_smoke_results.json`
- attribution histogram: `{'TRUE_DEPTH2_SEQUENCE_WIN': 5, 'NO_WIN': 13, 'SINGLE_STEP_DUPLICATE': 1, 'BASELINE_DUPLICATE': 10, 'NEEDS_REVIEW': 11}`
- **deferred-known true depth-2 wins (4/4):** `Set.ite_compl`, `Set.ite_inter`, `Set.ite_inter_compl_self`, `Set.ite_inter_self`
- **fresh true depth-2 wins (1):** `Set.ite_inter_inter`
- off-gate emissions: **0**
- SX3_SET_ITE_AESOP verdict: **RC3_CANDIDATE_CONFIRMED** (reproduced_deferred4=True, fresh=1, off_gate=0)

## Per-theorem attribution
| theorem | role | attribution | winning sequence | controls solved |
|---|---|---|---|---|
| `Bool.and_self` | canonical_smoke | **NEEDS_REVIEW** | `` | - |
| `List.append_nil` | canonical_smoke | **NEEDS_REVIEW** | `` | - |
| `Nat.add_zero` | canonical_smoke | **NEEDS_REVIEW** | `` | - |
| `Nat.succ_le_succ` | canonical_smoke | **NEEDS_REVIEW** | `` | - |
| `Nat.zero_add` | canonical_smoke | **NEEDS_REVIEW** | `` | - |
| `Set.ite_compl` | deferred_known | **TRUE_DEPTH2_SEQUENCE_WIN** | `simp [Set.ite] <;> aesop` | - |
| `Set.ite_inter` | deferred_known | **TRUE_DEPTH2_SEQUENCE_WIN** | `simp [Set.ite] <;> aesop` | - |
| `Set.ite_inter_compl_self` | deferred_known | **TRUE_DEPTH2_SEQUENCE_WIN** | `simp [Set.ite] <;> aesop` | - |
| `Set.ite_inter_self` | deferred_known | **TRUE_DEPTH2_SEQUENCE_WIN** | `simp [Set.ite] <;> aesop` | - |
| `Int.add_mul` | negative_control | **NEEDS_REVIEW** | `` | - |
| `List.append_nil` | negative_control | **NEEDS_REVIEW** | `` | - |
| `Multiset.cons_inj_left` | negative_control | **NEEDS_REVIEW** | `` | - |
| `Multiset.toFinset_eq_singleton_iff` | negative_control | **NO_WIN** | `` | - |
| `Nat.add_comm` | negative_control | **NEEDS_REVIEW** | `` | - |
| `Nat.mul_succ` | negative_control | **NEEDS_REVIEW** | `` | - |
| `Set.antitoneOn_iff_antitone` | set_cluster_failure | **NO_WIN** | `` | - |
| `Set.diff_singleton_subset_iff` | None | **NEEDS_REVIEW** | `` | - |
| `Set.diff_union_inter` | set_cluster_failure | **NO_WIN** | `` | - |
| `Set.insert_diff_eq_singleton` | set_cluster_failure | **BASELINE_DUPLICATE** | `ext x <;> aesop` | aesop |
| `Set.insert_diff_of_mem` | set_cluster_failure | **BASELINE_DUPLICATE** | `ext x <;> aesop` | aesop |
| `Set.pair_diff_left` | set_cluster_failure | **BASELINE_DUPLICATE** | `ext x <;> aesop` | simp_all, aesop |
| `Set.pair_eq_pair_iff` | set_cluster_failure | **NO_WIN** | `` | - |
| `Set.powerset_singleton` | set_cluster_failure | **NO_WIN** | `` | - |
| `Set.ssubset_singleton_iff` | set_cluster_failure | **NO_WIN** | `` | - |
| `Set.subset_insert_iff` | set_cluster_failure | **NO_WIN** | `` | - |
| `Set.subset_singleton_iff_eq` | set_cluster_failure | **NO_WIN** | `` | - |
| `Set.union_empty_iff` | set_cluster_failure | **NO_WIN** | `` | - |
| `Set.ite_eq_of_subset_left` | fresh_holdout | **NO_WIN** | `` | - |
| `Set.ite_eq_of_subset_right` | fresh_holdout | **NO_WIN** | `` | - |
| `Set.ite_inter_inter` | fresh_holdout | **TRUE_DEPTH2_SEQUENCE_WIN** | `simp [Set.ite] <;> aesop` | - |
| `Set.ite_inter_of_inter_eq` | fresh_holdout | **NO_WIN** | `` | - |
| `Set.ite_univ` | fresh_holdout | **SINGLE_STEP_DUPLICATE** | `simp [Set.ite] <;> aesop` | simp [Set.ite] |
| `Set.mem_dite` | fresh_holdout | **BASELINE_DUPLICATE** | `` | aesop |
| `Set.mem_dite_empty_left` | fresh_holdout | **BASELINE_DUPLICATE** | `` | aesop |
| `Set.mem_dite_empty_right` | fresh_holdout | **BASELINE_DUPLICATE** | `` | aesop |
| `Set.mem_dite_univ_left` | fresh_holdout | **BASELINE_DUPLICATE** | `` | aesop |
| `Set.mem_dite_univ_right` | fresh_holdout | **BASELINE_DUPLICATE** | `` | aesop |
| `Set.mem_ite_empty_left` | fresh_holdout | **BASELINE_DUPLICATE** | `` | aesop |
| `Set.mem_ite_empty_right` | fresh_holdout | **BASELINE_DUPLICATE** | `` | aesop |
| `Set.subset_ite` | fresh_holdout | **NO_WIN** | `` | - |

## Per-family
| family | true wins | single-step dup | baseline dup | source-spec | off-gate |
|---|---|---|---|---|---|
| SX3_SET_ITE_AESOP | 5 | 1 | 0 | 0 | 0 |
| SX3_SET_ITE_SIMPALL | 0 | 0 | 0 | 0 | 0 |
| SX3_SET_ITE_EXT | 0 | 0 | 0 | 0 | 0 |
| SX3_SET_ITE_EXT_AESOP | 0 | 0 | 0 | 0 | 0 |
| SX3_SET_EXT_AESOP | 0 | 0 | 3 | 0 | 0 |
| SX3_SET_EXT_SIMPALL | 0 | 0 | 0 | 0 | 0 |
| SX3_SET_IFF_CONSTRUCTOR_AESOP | 0 | 0 | 0 | 0 | 0 |
| SX3_SET_IFF_CONSTRUCTOR_SIMPALL | 0 | 0 | 0 | 0 | 0 |
| SX3_SET_SUBSET_ANTISYMM_AESOP | 0 | 0 | 0 | 0 | 0 |
| SX3_SET_SUBSET_ANTISYMM_SIMPALL | 0 | 0 | 0 | 0 | 0 |
| SX3_SET_BYCASES_SIMPALL | 0 | 0 | 0 | 0 | 0 |
| SX3_MULTISET_TOFINSET_AESOP | 0 | 0 | 0 | 0 | 0 |

> TRUE_DEPTH2_SEQUENCE_WIN = depth-2 sequence solved & ALL controls (incl single-shot simp[Set.ite] = RC2 credited mechanism) failed. SX3 families are generic batteries -> SOURCE_SPECIFIC structurally 0.