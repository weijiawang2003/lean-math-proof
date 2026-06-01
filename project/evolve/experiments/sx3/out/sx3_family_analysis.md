# SX3 Sequence-Family Generalization Analysis

- best family (by heuristic score): **SX3_SET_ITE_AESOP**

| family | seq | fresh | deferred | dup | off-gate | parse-err | multi | score | recommendation |
|---|---|---|---|---|---|---|---|---|---|
| SX3_SET_ITE_AESOP | `simp [Set.ite] <;> aesop` | 1 | 4 | 1 | 0 | 0 | True | 6 | **RC3_CANDIDATE** |
| SX3_SET_ITE_SIMPALL | `simp [Set.ite] <;> simp_` | 0 | 0 | 0 | 0 | 0 | False | 0 | **REJECT_NO_DELTA** |
| SX3_SET_ITE_EXT | `ext x <;> simp [Set.ite]` | 0 | 0 | 0 | 0 | 0 | False | 0 | **REJECT_NO_DELTA** |
| SX3_SET_ITE_EXT_AESOP | `ext x <;> simp [Set.ite]` | 0 | 0 | 0 | 0 | 0 | False | 0 | **REJECT_NO_DELTA** |
| SX3_SET_EXT_SIMPALL | `ext x <;> simp_all` | 0 | 0 | 0 | 0 | 0 | False | 0 | **REJECT_NO_DELTA** |
| SX3_SET_IFF_CONSTRUCTOR_AESOP | `constructor <;> intro h ` | 0 | 0 | 0 | 0 | 0 | False | 0 | **REJECT_NO_DELTA** |
| SX3_SET_IFF_CONSTRUCTOR_SIMPALL | `constructor <;> intro h ` | 0 | 0 | 0 | 0 | 0 | False | 0 | **REJECT_NO_DELTA** |
| SX3_SET_SUBSET_ANTISYMM_AESOP | `apply Set.Subset.antisym` | 0 | 0 | 0 | 0 | 0 | False | 0 | **REJECT_NO_DELTA** |
| SX3_SET_SUBSET_ANTISYMM_SIMPALL | `apply Set.Subset.antisym` | 0 | 0 | 0 | 0 | 0 | False | 0 | **REJECT_NO_DELTA** |
| SX3_SET_BYCASES_SIMPALL | `by_cases h : ?p <;> simp` | 0 | 0 | 0 | 0 | 0 | False | 0 | **REJECT_NO_DELTA** |
| SX3_MULTISET_TOFINSET_AESOP | `simp [Multiset.mem_toFin` | 0 | 0 | 0 | 0 | 0 | False | 0 | **REJECT_NO_DELTA** |
| SX3_SET_EXT_AESOP | `ext x <;> aesop` | 0 | 0 | 3 | 0 | 0 | False | -6 | **REJECT_NO_DELTA** |

- **SX3_SET_ITE_AESOP** true wins: `Set.ite_compl`, `Set.ite_inter`, `Set.ite_inter_compl_self`, `Set.ite_inter_inter`, `Set.ite_inter_self` (fresh=1, deferred=4)

> Generalization score is a heuristic; see raw evidence per family. Only SX3_SET_ITE_AESOP is an RC3 candidate; other families are exploratory.