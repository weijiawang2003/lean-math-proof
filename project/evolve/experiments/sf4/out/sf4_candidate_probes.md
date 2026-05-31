# SF4 candidate probes

- probes generated: **15**
- families: ['generic_aesop_simpall', 'multiset_tofinset_simp_aesop', 'set_ext_aesop', 'set_ext_simp', 'set_iff_constructor_aesop', 'set_ite_ext', 'set_ite_simp_aesop', 'set_subset_antisymm']

| family | tactic/sequence | seq? | risk | gate ns | gate features |
|---|---|---|---|---|---|
| set_ext_aesop | `ext x <;> aesop` | Y | low | Set | inter,union,diff,compl,singleton,pair,powerset |
| set_ext_aesop | `ext x <;> simp_all` | Y | medium | Set | inter,union,diff,compl,singleton,pair,powerset |
| set_ext_simp | `ext x <;> simp` | Y | low | Set | inter,union,diff,compl |
| set_iff_constructor_aesop | `constructor <;> intro h <;> aesop` | Y | medium | Set | _iff,iff_ |
| set_iff_constructor_aesop | `constructor <;> intro h <;> simp_all` | Y | medium | Set | _iff,iff_ |
| set_subset_antisymm | `apply Set.Subset.antisymm <;> intro x <;> aesop` | Y | high | Set | subset,ssubset |
| set_subset_antisymm | `apply Set.Subset.antisymm <;> intro x <;> simp_all` | Y | high | Set | subset,ssubset |
| set_ite_simp_aesop | `simp [Set.ite] <;> aesop` | Y | low | Set | ite,dite,.if |
| set_ite_simp_aesop | `simp [Set.ite, Set.ext_iff] <;> aesop` | Y | medium | Set | ite,dite,.if |
| set_ite_ext | `ext x <;> simp [Set.ite]` | Y | low | Set | ite,dite |
| set_ite_ext | `ext x <;> by_cases h : x ∈ _ <;> simp_all [Set.ite]` | Y | high | Set | ite,dite |
| multiset_tofinset_simp_aesop | `simp [Multiset.mem_toFinset] <;> aesop` | Y | medium | Multiset,Finset | toFinset |
| multiset_tofinset_simp_aesop | `simp [Finset.mem_coe, Multiset.mem_toFinset] <;> aesop` | Y | medium | Multiset,Finset | toFinset |
| generic_aesop_simpall | `aesop` | N | low | * | * |
| generic_aesop_simpall | `simp_all` | N | low | * | * |

> All probes generic/cluster-driven. No SOURCE_SPECIFIC rw bridges. High-risk broad tactics (subset_antisymm, nlinarith, generic) flagged; credit requires SX4 TRUE_DELTA + 0 off-gate.