# CX3 — Bool / Option catalog audit

Mining a fresh-namespace short-tactic family analogous to NS22's Int/omega, for Bool/Option (`decide` / `simp` / `cases <;> simp`). CX3 is **mining-only** — no training.

## Surface inventory

- Bool candidates: **42**
- Option candidates: **83**
- Total Bool/Option candidates: **125**
- Already used / probed in prior sets: **39** (Bool 35, Option 4)
- **Fresh unused: 86** (verified-available 42, needs-probe 44)
  - fresh Bool: 7, fresh Option: 79

> **Bool exhaustion.** Every verified-available Bool theorem (Bool/Basic.lean) was already consumed by `cx1_bool_option_int`. The fresh opportunity is **Option** (map/bind/pmap/pbind/isSome/isNone/getD/elim/orElse). Fresh Bool only appears via additional-file scans whose LeanDojo availability is unverified until Stage 4.

## Additional files scanned

| file | Bool/Option decls |
|---|---:|
| Mathlib/Data/Bool/Count.lean | 0 |
| Mathlib/Data/Bool/Set.lean | 2 |
| Mathlib/Data/Option/NAry.lean | 31 |
| Mathlib/Logic/Equiv/Option.lean | 0 |
| Mathlib/Data/List/ReduceOption.lean | 0 |

## Likely-family buckets (fresh candidates)

| bucket | count |
|---|---:|
| likely_simp | 70 |
| likely_cases_simp | 9 |
| likely_decide | 7 |

## Tag distribution (all candidates)

| tag | count |
|---|---:|
| option_simp | 52 |
| bool_decide | 42 |
| option_map_bind | 41 |
| bool_logic | 29 |
| option_cases | 26 |
| bool_order | 12 |
| iff | 11 |
| option_mem | 11 |
| bool_nat_cast | 7 |
| other | 2 |

## Fresh verified-available candidates

| theorem | ns | bucket | tags | file |
|---|---|---|---|---|
| `Option.bind_pmap` | Option | likely_cases_simp | option_map_bind,option_cases | Mathlib/Data/Option/Basic.lean |
| `Option.bnot_comp_isNone` | Option | likely_simp | option_simp | Mathlib/Data/Option/Basic.lean |
| `Option.bnot_comp_isSome` | Option | likely_simp | option_simp | Mathlib/Data/Option/Basic.lean |
| `Option.bnot_isNone` | Option | likely_simp | option_simp | Mathlib/Data/Option/Basic.lean |
| `Option.bnot_isSome` | Option | likely_simp | option_simp | Mathlib/Data/Option/Basic.lean |
| `Option.casesOn'_eq_elim` | Option | likely_simp | option_simp,option_cases | Mathlib/Data/Option/Basic.lean |
| `Option.casesOn'_none_coe` | Option | likely_simp | option_simp,option_cases | Mathlib/Data/Option/Basic.lean |
| `Option.elim'_eq_elim` | Option | likely_simp | option_simp,option_cases | Mathlib/Data/Option/Defs.lean |
| `Option.elim_apply` | Option | likely_simp | option_simp,option_cases | Mathlib/Data/Option/Basic.lean |
| `Option.elim_comp` | Option | likely_simp | option_simp,option_cases | Mathlib/Data/Option/Basic.lean |
| `Option.elim_none_some` | Option | likely_simp | option_simp,option_cases | Mathlib/Data/Option/Basic.lean |
| `Option.exists_mem_map` | Option | likely_simp | option_map_bind,option_mem | Mathlib/Data/Option/Basic.lean |
| `Option.forall_mem_map` | Option | likely_simp | option_map_bind,option_mem | Mathlib/Data/Option/Basic.lean |
| `Option.getD_default_eq_iget` | Option | likely_simp | option_simp | Mathlib/Data/Option/Basic.lean |
| `Option.get_map` | Option | likely_simp | option_map_bind | Mathlib/Data/Option/Basic.lean |
| `Option.guard_eq_some'` | Option | likely_simp | option_simp | Mathlib/Data/Option/Basic.lean |
| `Option.isNone_eq_false_iff` | Option | likely_simp | iff,option_simp | Mathlib/Data/Option/Basic.lean |
| `Option.isSome_map` | Option | likely_simp | option_map_bind,option_simp | Mathlib/Data/Option/Basic.lean |
| `Option.join_pmap_eq_pmap_join` | Option | likely_simp | option_map_bind,option_simp | Mathlib/Data/Option/Basic.lean |
| `Option.map_bind` | Option | likely_cases_simp | option_map_bind,option_cases | Mathlib/Data/Option/Basic.lean |
| `Option.map_bind'` | Option | likely_cases_simp | option_map_bind,option_cases | Mathlib/Data/Option/Basic.lean |
| `Option.map_comm` | Option | likely_simp | option_map_bind | Mathlib/Data/Option/Basic.lean |
| `Option.map_injective'` | Option | likely_simp | option_map_bind | Mathlib/Data/Option/Basic.lean |
| `Option.map_pbind` | Option | likely_cases_simp | option_map_bind,option_cases | Mathlib/Data/Option/Basic.lean |
| `Option.map_pmap` | Option | likely_simp | option_map_bind | Mathlib/Data/Option/Basic.lean |
| `Option.mem_map` | Option | likely_simp | option_map_bind,option_mem | Mathlib/Data/Option/Basic.lean |
| `Option.mem_pmem` | Option | likely_simp | option_map_bind,option_mem | Mathlib/Data/Option/Basic.lean |
| `Option.mem_some_iff` | Option | likely_simp | iff,option_simp,option_mem | Mathlib/Data/Option/Defs.lean |
| `Option.mem_toList` | Option | likely_simp | option_simp,option_mem | Mathlib/Data/Option/Defs.lean |
| `Option.none_orElse'` | Option | likely_simp | option_simp | Mathlib/Data/Option/Basic.lean |
| `Option.orElse_eq_none` | Option | likely_simp | option_simp | Mathlib/Data/Option/Basic.lean |
| `Option.orElse_eq_some` | Option | likely_simp | option_simp | Mathlib/Data/Option/Basic.lean |
| `Option.orElse_none'` | Option | likely_simp | option_simp | Mathlib/Data/Option/Basic.lean |
| `Option.pbind_eq_bind` | Option | likely_cases_simp | option_map_bind,option_cases | Mathlib/Data/Option/Basic.lean |
| `Option.pbind_eq_none` | Option | likely_simp | option_map_bind,option_simp,option_cases | Mathlib/Data/Option/Basic.lean |
| `Option.pbind_eq_some` | Option | likely_simp | option_map_bind,option_simp,option_cases | Mathlib/Data/Option/Basic.lean |
| `Option.pbind_map` | Option | likely_cases_simp | option_map_bind,option_cases | Mathlib/Data/Option/Basic.lean |
| `Option.pmap_bind` | Option | likely_cases_simp | option_map_bind,option_cases | Mathlib/Data/Option/Basic.lean |
| `Option.pmap_eq_map` | Option | likely_simp | option_map_bind | Mathlib/Data/Option/Basic.lean |
| `Option.pmap_eq_none_iff` | Option | likely_simp | iff,option_map_bind,option_simp | Mathlib/Data/Option/Basic.lean |
| `Option.pmap_eq_some_iff` | Option | likely_simp | iff,option_map_bind,option_simp | Mathlib/Data/Option/Basic.lean |
| `Option.pmap_map` | Option | likely_simp | option_map_bind | Mathlib/Data/Option/Basic.lean |

## Fresh needs-probe candidates (source-scan / discovered)

| theorem | ns | availability | bucket | file |
|---|---|---|---|---|
| `Bool.exists_bool` | Bool | discovered | likely_decide | Mathlib/Data/Bool/Basic.lean |
| `Bool.false_lt_true` | Bool | discovered | likely_decide | Mathlib/Data/Bool/Basic.lean |
| `Bool.forall_bool` | Bool | discovered | likely_decide | Mathlib/Data/Bool/Basic.lean |
| `Bool.ne_not` | Bool | discovered | likely_decide | Mathlib/Data/Bool/Basic.lean |
| `Bool.not_ne_id` | Bool | discovered | likely_decide | Mathlib/Data/Bool/Basic.lean |
| `Bool.range_eq` | Bool | scan | likely_decide | Mathlib/Data/Bool/Set.lean |
| `Bool.univ_eq` | Bool | scan | likely_decide | Mathlib/Data/Bool/Set.lean |
| `Option.Mem.leftUnique` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.bind_congr'` | Option | discovered | likely_cases_simp | Mathlib/Data/Option/Basic.lean |
| `Option.bind_eq_bind'` | Option | discovered | likely_cases_simp | Mathlib/Data/Option/Basic.lean |
| `Option.casesOn'_coe` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.casesOn'_none` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.casesOn'_some` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.choice_eq_none` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.coe_def` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.coe_get` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.elim'_none` | Option | discovered | likely_simp | Mathlib/Data/Option/Defs.lean |
| `Option.elim'_some` | Option | discovered | likely_simp | Mathlib/Data/Option/Defs.lean |
| `Option.eq_none_or_eq_some` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.eq_of_mem_of_mem` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.get` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.iget_mem` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.iget_of_mem` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.iget_some` | Option | discovered | likely_simp | Mathlib/Data/Option/Defs.lean |
| `Option.joinM_eq_join` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.liftOrGet_choice` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.map` | Option | discovered | likely_simp | Mathlib/Data/Option/NAry.lean |
| `Option.map_coe` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.map_coe'` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.map_comp_some` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.map_eq_id` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.map_inj` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.map_injective` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.map_map` | Option | discovered | likely_simp | Mathlib/Data/Option/NAry.lean |
| `Option.map_uncurry` | Option | discovered | likely_simp | Mathlib/Data/Option/NAry.lean |
| `Option.none_bind'` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.orElse_eq_none'` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.orElse_eq_some'` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.pmap_none` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.pmap_some` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.seq_some` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.some_bind'` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.some_injective` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
| `Option.some_orElse'` | Option | discovered | likely_simp | Mathlib/Data/Option/Basic.lean |
