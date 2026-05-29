# MX1 symbolic-action frontier audit

Fresh, availability-screened candidates with likely symbolic-action potential, excluding every theorem consumed by prior CX/WX/AX/SX arcs (and demo_v1 / training sets, which are in the `tasks.THEOREM_SETS` registry). No Lean is run here — availability is confirmed at mine time.

- pool: cx1 availability probe (1817); tags: discovered_cx1 (3989)
- already-used excluded: **1799**
- **total fresh candidates: 1667**
- depth-2 sequence candidates flagged: 300

## By namespace

| namespace | fresh candidates |
|---|---|
| Multiset | 68 |
| Finset | 606 |
| List | 237 |
| Option | 0 |
| Set | 756 |

## By likely action family

| family | count |
|---|---|
| `set_ext_simp` | 756 |
| `finset_cases_simp` | 417 |
| `finset_ext_simp` | 189 |
| `list_cases_simp` | 125 |
| `list_induction_simp` | 112 |
| `multiset_induction_simp` | 44 |
| `multiset_ext_simp` | 24 |

## By namespace × family

- **Multiset**: multiset_induction_simp=44, multiset_ext_simp=24
- **Finset**: finset_cases_simp=417, finset_ext_simp=189
- **List**: list_cases_simp=125, list_induction_simp=112
- **Set**: set_ext_simp=756
