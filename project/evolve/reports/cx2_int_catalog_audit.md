# CX2 — Int catalog audit

## Surface inventory

- CX1 catalog (LeanDojo-verified available): **120** Int theorems
- CX2 additional source-scan candidates: **96** fresh Int theorems
- **Total Int candidates: 216**

Additional files scanned:

| file | theorems extracted |
|---|---:|
| Mathlib/Data/Int/ModEq.lean | 45 |
| Mathlib/Data/Int/Order/Lemmas.lean | 5 |
| Mathlib/Data/Int/Order/Units.lean | 11 |
| Mathlib/Data/Int/Lemmas.lean | 17 |
| Mathlib/Data/Int/SuccPred.lean | 9 |
| Mathlib/Data/Int/Cast/Lemmas.lean | 40 |

## Tag distribution

| tag | count |
|---|---:|
| add_sub_arith | 56 |
| dvd_gcd_lcm | 48 |
| mod_div | 40 |
| le_lt_order | 36 |
| iff_candidate | 33 |
| bitwise | 31 |
| cast_natCast | 24 |
| abs_natAbs_sign | 24 |
| other | 22 |
| succ_pred | 19 |

## Pool-mining buckets

- **iff_omega_pair candidates: 13** (iff + le/lt/add/sub, non-bitwise, non-dvd)
- omega-only candidates: 53 (le/lt/add/sub without iff, non-bitwise, non-dvd)
- cast/natCast candidates: 24 (probed for norm_cast → omega)

Known CX1 wrapper-only-vs-NS9 wins (excluded from mining):

- `Int.emod_two_eq_zero_or_one`
- `Int.le_add_one_iff`
- `Int.le_iff_lt_or_eq`

## iff_omega candidate sample (all listed)

| theorem | file | source |
|---|---|---|
| `Int.le_antisymm_iff` | Mathlib/Data/Int/Defs.lean | cx1_catalog |
| `Int.le_iff_eq_or_lt` | Mathlib/Data/Int/Defs.lean | cx1_catalog |
| `Int.mul_nonneg_iff_of_pos_right` | Mathlib/Data/Int/Defs.lean | cx1_catalog |
| `Int.sub_one_lt_iff` | Mathlib/Data/Int/Defs.lean | cx1_catalog |
| `Int.le_sub_one_iff` | Mathlib/Data/Int/Defs.lean | cx1_catalog |
| `Int.modEq_iff_add_fac` | Mathlib/Data/Int/ModEq.lean | cx2_scan |
| `Int.natAbs_eq_iff_mul_self_eq` | Mathlib/Data/Int/Order/Lemmas.lean | cx2_scan |
| `Int.natAbs_lt_iff_mul_self_lt` | Mathlib/Data/Int/Order/Lemmas.lean | cx2_scan |
| `Int.natAbs_le_iff_mul_self_le` | Mathlib/Data/Int/Order/Lemmas.lean | cx2_scan |
| `Int.natAbs_lt_iff_sq_lt` | Mathlib/Data/Int/Lemmas.lean | cx2_scan |
| `Int.natAbs_le_iff_sq_le` | Mathlib/Data/Int/Lemmas.lean | cx2_scan |
| `Int.pos_iff_one_le` | Mathlib/Data/Int/SuccPred.lean | cx2_scan |
| `Int.covBy_iff_succ_eq` | Mathlib/Data/Int/SuccPred.lean | cx2_scan |
