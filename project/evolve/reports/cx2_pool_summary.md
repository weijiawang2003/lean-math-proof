# CX2 — Int wrapper-only pool

## Per-set mining summary

| arc | set | raw | wrap | wrap-only | Int wrap-only |
|---|---|---:|---:|---:|---:|
| CX1 | cx1_bool_option_int | 26 | 29 | 3 | **3** |
| CX2 | cx2_int_iff_omega_easy | 1 | 5 | 4 | **4** |
| CX2 | cx2_int_iff_omega_medium | 0 | 2 | 2 | **2** |
| CX2 | cx2_int_order_arith | 4 | 16 | 12 | **12** |
| CX2 | cx2_int_mixed | 4 | 6 | 2 | **2** |

## Combined pool (Int namespace)

| family | unique wins | gate met? | recommended oversample |
|---|---:|:---:|---:|
| `fallback_omega` | **13** | ✓ | 2× |
| `iff_omega_pair` | **10** | ✓ | 5× |

## Theorem detail

### `fallback_omega` (13 unique)

| theorem | winning tactic | first seen |
|---|---|---|
| `Int.emod_two_eq_zero_or_one` | `omega` | CX1:cx1_bool_option_int |
| `Int.le_of_eq` | `omega` | CX2:cx2_int_order_arith |
| `Int.natAbs_coe_sub_coe_lt_of_lt` | `omega` | CX2:cx2_int_order_arith |
| `Int.le_or_lt` | `omega` | CX2:cx2_int_order_arith |
| `Int.natAbs_coe_sub_coe_le_of_le` | `omega` | CX2:cx2_int_order_arith |
| `Int.zero_le_ofNat` | `omega` | CX2:cx2_int_order_arith |
| `Int.lt_or_lt_of_ne` | `omega` | CX2:cx2_int_order_arith |
| `Int.natAbs_add_of_nonpos` | `omega` | CX2:cx2_int_order_arith |
| `Int.lt_asymm` | `omega` | CX2:cx2_int_order_arith |
| `Int.le_natCast_sub` | `omega` | CX2:cx2_int_order_arith |
| `Int.neg_emod_two` | `omega` | CX2:cx2_int_order_arith |
| `Int.lt_or_le` | `omega` | CX2:cx2_int_order_arith |
| `Int.natCast_pred_of_pos` | `omega` | CX2:cx2_int_mixed |

### `iff_omega_pair` (10 unique)

| theorem | winning tactic | first seen |
|---|---|---|
| `Int.le_add_one_iff` | `exact ⟨fun h => by omega, fun h => by omega⟩` | CX1:cx1_bool_option_int |
| `Int.le_iff_lt_or_eq` | `exact ⟨fun h => by omega, fun h => by omega⟩` | CX1:cx1_bool_option_int |
| `Int.le_sub_one_iff` | `exact ⟨fun h => by omega, fun h => by omega⟩` | CX2:cx2_int_iff_omega_easy |
| `Int.sub_one_lt_iff` | `exact ⟨fun h => by omega, fun h => by omega⟩` | CX2:cx2_int_iff_omega_easy |
| `Int.le_antisymm_iff` | `exact ⟨fun h => by omega, fun h => by omega⟩` | CX2:cx2_int_iff_omega_easy |
| `Int.le_iff_eq_or_lt` | `exact ⟨fun h => by omega, fun h => by omega⟩` | CX2:cx2_int_iff_omega_easy |
| `Int.natCast_nonpos_iff` | `exact ⟨fun h => by omega, fun h => by omega⟩` | CX2:cx2_int_iff_omega_medium |
| `Int.natCast_ne_zero_iff_pos` | `exact ⟨fun h => by omega, fun h => by omega⟩` | CX2:cx2_int_iff_omega_medium |
| `Int.lt_toNat` | `exact ⟨fun h => by omega, fun h => by omega⟩` | CX2:cx2_int_order_arith |
| `Int.natCast_eq_zero` | `exact ⟨fun h => by omega, fun h => by omega⟩` | CX2:cx2_int_mixed |

