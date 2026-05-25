# NS18 — wrapper-variant signal comparison

Per-variant, per-set summary of:
- `proved`: this variant + NS15 routed
- `vs raw`: variant proved − raw NS15 routed proved
- `vs wrap`: variant proved − NS9 wrapper baseline proved
- wrapper-only-new = theorems variant proves that raw NS15 routed does not

## `constructor_omega`

| set | proved | raw | wrap | Δraw | Δwrap | new wrapper-only |
|---|---:|---:|---:|---:|---:|---|
| `nat_defs_medium` | 37 | 23 | 37 | +14 | +0 | `Nat.add_mod_eq_add_mod_left`, `Nat.add_mod_eq_add_mod_right`, `Nat.add_mod_eq_ite`, `Nat.div_le_div_right`, `Nat.div_lt_iff_lt_mul'`, `Nat.div_lt_one_iff`, `Nat.div_pos`, `Nat.div_pos_iff`, `Nat.dvd_iff_div_mul_eq`, `Nat.eq_one_of_mul_eq_one_left`, `Nat.mul_eq_left`, `Nat.mul_eq_right`, `Nat.pow_lt_pow_iff_left`, `Nat.sqrt_lt` |
| `demo_v1` | 11 | 10 | 11 | +1 | +0 | `Nat.mul_add_mod'` |
| `ns14_nat_extra` | 9 | 9 | 9 | +0 | +0 |  |
| `ns16_nat_iff_extra` | 2 | 1 | 2 | +1 | +0 | `Nat.lt_find_iff` |

**Total wrapper-only-new (vs raw) across sets: 16**

## `split_ifs_omega`

| set | proved | raw | wrap | Δraw | Δwrap | new wrapper-only |
|---|---:|---:|---:|---:|---:|---|
| `demo_v1` | 11 | 10 | 11 | +1 | +0 | `Nat.mul_add_mod'` |
| `ns16_nat_div_mod_extra` | 1 | 0 | 1 | +1 | +0 | `Nat.mul_add_mod_of_lt` |
| `ns16_nat_mixed_extra` | 0 | 0 | 0 | +0 | +0 |  |
| `ns17_nat_remaining` | 1 | 1 | 1 | +0 | +0 |  |

**Total wrapper-only-new (vs raw) across sets: 2**

## `nat_simp_arith`

| set | proved | raw | wrap | Δraw | Δwrap | new wrapper-only |
|---|---:|---:|---:|---:|---:|---|
| `demo_v1` | 11 | 10 | 11 | +1 | +0 | `Nat.mul_add_mod'` |
| `ns16_nat_div_mod_extra` | 2 | 0 | 1 | +2 | +1 | `Nat.mul_add_mod_of_lt`, `Nat.mul_mod_mod` |
| `ns16_nat_order_extra` | 3 | 3 | 3 | +0 | +0 |  |
| `ns16_nat_mixed_extra` | 0 | 0 | 0 | +0 | +0 |  |

**Total wrapper-only-new (vs raw) across sets: 3**

## `aesop_wrapper`

| set | proved | raw | wrap | Δraw | Δwrap | new wrapper-only |
|---|---:|---:|---:|---:|---:|---|
| `demo_v1` | 11 | 10 | 11 | +1 | +0 | `Nat.mul_add_mod'` |
| `ns16_nat_mixed_extra` | 0 | 0 | 0 | +0 | +0 |  |
| `ns17_set_extra` | 17 | 18 | 18 | -1 | -1 |  |
| `ns17_finset_extra` | 15 | 12 | 12 | +3 | +3 | `Finset.coe_insert`, `Finset.cons_eq_insert`, `Finset.disjUnion_singleton` |
| `ns17_list_multiset` | 11 | 11 | 11 | +0 | +0 |  |
| `ns17_nat_remaining` | 1 | 1 | 1 | +0 | +0 |  |

**Total wrapper-only-new (vs raw) across sets: 4**

## `bool_option_cases`

| set | proved | raw | wrap | Δraw | Δwrap | new wrapper-only |
|---|---:|---:|---:|---:|---:|---|
| `demo_v1` | 11 | 10 | 11 | +1 | +0 | `Nat.mul_add_mod'` |
| `ns17_set_extra` | 18 | 18 | 18 | +0 | +0 |  |
| `ns17_finset_extra` | 12 | 12 | 12 | +0 | +0 |  |
| `ns17_list_multiset` | 11 | 11 | 11 | +0 | +0 |  |

**Total wrapper-only-new (vs raw) across sets: 1**

## `combined_safe`

| set | proved | raw | wrap | Δraw | Δwrap | new wrapper-only |
|---|---:|---:|---:|---:|---:|---|
| `nat_defs_medium` | 37 | 23 | 37 | +14 | +0 | `Nat.add_mod_eq_add_mod_left`, `Nat.add_mod_eq_add_mod_right`, `Nat.add_mod_eq_ite`, `Nat.div_le_div_right`, `Nat.div_lt_iff_lt_mul'`, `Nat.div_lt_one_iff`, `Nat.div_pos`, `Nat.div_pos_iff`, `Nat.dvd_iff_div_mul_eq`, `Nat.eq_one_of_mul_eq_one_left`, `Nat.mul_eq_left`, `Nat.mul_eq_right`, `Nat.pow_lt_pow_iff_left`, `Nat.sqrt_lt` |
| `nat_defs_large_v5` | 50 | 35 | 49 | +15 | +1 | `Nat.add_mod_eq_add_mod_left`, `Nat.add_mod_eq_add_mod_right`, `Nat.add_mod_eq_ite`, `Nat.div_le_div_right`, `Nat.div_lt_iff_lt_mul'`, `Nat.div_lt_one_iff`, `Nat.div_pos`, `Nat.div_pos_iff`, `Nat.dvd_iff_div_mul_eq`, `Nat.eq_one_of_mul_eq_one_left`, `Nat.mod_mul_mod`, `Nat.mul_eq_left`, `Nat.mul_eq_right`, `Nat.pow_lt_pow_iff_left`, `Nat.sqrt_lt` |
| `demo_v1` | 11 | 10 | 11 | +1 | +0 | `Nat.mul_add_mod'` |
| `ns14_nat_extra` | 9 | 9 | 9 | +0 | +0 |  |
| `ns14_set_finset_extra` | 12 | 13 | 13 | -1 | -1 | `Set.inter_nonempty_iff_exists_left` |
| `ns16_nat_iff_extra` | 2 | 1 | 2 | +1 | +0 | `Nat.lt_find_iff` |
| `ns16_nat_div_mod_extra` | 2 | 0 | 1 | +2 | +1 | `Nat.mul_add_mod_of_lt`, `Nat.mul_mod_mod` |
| `ns16_nat_order_extra` | 3 | 3 | 3 | +0 | +0 |  |
| `ns16_nat_mixed_extra` | 0 | 0 | 0 | +0 | +0 |  |
| `ns17_set_extra` | 18 | 18 | 18 | +0 | +0 |  |
| `ns17_finset_extra` | 12 | 12 | 12 | +0 | +0 |  |
| `ns17_list_multiset` | 11 | 11 | 11 | +0 | +0 |  |
| `ns17_nat_remaining` | 1 | 1 | 1 | +0 | +0 |  |

**Total wrapper-only-new (vs raw) across sets: 34**

