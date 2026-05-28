# NS16 — Transfer + wrapper analysis

Per-set, per-model raw eval counts plus the diff of wrapper-only theorems (wrapper proves, raw does not). All NS16 sub-models were trained from gen_v5 base using a 19-row wrapper-only Nat corpus mined from the medium+large+NS16 wrapper traces.

## `nat_defs_medium` (total 38)

| model | proved |
|---|---:|
| `ns13_routed` | 9 |
| `ns15_routed` | 23 |
| `ns15_wrapper` | 37 |
| `ns16_10x` | 23 |
| `ns16_20x` | 22 |
| `ns16_curriculum` | 17 |
| `ns16_routed` | 23 |
| `ns16_wrapper` | 37 |

NS16 wrapper-only on this set (wrapper proves, NS16 routed does not), count 14:
- `Nat.add_mod_eq_add_mod_left`
- `Nat.add_mod_eq_add_mod_right`
- `Nat.add_mod_eq_ite`
- `Nat.div_le_div_right`
- `Nat.div_lt_iff_lt_mul'`
- `Nat.div_lt_one_iff`
- `Nat.div_pos`
- `Nat.div_pos_iff`
- `Nat.dvd_iff_div_mul_eq`
- `Nat.eq_one_of_mul_eq_one_left`
- `Nat.mul_eq_left`
- `Nat.mul_eq_right`
- `Nat.pow_lt_pow_iff_left`
- `Nat.sqrt_lt`

## `nat_defs_large_v5` (total 65)

| model | proved |
|---|---:|
| `ns13_routed` | 13 |
| `ns15_routed` | 35 |
| `ns15_wrapper` | 49 |
| `ns16_10x` | 35 |
| `ns16_20x` | 34 |
| `ns16_curriculum` | 26 |
| `ns16_routed` | 35 |
| `ns16_wrapper` | 49 |

NS16 wrapper-only on this set (wrapper proves, NS16 routed does not), count 14:
- `Nat.add_mod_eq_add_mod_left`
- `Nat.add_mod_eq_add_mod_right`
- `Nat.add_mod_eq_ite`
- `Nat.div_le_div_right`
- `Nat.div_lt_iff_lt_mul'`
- `Nat.div_lt_one_iff`
- `Nat.div_pos`
- `Nat.div_pos_iff`
- `Nat.dvd_iff_div_mul_eq`
- `Nat.eq_one_of_mul_eq_one_left`
- `Nat.mul_eq_left`
- `Nat.mul_eq_right`
- `Nat.pow_lt_pow_iff_left`
- `Nat.sqrt_lt`

## `demo_v1` (total 15)

| model | proved |
|---|---:|
| `ns13_routed` | 10 |
| `ns15_routed` | 10 |
| `ns15_wrapper` | 11 |
| `ns16_10x` | 9 |
| `ns16_20x` | 9 |
| `ns16_curriculum` | 10 |
| `ns16_routed` | 10 |
| `ns16_wrapper` | 11 |

NS16 wrapper-only on this set (wrapper proves, NS16 routed does not), count 1:
- `Nat.mul_add_mod'`

## `ns14_nat_extra` (total 20)

| model | proved |
|---|---:|
| `ns13_routed` | 0 |
| `ns15_routed` | 9 |
| `ns15_wrapper` | 9 |
| `ns16_10x` | 9 |
| `ns16_20x` | 8 |
| `ns16_curriculum` | 3 |
| `ns16_routed` | 9 |
| `ns16_wrapper` | 9 |

NS14 wrapper-only Nat wins retained by NS16 raw router: 8/8
- ✓ `Nat.add_sub_sub_cancel`
- ✓ `Nat.lt_of_lt_pred`
- ✓ `Nat.lt_sub_iff_add_lt'`
- ✓ `Nat.pred_eq_succ_iff`
- ✓ `Nat.pred_sub`
- ✓ `Nat.sub_add_sub_cancel`
- ✓ `Nat.sub_lt_sub_iff_right`
- ✓ `Nat.sub_sub_sub_cancel_right`

## `ns14_set_finset_extra` (total 20)

Missing metrics: `ns16_10x`, `ns16_20x`, `ns16_curriculum`

| model | proved |
|---|---:|
| `ns13_routed` | 13 |
| `ns15_routed` | 13 |
| `ns15_wrapper` | 13 |
| `ns16_routed` | 13 |
| `ns16_wrapper` | 13 |

## `ns16_nat_iff_extra` (total 17)

Missing metrics: `ns13_routed`

| model | proved |
|---|---:|
| `ns15_routed` | 1 |
| `ns15_wrapper` | 2 |
| `ns16_10x` | 1 |
| `ns16_20x` | 1 |
| `ns16_curriculum` | 0 |
| `ns16_routed` | 1 |
| `ns16_wrapper` | 2 |

NS16 wrapper-only on this set (wrapper proves, NS16 routed does not), count 1:
- `Nat.lt_find_iff`

## `ns16_nat_div_mod_extra` (total 25)

Missing metrics: `ns13_routed`

| model | proved |
|---|---:|
| `ns15_routed` | 0 |
| `ns15_wrapper` | 1 |
| `ns16_10x` | 0 |
| `ns16_20x` | 0 |
| `ns16_curriculum` | 0 |
| `ns16_routed` | 0 |
| `ns16_wrapper` | 1 |

NS16 wrapper-only on this set (wrapper proves, NS16 routed does not), count 1:
- `Nat.mul_add_mod_of_lt`

## `ns16_nat_order_extra` (total 16)

Missing metrics: `ns13_routed`

| model | proved |
|---|---:|
| `ns15_routed` | 3 |
| `ns15_wrapper` | 3 |
| `ns16_10x` | 3 |
| `ns16_20x` | 3 |
| `ns16_curriculum` | 3 |
| `ns16_routed` | 3 |
| `ns16_wrapper` | 3 |

## `ns16_nat_mixed_extra` (total 28)

Missing metrics: `ns13_routed`

| model | proved |
|---|---:|
| `ns15_routed` | 0 |
| `ns15_wrapper` | 0 |
| `ns16_10x` | 0 |
| `ns16_20x` | 0 |
| `ns16_curriculum` | 0 |
| `ns16_routed` | 0 |
| `ns16_wrapper` | 0 |

## Headline transfer

- NS14 wrapper-only Nat wins (8): retained 8 by NS16 router (unchanged from NS15 routed).

- **No new raw wins** on any NS16 set vs NS15 routed. The 19-row wrapper-only corpus was too sparse/varied to produce NS14-style transfer.

