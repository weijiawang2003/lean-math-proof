# NS22 — Int transfer vs memorization analysis

## Summary

| ckpt | own family | own pool | via trained tactic | other pool | held-out gains | neg losses | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `gen_v5_ns22_int_iff_omega_5x` | iff_omega_pair | 2/10 | 0 | 2/13 | 0 | 3 | **weak_or_no_signal** |
| `gen_v5_ns22_int_iff_omega_10x` | iff_omega_pair | 0/10 | 0 | 1/13 | 0 | 0 | **weak_or_no_signal** |
| `gen_v5_ns22_int_fallback_omega_5x` | fallback_omega | 13/13 | 13 | 9/10 | 0 | 2 | **cross_family_transfer** |

## Own-pool detail

### `gen_v5_ns22_int_iff_omega_5x` (own pool = iff_omega_pair)

| theorem | solved | tactic emitted |
|---|:---:|---|
| `Int.natCast_eq_zero` | ✓ | `omega` |
| `Int.sub_one_lt_iff` | — | `` |
| `Int.le_antisymm_iff` | — | `` |
| `Int.lt_toNat` | — | `` |
| `Int.le_iff_eq_or_lt` | — | `` |
| `Int.le_sub_one_iff` | — | `` |
| `Int.natCast_ne_zero_iff_pos` | — | `` |
| `Int.natCast_nonpos_iff` | ✓ | `omega` |
| `Int.le_add_one_iff` | — | `` |
| `Int.le_iff_lt_or_eq` | — | `` |

### `gen_v5_ns22_int_iff_omega_10x` (own pool = iff_omega_pair)

| theorem | solved | tactic emitted |
|---|:---:|---|
| `Int.natCast_eq_zero` | — | `` |
| `Int.sub_one_lt_iff` | — | `` |
| `Int.le_antisymm_iff` | — | `` |
| `Int.lt_toNat` | — | `` |
| `Int.le_iff_eq_or_lt` | — | `` |
| `Int.le_sub_one_iff` | — | `` |
| `Int.natCast_ne_zero_iff_pos` | — | `` |
| `Int.natCast_nonpos_iff` | — | `` |
| `Int.le_add_one_iff` | — | `` |
| `Int.le_iff_lt_or_eq` | — | `` |

### `gen_v5_ns22_int_fallback_omega_5x` (own pool = fallback_omega)

| theorem | solved | tactic emitted |
|---|:---:|---|
| `Int.emod_two_eq_zero_or_one` | ✓ | `omega` |
| `Int.lt_asymm` | ✓ | `omega` |
| `Int.natAbs_coe_sub_coe_le_of_le` | ✓ | `omega` |
| `Int.natAbs_add_of_nonpos` | ✓ | `omega` |
| `Int.lt_or_le` | ✓ | `omega` |
| `Int.natCast_pred_of_pos` | ✓ | `omega` |
| `Int.zero_le_ofNat` | ✓ | `omega` |
| `Int.lt_or_lt_of_ne` | ✓ | `omega` |
| `Int.le_or_lt` | ✓ | `omega` |
| `Int.le_natCast_sub` | ✓ | `omega` |
| `Int.neg_emod_two` | ✓ | `omega` |
| `Int.natAbs_coe_sub_coe_lt_of_lt` | ✓ | `omega` |
| `Int.le_of_eq` | ✓ | `omega` |

## Cross-family pool resolution

How many of the *other* pool's theorems each candidate solves. This is the cleanest evidence of broader transfer: training on family X solving family Y's theorems via the trained-family tactic (or any other shorter tactic).

### `gen_v5_ns22_int_iff_omega_5x`: 2/13

| theorem | tactic |
|---|---|
| `Int.natCast_pred_of_pos` | `omega` |
| `Int.le_of_eq` | `simp_all` |

### `gen_v5_ns22_int_iff_omega_10x`: 1/13

| theorem | tactic |
|---|---|
| `Int.lt_or_lt_of_ne` | `exact fun h => by omega` |

### `gen_v5_ns22_int_fallback_omega_5x`: 9/10

| theorem | tactic |
|---|---|
| `Int.sub_one_lt_iff` | `omega` |
| `Int.natCast_eq_zero` | `omega` |
| `Int.le_antisymm_iff` | `omega` |
| `Int.le_iff_eq_or_lt` | `omega` |
| `Int.natCast_ne_zero_iff_pos` | `omega` |
| `Int.le_sub_one_iff` | `omega` |
| `Int.natCast_nonpos_iff` | `omega` |
| `Int.le_add_one_iff` | `omega` |
| `Int.le_iff_lt_or_eq` | `omega` |

## Held-out Int gains (beyond all pool theorems)

### `gen_v5_ns22_int_iff_omega_5x`

- held-out Int theorems probed: 135
- NS12 baseline wins: 35
- candidate wins: 35
- **gains: 0**, losses: 0

### `gen_v5_ns22_int_iff_omega_10x`

- held-out Int theorems probed: 135
- NS12 baseline wins: 35
- candidate wins: 34
- **gains: 0**, losses: 1

### `gen_v5_ns22_int_fallback_omega_5x`

- held-out Int theorems probed: 135
- NS12 baseline wins: 35
- candidate wins: 35
- **gains: 0**, losses: 0

## Negative control (Set/Finset/demo)

### `gen_v5_ns22_int_iff_omega_5x`

| set | baseline | candidate | gains | losses |
|---|---:|---:|---:|---:|
| demo_v1 | 10 | 10 | 1 | **1** |
| ns17_set_extra | 18 | 18 | 0 | **0** |
| ns17_finset_extra | 12 | 13 | 1 | **0** |
| ns14_set_finset_extra | 13 | 11 | 0 | **2** |

### `gen_v5_ns22_int_iff_omega_10x`

| set | baseline | candidate | gains | losses |
|---|---:|---:|---:|---:|
| demo_v1 | 10 | 10 | 0 | **0** |
| ns17_set_extra | 18 | 19 | 1 | **0** |
| ns17_finset_extra | 12 | 14 | 2 | **0** |
| ns14_set_finset_extra | 13 | 13 | 0 | **0** |

### `gen_v5_ns22_int_fallback_omega_5x`

| set | baseline | candidate | gains | losses |
|---|---:|---:|---:|---:|
| demo_v1 | 10 | 10 | 0 | **0** |
| ns17_set_extra | 18 | 19 | 1 | **0** |
| ns17_finset_extra | 12 | 12 | 0 | **0** |
| ns14_set_finset_extra | 13 | 11 | 0 | **2** |

