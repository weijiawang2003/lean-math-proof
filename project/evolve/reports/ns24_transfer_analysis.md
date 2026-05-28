# NS24 — Int minimal-omega transfer vs absorption analysis

Baseline = NS22 `gen_v5_ns22_int_fallback_omega_5x`. Pool groups under NS23 repaired labels: old_ns22_omega=12, relabeled_iff=9, constructor=1.

## Summary

| candidate | Int NS12 | Int NS22 | Int cand | Δ vs NS22 | relabeled_iff NS22→cand | held-out gains | held-out losses | demo losses | verdict |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|---|
| `gen_v5_ns24_int_minimal_omega_5x` | 35 | 57 | **57** | +0 | 9→9 | 1 | 1 | 0 | **reproduction_near_null** |
| `gen_v5_ns24_int_minimal_omega_10x` | 35 | 57 | **58** | +1 | 9→9 | 1 | 0 | 0 | **marginal_gain** |
| `gen_v5_ns24_int_minimal_omega_plus_constructor_5x` | 35 | 57 | **58** | +1 | 9→9 | 1 | 0 | 0 | **marginal_gain** |
| `gen_v5_ns24_int_minimal_omega_5x_from_ns12` | 35 | 57 | **58** | +1 | 9→9 | 1 | 0 | 0 | **marginal_gain** |

## `gen_v5_ns24_int_minimal_omega_5x`

### Pool-group resolution (NS12 / NS22 / candidate)

| group | size | NS12 | NS22 | candidate |
|---|---:|---:|---:|---:|
| old_ns22_omega | 12 | 0 | 12 | **12** |
| relabeled_iff | 9 | 0 | 9 | **9** |
| constructor | 1 | 0 | 1 | **1** |

### Per-theorem (relabeled_iff group — the NS24 test)

| theorem | NS22 | candidate | candidate tactic |
|---|:---:|:---:|---|
| `Int.le_add_one_iff` | ✓ | ✓ | `omega` |
| `Int.le_iff_lt_or_eq` | ✓ | ✓ | `omega` |
| `Int.le_sub_one_iff` | ✓ | ✓ | `omega` |
| `Int.sub_one_lt_iff` | ✓ | ✓ | `omega` |
| `Int.le_antisymm_iff` | ✓ | ✓ | `omega` |
| `Int.le_iff_eq_or_lt` | ✓ | ✓ | `omega` |
| `Int.natCast_nonpos_iff` | ✓ | ✓ | `omega` |
| `Int.natCast_ne_zero_iff_pos` | ✓ | ✓ | `omega` |
| `Int.natCast_eq_zero` | ✓ | ✓ | `omega` |

### Held-out Int (not in 22-pool)

- probed: 136; NS22 wins 35, candidate wins 35
- **gains vs NS22: 1**, losses: 1

| gained theorem | tactic | set |
|---|---|---|
| `Int.cast_ite` | `aesop` | cx2_int_mixed |

Losses: `Bool.lt_iff`

### Emitted tactic distribution (solved Int): {'omega': 28, 'aesop': 5, 'simp_all': 24}

### Negative control

| set | routed away? | NS22 | candidate | gains | losses |
|---|:---:|---:|---:|---:|---:|
| demo_v1 | no | 10 | 11 | 1 | **0** |
| ns17_set_extra | yes | 19 | 18 | 0 | **1** |
| ns17_finset_extra | yes | 12 | 14 | 2 | **0** |
| ns14_set_finset_extra | yes | 11 | 13 | 2 | **0** |

## `gen_v5_ns24_int_minimal_omega_10x`

### Pool-group resolution (NS12 / NS22 / candidate)

| group | size | NS12 | NS22 | candidate |
|---|---:|---:|---:|---:|
| old_ns22_omega | 12 | 0 | 12 | **12** |
| relabeled_iff | 9 | 0 | 9 | **9** |
| constructor | 1 | 0 | 1 | **1** |

### Per-theorem (relabeled_iff group — the NS24 test)

| theorem | NS22 | candidate | candidate tactic |
|---|:---:|:---:|---|
| `Int.le_add_one_iff` | ✓ | ✓ | `omega` |
| `Int.le_iff_lt_or_eq` | ✓ | ✓ | `omega` |
| `Int.le_sub_one_iff` | ✓ | ✓ | `omega` |
| `Int.sub_one_lt_iff` | ✓ | ✓ | `omega` |
| `Int.le_antisymm_iff` | ✓ | ✓ | `omega` |
| `Int.le_iff_eq_or_lt` | ✓ | ✓ | `omega` |
| `Int.natCast_nonpos_iff` | ✓ | ✓ | `omega` |
| `Int.natCast_ne_zero_iff_pos` | ✓ | ✓ | `omega` |
| `Int.natCast_eq_zero` | ✓ | ✓ | `omega` |

### Held-out Int (not in 22-pool)

- probed: 136; NS22 wins 35, candidate wins 36
- **gains vs NS22: 1**, losses: 0

| gained theorem | tactic | set |
|---|---|---|
| `Int.cast_ite` | `aesop` | cx2_int_mixed |

### Emitted tactic distribution (solved Int): {'omega': 28, 'aesop': 5, 'simp_all': 24, 'other': 1}

### Negative control

| set | routed away? | NS22 | candidate | gains | losses |
|---|:---:|---:|---:|---:|---:|
| demo_v1 | no | 10 | 11 | 1 | **0** |
| ns17_set_extra | yes | 19 | 17 | 0 | **2** |
| ns17_finset_extra | yes | 12 | 14 | 2 | **0** |
| ns14_set_finset_extra | yes | 11 | 12 | 2 | **1** |

## `gen_v5_ns24_int_minimal_omega_plus_constructor_5x`

### Pool-group resolution (NS12 / NS22 / candidate)

| group | size | NS12 | NS22 | candidate |
|---|---:|---:|---:|---:|
| old_ns22_omega | 12 | 0 | 12 | **12** |
| relabeled_iff | 9 | 0 | 9 | **9** |
| constructor | 1 | 0 | 1 | **1** |

### Per-theorem (relabeled_iff group — the NS24 test)

| theorem | NS22 | candidate | candidate tactic |
|---|:---:|:---:|---|
| `Int.le_add_one_iff` | ✓ | ✓ | `omega` |
| `Int.le_iff_lt_or_eq` | ✓ | ✓ | `omega` |
| `Int.le_sub_one_iff` | ✓ | ✓ | `omega` |
| `Int.sub_one_lt_iff` | ✓ | ✓ | `omega` |
| `Int.le_antisymm_iff` | ✓ | ✓ | `omega` |
| `Int.le_iff_eq_or_lt` | ✓ | ✓ | `omega` |
| `Int.natCast_nonpos_iff` | ✓ | ✓ | `omega` |
| `Int.natCast_ne_zero_iff_pos` | ✓ | ✓ | `omega` |
| `Int.natCast_eq_zero` | ✓ | ✓ | `omega` |

### Held-out Int (not in 22-pool)

- probed: 136; NS22 wins 35, candidate wins 36
- **gains vs NS22: 1**, losses: 0

| gained theorem | tactic | set |
|---|---|---|
| `Int.cast_ite` | `aesop` | cx2_int_mixed |

### Emitted tactic distribution (solved Int): {'omega': 28, 'aesop': 6, 'simp_all': 23, 'other': 1}

### Negative control

| set | routed away? | NS22 | candidate | gains | losses |
|---|:---:|---:|---:|---:|---:|
| demo_v1 | — | n/a | n/a | n/a | n/a |
| ns17_set_extra | — | n/a | n/a | n/a | n/a |
| ns17_finset_extra | — | n/a | n/a | n/a | n/a |
| ns14_set_finset_extra | — | n/a | n/a | n/a | n/a |

## `gen_v5_ns24_int_minimal_omega_5x_from_ns12`

### Pool-group resolution (NS12 / NS22 / candidate)

| group | size | NS12 | NS22 | candidate |
|---|---:|---:|---:|---:|
| old_ns22_omega | 12 | 0 | 12 | **12** |
| relabeled_iff | 9 | 0 | 9 | **9** |
| constructor | 1 | 0 | 1 | **1** |

### Per-theorem (relabeled_iff group — the NS24 test)

| theorem | NS22 | candidate | candidate tactic |
|---|:---:|:---:|---|
| `Int.le_add_one_iff` | ✓ | ✓ | `omega` |
| `Int.le_iff_lt_or_eq` | ✓ | ✓ | `omega` |
| `Int.le_sub_one_iff` | ✓ | ✓ | `omega` |
| `Int.sub_one_lt_iff` | ✓ | ✓ | `omega` |
| `Int.le_antisymm_iff` | ✓ | ✓ | `omega` |
| `Int.le_iff_eq_or_lt` | ✓ | ✓ | `omega` |
| `Int.natCast_nonpos_iff` | ✓ | ✓ | `omega` |
| `Int.natCast_ne_zero_iff_pos` | ✓ | ✓ | `omega` |
| `Int.natCast_eq_zero` | ✓ | ✓ | `omega` |

### Held-out Int (not in 22-pool)

- probed: 136; NS22 wins 35, candidate wins 36
- **gains vs NS22: 1**, losses: 0

| gained theorem | tactic | set |
|---|---|---|
| `Int.cast_ite` | `aesop` | cx2_int_mixed |

### Emitted tactic distribution (solved Int): {'omega': 28, 'aesop': 5, 'simp_all': 24, 'other': 1}

### Negative control

| set | routed away? | NS22 | candidate | gains | losses |
|---|:---:|---:|---:|---:|---:|
| demo_v1 | — | n/a | n/a | n/a | n/a |
| ns17_set_extra | — | n/a | n/a | n/a | n/a |
| ns17_finset_extra | — | n/a | n/a | n/a | n/a |
| ns14_set_finset_extra | — | n/a | n/a | n/a | n/a |

