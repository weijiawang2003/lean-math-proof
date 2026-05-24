# NS15 — Per-theorem solved-set comparison

Offline analysis of which theorems each base / routed / NS15 model proves on each eval set. Includes the oracle union as an upper bound for a router restricted to the listed candidates.

## `nat_defs_medium` (total 38)

Missing metrics for: `ns15_combined_all`, `ns15_nat_oversample`, `ns15_balanced_namespace`, `ns15_curriculum`, `ns15_routed`

| model | proved | union? | intersection? |
|---|---:|---:|---:|
| `gen_v5` | 3 | 9 | 3 |
| `ns11_combined` | 9 | 9 | 3 |
| `ns12_balanced` | 5 | 9 | 3 |
| `ns13_routed` | 9 | 9 | 3 |

**Oracle union (perfect router upper bound): 9/38**

Exclusive wins (this model proves it, no other does):
- (none)

## `nat_defs_large_v5` (total 65)

Missing metrics for: `ns15_combined_all`, `ns15_nat_oversample`, `ns15_balanced_namespace`, `ns15_curriculum`, `ns15_routed`

| model | proved | union? | intersection? |
|---|---:|---:|---:|
| `ns11_combined` | 13 | 13 | 6 |
| `ns12_balanced` | 6 | 13 | 6 |
| `ns13_routed` | 13 | 13 | 6 |

**Oracle union (perfect router upper bound): 13/65**

Exclusive wins (this model proves it, no other does):
- (none)

## `demo_v1` (total 15)

| model | proved | union? | intersection? |
|---|---:|---:|---:|
| `gen_v5` | 10 | 10 | 8 |
| `ns11_combined` | 8 | 10 | 8 |
| `ns12_balanced` | 10 | 10 | 8 |
| `ns13_routed` | 10 | 10 | 8 |
| `ns15_combined_all` | 8 | 10 | 8 |
| `ns15_nat_oversample` | 9 | 10 | 8 |
| `ns15_balanced_namespace` | 10 | 10 | 8 |
| `ns15_curriculum` | 9 | 10 | 8 |
| `ns15_routed` | 10 | 10 | 8 |

**Oracle union (perfect router upper bound): 10/15**

Exclusive wins (this model proves it, no other does):
- (none)

## `ns14_nat_extra` (total 20)

| model | proved | union? | intersection? |
|---|---:|---:|---:|
| `ns13_routed` | 0 | 9 | 0 |
| `ns15_combined_all` | 1 | 9 | 0 |
| `ns15_nat_oversample` | 9 | 9 | 0 |
| `ns15_balanced_namespace` | 0 | 9 | 0 |
| `ns15_curriculum` | 9 | 9 | 0 |
| `ns15_routed` | 9 | 9 | 0 |

**Oracle union (perfect router upper bound): 9/20**

Exclusive wins (this model proves it, no other does):
- (none)

## `ns14_set_finset_extra` (total 20)

| model | proved | union? | intersection? |
|---|---:|---:|---:|
| `ns13_routed` | 13 | 14 | 10 |
| `ns15_combined_all` | 12 | 14 | 10 |
| `ns15_nat_oversample` | 11 | 14 | 10 |
| `ns15_balanced_namespace` | 10 | 14 | 10 |
| `ns15_curriculum` | 10 | 14 | 10 |
| `ns15_routed` | 13 | 14 | 10 |

**Oracle union (perfect router upper bound): 14/20**

Exclusive wins (this model proves it, no other does):
- `ns15_combined_all` (1): `Set.inter_nonempty_iff_exists_left`

## `ns14_mixed_easy` (total 15)

| model | proved | union? | intersection? |
|---|---:|---:|---:|
| `ns15_combined_all` | 8 | 13 | 6 |
| `ns15_nat_oversample` | 11 | 13 | 6 |
| `ns15_balanced_namespace` | 6 | 13 | 6 |
| `ns15_curriculum` | 11 | 13 | 6 |
| `ns15_routed` | 12 | 13 | 6 |

**Oracle union (perfect router upper bound): 13/15**

Exclusive wins (this model proves it, no other does):
- `ns15_routed` (1): `Set.not_subset`
- `ns15_combined_all` (1): `Set.inter_nonempty_iff_exists_left`

## `ns14_mixed_medium` (total 15)

| model | proved | union? | intersection? |
|---|---:|---:|---:|
| `ns15_combined_all` | 1 | 3 | 1 |
| `ns15_nat_oversample` | 3 | 3 | 1 |
| `ns15_balanced_namespace` | 1 | 3 | 1 |
| `ns15_curriculum` | 3 | 3 | 1 |
| `ns15_routed` | 3 | 3 | 1 |

**Oracle union (perfect router upper bound): 3/15**

Exclusive wins (this model proves it, no other does):
- (none)

## NS14 wrapper-only Nat transfer

Did any NS15 raw model learn the 8 NS14 wrapper-only Nat wins?

| model | learned | / target |
|---|---:|---:|
| `ns13_routed` | 0 | 8 |
| `ns15_combined_all` | 1 | 8 |
| `ns15_nat_oversample` | 8 | 8 |
| `ns15_balanced_namespace` | 0 | 8 |
| `ns15_curriculum` | 8 | 8 |
| `ns15_routed` | 8 | 8 |

Wins per model:
- `ns13_routed`: (none)
- `ns15_combined_all`: `Nat.pred_eq_succ_iff`
- `ns15_nat_oversample`: `Nat.add_sub_sub_cancel`, `Nat.lt_of_lt_pred`, `Nat.lt_sub_iff_add_lt'`, `Nat.pred_eq_succ_iff`, `Nat.pred_sub`, `Nat.sub_add_sub_cancel`, `Nat.sub_lt_sub_iff_right`, `Nat.sub_sub_sub_cancel_right`
- `ns15_balanced_namespace`: (none)
- `ns15_curriculum`: `Nat.add_sub_sub_cancel`, `Nat.lt_of_lt_pred`, `Nat.lt_sub_iff_add_lt'`, `Nat.pred_eq_succ_iff`, `Nat.pred_sub`, `Nat.sub_add_sub_cancel`, `Nat.sub_lt_sub_iff_right`, `Nat.sub_sub_sub_cancel_right`
- `ns15_routed`: `Nat.add_sub_sub_cancel`, `Nat.lt_of_lt_pred`, `Nat.lt_sub_iff_add_lt'`, `Nat.pred_eq_succ_iff`, `Nat.pred_sub`, `Nat.sub_add_sub_cancel`, `Nat.sub_lt_sub_iff_right`, `Nat.sub_sub_sub_cancel_right`

## demo_v1 regression retention

Are `Set.subset_univ` and `Set.empty_subset` still proved?

| model | retained | / target |
|---|---:|---:|
| `gen_v5` | 2 | 2 |
| `ns11_combined` | 0 | 2 |
| `ns12_balanced` | 2 | 2 |
| `ns13_routed` | 2 | 2 |
| `ns15_combined_all` | 0 | 2 |
| `ns15_nat_oversample` | 1 | 2 |
| `ns15_balanced_namespace` | 2 | 2 |
| `ns15_curriculum` | 1 | 2 |
| `ns15_routed` | 2 | 2 |

## ns14_set_finset_extra retention

| model | proved |
|---|---:|
| `ns13_routed` | 13 |
| `ns15_combined_all` | 12 |
| `ns15_nat_oversample` | 11 |
| `ns15_balanced_namespace` | 10 |
| `ns15_curriculum` | 10 |
| `ns15_routed` | 13 |

