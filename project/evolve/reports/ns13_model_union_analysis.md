# NS13 — Per-theorem model union analysis

Offline diff of which theorems each base-model variant proves on each theorem set. Includes the oracle union (any-model-proves-it) as an upper bound for a perfect router.

## `nat_defs_medium` (total 38)

| model | proved | union? | intersection? |
|---|---:|---:|---:|
| `gen_v5` | 3 | 9 | 3 |
| `ns11_combined` | 9 | 9 | 3 |
| `ns12_balanced` | 5 | 9 | 3 |
| `ns12_replay` | 5 | 9 | 3 |
| `ns12_low_lr` | 3 | 9 | 3 |
| `routed` | 9 | 9 | 3 |

**Oracle union (upper bound for a perfect router): 9/38**

Solved by every model in the comparison (intersection size 3):
- `Nat.lt_iff_add_one_le`
- `Nat.pred_eq_of_eq_succ`
- `Nat.succ_succ_ne_one`

Exclusive wins (this model proves it, no other does):
- (none)

**Router gap**: none — routed matches the union of all single-model wins.


## `nat_defs_large_v5` (total 65)

| model | proved | union? | intersection? |
|---|---:|---:|---:|
| `ns11_combined` | 13 | 13 | 6 |
| `ns12_balanced` | 6 | 13 | 6 |
| `ns12_replay` | 6 | 13 | 6 |
| `routed` | 13 | 13 | 6 |

**Oracle union (upper bound for a perfect router): 13/65**

Solved by every model in the comparison (intersection size 6):
- `Nat.add_eq_left`
- `Nat.add_eq_right`
- `Nat.lt_iff_add_one_le`
- `Nat.pred_eq_of_eq_succ`
- `Nat.sub_eq_of_eq_add'`
- `Nat.succ_succ_ne_one`

Exclusive wins (this model proves it, no other does):
- (none)

**Router gap**: none — routed matches the union of all single-model wins.


## `demo_v1` (total 15)

| model | proved | union? | intersection? |
|---|---:|---:|---:|
| `gen_v5` | 10 | 10 | 8 |
| `ns11_combined` | 8 | 10 | 8 |
| `ns12_balanced` | 10 | 10 | 8 |
| `ns12_replay` | 9 | 10 | 8 |
| `ns12_low_lr` | 8 | 10 | 8 |
| `routed` | 10 | 10 | 8 |

**Oracle union (upper bound for a perfect router): 10/15**

Solved by every model in the comparison (intersection size 8):
- `Set.empty_union`
- `Set.inter_comm`
- `Set.inter_univ`
- `Set.mem_inter_iff`
- `Set.mem_union`
- `Set.union_comm`
- `Set.union_empty`
- `Set.univ_inter`

Exclusive wins (this model proves it, no other does):
- (none)

**Router gap**: none — routed matches the union of all single-model wins.


