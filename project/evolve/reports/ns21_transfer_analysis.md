# NS21 — transfer vs memorization analysis

## Summary

| ckpt | pool solved | held-out Finset gains | neg-control losses | verdict |
|---|---:|---:|---:|---|
| `gen_v5_ns21_finset_aesop_10x` | 5/6 | 0 | 2 | **memorization** |
| `gen_v5_ns21_finset_aesop_20x` | 5/6 | 0 | 2 | **memorization** |
| `gen_v5_ns21_finset_aesop_minimal` | 5/6 | 0 | 4 | **memorization** |

## Pool detail

Which of the 6 training-pool theorems each candidate solves raw, and what tactic it emits.

### `gen_v5_ns21_finset_aesop_10x`

| theorem | solved? | tactic |
|---|:---:|---|
| `Finset.card_insert_eq_ite` | ✓ | `aesop` |
| `Finset.coe_cons` | — | `` |
| `Finset.coe_insert` | ✓ | `aesop` |
| `Finset.cons_eq_insert` | ✓ | `aesop` |
| `Finset.disjUnion_singleton` | ✓ | `aesop` |
| `Finset.image_id` | ✓ | `aesop` |

### `gen_v5_ns21_finset_aesop_20x`

| theorem | solved? | tactic |
|---|:---:|---|
| `Finset.card_insert_eq_ite` | ✓ | `aesop` |
| `Finset.coe_cons` | — | `` |
| `Finset.coe_insert` | ✓ | `aesop` |
| `Finset.cons_eq_insert` | ✓ | `aesop` |
| `Finset.disjUnion_singleton` | ✓ | `aesop` |
| `Finset.image_id` | ✓ | `aesop` |

### `gen_v5_ns21_finset_aesop_minimal`

| theorem | solved? | tactic |
|---|:---:|---|
| `Finset.card_insert_eq_ite` | ✓ | `aesop` |
| `Finset.coe_cons` | — | `` |
| `Finset.coe_insert` | ✓ | `aesop` |
| `Finset.cons_eq_insert` | ✓ | `aesop` |
| `Finset.disjUnion_singleton` | ✓ | `aesop` |
| `Finset.image_id` | ✓ | `aesop` |

## Held-out Finset transfer

### `gen_v5_ns21_finset_aesop_10x`

| set | held-out | baseline wins | candidate wins | gains | losses |
|---|---:|---:|---:|---:|---:|
| ns17_finset_extra | 27 | 12 | 12 | **0** | 0 |
| cx1_finset_image_filter | 98 | 28 | 28 | **0** | 0 |
| ns20_finset_aesop_extra_easy | 0 | 0 | 0 | **0** | 0 |
| ns20_finset_aesop_extra_medium | 16 | 7 | 7 | **0** | 0 |
| ns20_finset_aesop_extra_hard | 0 | 0 | 0 | **0** | 0 |

### `gen_v5_ns21_finset_aesop_20x`

| set | held-out | baseline wins | candidate wins | gains | losses |
|---|---:|---:|---:|---:|---:|
| ns17_finset_extra | 27 | 12 | 12 | **0** | 0 |
| cx1_finset_image_filter | 98 | 28 | 28 | **0** | 0 |
| ns20_finset_aesop_extra_easy | 0 | 0 | 0 | **0** | 0 |
| ns20_finset_aesop_extra_medium | 16 | 7 | 7 | **0** | 0 |
| ns20_finset_aesop_extra_hard | 0 | 0 | 0 | **0** | 0 |

### `gen_v5_ns21_finset_aesop_minimal`

| set | held-out | baseline wins | candidate wins | gains | losses |
|---|---:|---:|---:|---:|---:|
| ns17_finset_extra | 27 | 12 | 12 | **0** | 0 |
| cx1_finset_image_filter | 98 | 28 | 28 | **0** | 0 |
| ns20_finset_aesop_extra_easy | 0 | 0 | 0 | **0** | 0 |
| ns20_finset_aesop_extra_medium | 16 | 7 | 7 | **0** | 0 |
| ns20_finset_aesop_extra_hard | 0 | 0 | 0 | **0** | 0 |

## Negative control (Set/demo)

### `gen_v5_ns21_finset_aesop_10x`

| set | baseline wins | candidate wins | gains | losses |
|---|---:|---:|---:|---:|
| ns17_set_extra | 18 | 19 | 1 | **0** |
| ns14_set_finset_extra | 13 | 11 | 0 | **2** |
| demo_v1 | 10 | 10 | 0 | **0** |

### `gen_v5_ns21_finset_aesop_20x`

| set | baseline wins | candidate wins | gains | losses |
|---|---:|---:|---:|---:|
| ns17_set_extra | 18 | 18 | 1 | **1** |
| ns14_set_finset_extra | 13 | 12 | 0 | **1** |
| demo_v1 | 10 | 11 | 1 | **0** |

### `gen_v5_ns21_finset_aesop_minimal`

| set | baseline wins | candidate wins | gains | losses |
|---|---:|---:|---:|---:|
| ns17_set_extra | 18 | 18 | 1 | **1** |
| ns14_set_finset_extra | 13 | 10 | 0 | **3** |
| demo_v1 | 10 | 10 | 0 | **0** |

