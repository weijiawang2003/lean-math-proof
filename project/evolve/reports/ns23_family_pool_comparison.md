# NS23 — family pool comparison (original vs minimal-tactic relabel)

- theorems re-tested: 32
- unchanged labels: 18
- relabeled: 13
- unresolved (no battery tactic closed it): 1

## Cross-tabulation: original → minimal

| original family | → minimal family | count |
|---|---|---:|
| `aesop` | `aesop` | 6 |
| `simp_all` | `wrapper_original` | 3 |
| `iff_omega_pair` | `fallback_omega` | 9 |
| `iff_omega_pair` | `unresolved` | 1 |
| `fallback_omega` | `fallback_omega` | 12 |
| `fallback_omega` | `constructor_omega` | 1 |

## Per-namespace omega aggregate

**Aggregate of** `fallback_omega` + `iff_omega_pair` + `constructor_omega` + `split_ifs_omega`.

| namespace | unique | gate met? |
|---|---:|:---:|
| Int | **22** | ✓ |

## Gated pools (≥5 unique under minimal labels)

| family | namespace | unique | aggregate of |
|---|---|---:|---|
| `aesop` | Finset | **6** | — |
| `fallback_omega` | Int | **21** | — |
| `omega_aggregate` | Int | **22** | fallback_omega, iff_omega_pair, constructor_omega, split_ifs_omega |

## Aesop-irreducible theorems

These theorems are NOT closed by any tactic strictly simpler than `aesop` in the battery — they remain aesop-minimal and form the residual aesop pool.

| theorem | namespace | original family |
|---|---|---|
| `Finset.disjUnion_singleton` | Finset | aesop |
| `Finset.cons_eq_insert` | Finset | aesop |
| `Finset.coe_insert` | Finset | aesop |
| `Finset.coe_cons` | Finset | aesop |
| `Finset.card_insert_eq_ite` | Finset | aesop |
| `Finset.image_id` | Finset | aesop |

## Per-theorem detail

| theorem | namespace | orig | minimal | minimal tactic | changed |
|---|---|---|---|---|:---:|
| `Finset.disjUnion_singleton` | Finset | `aesop` | `aesop` | `aesop` | — |
| `Finset.cons_eq_insert` | Finset | `aesop` | `aesop` | `aesop` | — |
| `Finset.coe_insert` | Finset | `aesop` | `aesop` | `aesop` | — |
| `Finset.coe_cons` | Finset | `aesop` | `aesop` | `aesop` | — |
| `Finset.card_insert_eq_ite` | Finset | `aesop` | `aesop` | `aesop` | — |
| `Finset.image_id` | Finset | `aesop` | `aesop` | `aesop` | — |
| `Nat.mul_mod_mod` | Nat | `simp_all` | `wrapper_original` | `simp_all [Nat.add_mod, Nat.mul_mod, Nat.mod_eq_of_` | **✓** |
| `Nat.mod_mul_mod` | Nat | `simp_all` | `wrapper_original` | `simp_all [Nat.add_mod, Nat.mul_mod, Nat.mod_eq_of_` | **✓** |
| `Nat.add_mod_of_add_mod_lt` | Nat | `simp_all` | `wrapper_original` | `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]` | **✓** |
| `Int.le_add_one_iff` | Int | `iff_omega_pair` | `fallback_omega` | `omega` | **✓** |
| `Int.le_iff_lt_or_eq` | Int | `iff_omega_pair` | `fallback_omega` | `omega` | **✓** |
| `Int.emod_two_eq_zero_or_one` | Int | `fallback_omega` | `fallback_omega` | `omega` | — |
| `Int.le_of_eq` | Int | `fallback_omega` | `fallback_omega` | `omega` | — |
| `Int.natAbs_coe_sub_coe_lt_of_lt` | Int | `fallback_omega` | `fallback_omega` | `omega` | — |
| `Int.le_or_lt` | Int | `fallback_omega` | `fallback_omega` | `omega` | — |
| `Int.natAbs_coe_sub_coe_le_of_le` | Int | `fallback_omega` | `fallback_omega` | `omega` | — |
| `Int.zero_le_ofNat` | Int | `fallback_omega` | `constructor_omega` | `constructor <;> omega` | **✓** |
| `Int.lt_or_lt_of_ne` | Int | `fallback_omega` | `fallback_omega` | `omega` | — |
| `Int.natAbs_add_of_nonpos` | Int | `fallback_omega` | `fallback_omega` | `omega` | — |
| `Int.lt_asymm` | Int | `fallback_omega` | `fallback_omega` | `omega` | — |
| `Int.le_natCast_sub` | Int | `fallback_omega` | `fallback_omega` | `omega` | — |
| `Int.neg_emod_two` | Int | `fallback_omega` | `fallback_omega` | `omega` | — |
| `Int.lt_or_le` | Int | `fallback_omega` | `fallback_omega` | `omega` | — |
| `Int.natCast_pred_of_pos` | Int | `fallback_omega` | `fallback_omega` | `omega` | — |
| `Int.le_sub_one_iff` | Int | `iff_omega_pair` | `fallback_omega` | `omega` | **✓** |
| `Int.sub_one_lt_iff` | Int | `iff_omega_pair` | `fallback_omega` | `omega` | **✓** |
| `Int.le_antisymm_iff` | Int | `iff_omega_pair` | `fallback_omega` | `omega` | **✓** |
| `Int.le_iff_eq_or_lt` | Int | `iff_omega_pair` | `fallback_omega` | `omega` | **✓** |
| `Int.natCast_nonpos_iff` | Int | `iff_omega_pair` | `fallback_omega` | `omega` | **✓** |
| `Int.natCast_ne_zero_iff_pos` | Int | `iff_omega_pair` | `fallback_omega` | `omega` | **✓** |
| `Int.lt_toNat` | Int | `iff_omega_pair` | `unresolved` | `—` | — |
| `Int.natCast_eq_zero` | Int | `iff_omega_pair` | `fallback_omega` | `omega` | **✓** |
