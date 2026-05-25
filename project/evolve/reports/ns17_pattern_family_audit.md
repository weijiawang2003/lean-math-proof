# NS17 — Pattern family audit

Inventory of supervised (state, tactic) pairs we've accumulated through NS16, grouped by *tactic family*. The hypothesis from NS15→NS16 was that transfer requires enough rows *per family*, not per dataset; this audit checks that claim against the data we have.

## Inputs

| source | rows |
|---|---:|
| `ns11_combined` | 5729 |
| `ns14_combined` | 30 |
| `ns16_wrapper_only` | 19 |
| `trace_close_pre_dedup` | 139 |
| `trace_close_post_dedup` | 90 |

## Partition: `v5_base` (5577 rows, 1 thms)

| family | rows | thms | wrapper-only | held-out | example |
|---|---:|---:|---:|---:|---|
| `simp_baseline` | 2941 | 1 | 0 | 0 | `simp` |
| `other` | 1151 | 1 | 0 | 0 | `apply h` |
| `nat_simp_arith` | 711 | 1 | 0 | 0 | `simp [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]` |
| `fallback_aesop` | 542 | 1 | 0 | 0 | `aesop` |
| `set_subset_simp` | 111 | 1 | 0 | 0 | `simp [Set.subset_def]` |
| `set_ext_simp` | 92 | 1 | 0 | 0 | `simp [Set.ext_iff]` |
| `rw_named` | 10 | 1 | 0 | 0 | `rw [Finset.disjoint_left]` |
| `fallback_omega` | 8 | 1 | 0 | 0 | `omega` |
| `fallback_rfl` | 7 | 1 | 0 | 0 | `rfl` |
| `exact_named` | 4 | 1 | 0 | 0 | `exact Set.mem_union_left _ hx` |

## Partition: `evolved` (201 rows, 86 thms)

| family | rows | thms | wrapper-only | held-out | example |
|---|---:|---:|---:|---:|---|
| `fallback_omega` | 58 | 32 | 0 | 0 | `omega` |
| `other` | 49 | 35 | 9 | 7 | `by_cases hc : c = 0 <;> [simp [hc]; exact (Nat.le_div_iff_mul_le' (…` |
| `iff_omega_pair` | 28 | 27 | 0 | 0 | `exact ⟨fun h => by omega, fun h => by omega⟩` |
| `constructor_omega` | 16 | 16 | 0 | 0 | `constructor <;> intro h_split <;> omega` |
| `nat_simp_arith` | 14 | 11 | 4 | 0 | `simp [Nat.add_mod, Nat.mod_eq_of_lt]` |
| `fallback_aesop` | 13 | 13 | 0 | 0 | `aesop` |
| `set_subset_simp` | 5 | 5 | 0 | 0 | `simp [Set.subset_def]` |
| `set_ext_simp` | 5 | 5 | 0 | 0 | `simp [Set.ext_iff]` |
| `simp_baseline` | 4 | 4 | 0 | 0 | `simp [List.length_cons]` |
| `exact_named` | 3 | 2 | 2 | 1 | `exact (Nat.le_div_iff_mul_le hb).mpr (by simpa using hba)` |
| `split_ifs_omega` | 2 | 1 | 1 | 0 | `split_ifs <;> omega` |
| `nat_div_rw` | 2 | 2 | 2 | 2 | `rw [Nat.div_lt_iff_lt_mul hb, Nat.one_mul]` |
| `rw_named` | 1 | 1 | 0 | 0 | `rw [Nat.add_comm]` |
| `apply_named` | 1 | 1 | 1 | 1 | `apply Nat.div_lt_iff_lt_mul` |

## Partition: `traces_close_only` (90 rows, 90 thms)

| family | rows | thms | wrapper-only | held-out | example |
|---|---:|---:|---:|---:|---|
| `iff_omega_pair` | 28 | 28 | 0 | 0 | `exact ⟨fun h => by omega, fun h => by omega⟩` |
| `fallback_omega` | 19 | 19 | 0 | 0 | `omega` |
| `fallback_aesop` | 11 | 11 | 0 | 0 | `aesop` |
| `other` | 10 | 10 | 0 | 6 | `by_cases hc : c = 0 <;> [simp [hc]; exact (Nat.le_div_iff_mul_le' (…` |
| `set_subset_simp` | 6 | 6 | 0 | 0 | `simp [Set.subset_def]` |
| `nat_simp_arith` | 5 | 5 | 0 | 0 | `simp [Nat.mul_one]` |
| `set_ext_simp` | 5 | 5 | 0 | 0 | `simp [Set.ext_iff]` |
| `nat_div_rw` | 2 | 2 | 0 | 2 | `rw [Nat.div_lt_iff_lt_mul hb, Nat.one_mul]` |
| `exact_named` | 2 | 2 | 0 | 1 | `exact (Nat.le_div_iff_mul_le hb).mpr (by simpa using hba)` |
| `split_ifs_omega` | 1 | 1 | 0 | 0 | `split_ifs <;> omega` |
| `simp_baseline` | 1 | 1 | 0 | 0 | `simp [Nat.one_mul]` |

## Family-by-family commentary

- `iff_omega_pair` — The NS14 winner. Look for: row count, namespace breadth, whether trace partition has more rows than JSONL partition.
- `iff_omega_left_only` — Variant of the winner with one side not omega; smaller pool.
- `split_ifs_omega` — Reaches Nat if-then-else theorems.
- `nat_simp_arith` — simp [Nat.add_mod, Nat.mul_…] patterns; broad family with many sub-variants.
- `nat_div_rw` — rw [Nat.div_lt_iff_…]; high homogeneity but narrow coverage.
- `set_subset_simp` — demo_v1 retention driver.
- `set_ext_simp` — Set.ext_iff emissions.
- `fallback_omega` — Bare omega — should already be in raw.
- `fallback_aesop` — Heuristic Mathlib closer.
- `rw_named` — Any rw of a named lemma.
- `apply_named` — Any apply of a named lemma.

## Headline numbers

- Evolved supervision total: **201 rows / 86 theorems**.

  - `iff_omega_pair`: 28 rows / 27 thms (0 wrapper-only)
  - `fallback_omega`: 58 rows / 32 thms (0 wrapper-only)
  - `split_ifs_omega`: 2 rows / 1 thms (1 wrapper-only)
  - `nat_div_rw`: 2 rows / 2 thms (2 wrapper-only)
  - `nat_simp_arith`: 14 rows / 11 thms (4 wrapper-only)
  - `set_subset_simp`: 5 rows / 5 thms (0 wrapper-only)

From raw wrapper trace closings (deduplicated by state+tactic, not yet filtered to wrapper-only):

  - `iff_omega_pair` traces: 28 rows / 28 thms
  - `fallback_omega` traces: 19 rows / 19 thms
  - `split_ifs_omega` traces: 1 rows / 1 thms
  - `nat_div_rw` traces: 2 rows / 2 thms
  - `nat_simp_arith` traces: 5 rows / 5 thms
  - `set_subset_simp` traces: 6 rows / 6 thms
  - `fallback_aesop` traces: 11 rows / 11 thms

## NS18 transfer-readiness gate

A family is a strong NS18 training candidate if **all** of:
- ≥ 10 wrapper-only rows in the evolved partition, OR
- ≥ 20 close transitions across distinct theorems in traces,
- consistent tactic surface (small example-tactic count),
- a held-out sibling theorem surface exists to evaluate.

| family | evolved rows | unique thms | wrapper-only | gate |
|---|---:|---:|---:|---|
| `fallback_omega` | 58 | 32 | 0 | PASS |
| `other` | 49 | 35 | 9 | PASS |
| `iff_omega_pair` | 28 | 27 | 0 | PASS |
| `constructor_omega` | 16 | 16 | 0 | PASS |
| `nat_simp_arith` | 14 | 11 | 4 | PASS |
| `fallback_aesop` | 13 | 13 | 0 | PASS |
| `set_subset_simp` | 5 | 5 | 0 | fail |
| `set_ext_simp` | 5 | 5 | 0 | fail |
| `simp_baseline` | 4 | 4 | 0 | fail |
| `exact_named` | 3 | 2 | 2 | fail |
| `split_ifs_omega` | 2 | 1 | 1 | fail |
| `nat_div_rw` | 2 | 2 | 2 | fail |
| `rw_named` | 1 | 1 | 0 | fail |
| `apply_named` | 1 | 1 | 1 | fail |

