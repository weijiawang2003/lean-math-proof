# CX2 — Int / iff_omega catalog mining

**Branch:** `cx2-int-iff-omega-mining`
**Parent:** NS21 commit `e48409d`.
**Goal:** grow the Int `iff_omega_pair` wrapper-only-vs-NS9 pool from
2 unique wins (CX1's secondary finding) to ≥5, the NS22 training gate.
**Outcome:** **both Int wrapper-only families now exceed the gate**:
`iff_omega_pair / Int` at **10 unique** (gate ≥5) and a previously
unrecognized `fallback_omega / Int` at **13 unique**. CX2 was a
mining-only arc — no training, no checkpoint or NS9-genome changes.

## 1. NS21 recap

NS21 trained a narrow Finset/aesop imitation model (`gen_v5_ns21_finset_aesop_20x`),
routed Finset goals to it, and demonstrated honest memorization on
the 5/6 pool theorems with no broader Finset transfer. Held-out gains
were zero because NS12 balanced already emitted `aesop` on most of
the held-out surface. Conclusion: future training gains require a
**fresh namespace** the base model has no prior on. **Int was the
identified next target** because the iff-omega pattern from NS15
generalized directly from Nat to Int with only 2 wrapper-only wins,
suggesting a high-yield mining surface.

## 2. Int catalog audit (Stage 1)

`scripts/cx2_int_catalog_audit.py` reads
`project/data/cx1_available_theorems.json` (120 LeanDojo-verified Int
theorems from `Mathlib/Data/Int/{Defs,Bitwise,GCD}.lean`) and extends
with a regex source-scan of 6 additional Int Mathlib files
(`ModEq.lean`, `Order/Lemmas.lean`, `Order/Units.lean`, `Lemmas.lean`,
`SuccPred.lean`, `Cast/Lemmas.lean`).

**Result:** 216 total Int candidates (CX1: 120 + CX2 fresh: 96).

Tag-distribution highlights:

| tag | count |
|---|---:|
| le_lt_order | 51 |
| dvd_gcd_lcm | 51 |
| add_sub_arith | 38 |
| iff_candidate | 24 |
| cast_natCast | 24 |
| mod_div | 17 |
| bitwise | 30 |

After removing bitwise / dvd-gcd-lcm / already-probed candidates:

- **iff_omega candidates: 13** (le/lt/add/sub with iff, not bitwise/dvd)
- **omega-only candidates: 53** (le/lt/add/sub without iff)
- **cast candidates: 24** (norm_cast → omega closure)

Output: `project/data/cx2_int_catalog_audit_meta.json`,
`project/evolve/reports/cx2_int_catalog_audit.md`.

## 3. Theorem sets (Stage 2)

`scripts/build_cx2_theorem_sets.py` excludes the 46 Int theorems
already probed in `cx1_bool_option_int` plus the 3 known
wrapper-only Int wins, leaving 78 fresh Int candidates partitioned
across 4 mining surfaces:

| set | size | composition |
|---|---:|---|
| `cx2_int_iff_omega_easy` | 12 | fresh iff/le/lt/add/sub candidates |
| `cx2_int_iff_omega_medium` | 3 | cast+iff candidates (norm_cast lead-in) |
| `cx2_int_order_arith` | 50 | omega-only le/lt/add/sub |
| `cx2_int_mixed` | 13 | remaining Int arithmetic |

Output: `project/evolve/routing/cx2_theorem_sets.json`; loaded via
`tasks.py:_load_cx2_sets()`.

## 4. Eval matrix (Stage 3)

Driver: `scripts/cx2_run_eval.sh` with the NS15 router (Set/Finset
routed via NS12 balanced; Int falls through to default = NS12, since
neither NS15 nor NS21 routes Int). `--top-k 8 --max-steps 8` per cell.

| set | size | raw (NS15-routed) | NS9 wrap | wrapper-only |
|---|---:|---:|---:|---:|
| `cx2_int_iff_omega_easy` | 12 | 1 | 5 | **+4** |
| `cx2_int_iff_omega_medium` | 3 | 0 | 2 | **+2** |
| `cx2_int_order_arith` | 50 | 4 | 16 | **+12** |
| `cx2_int_mixed` | 13 | 4 | 6 | **+2** |
| **total** | 78 | 9 | 29 | **+20** |

Strike rate: 20/78 = **26% wrapper-only**, vastly above the 2.5%
strike rate that CX1 (`cx1_bool_option_int`) achieved. The Int
surface is dramatically underserved by the routed NS12 base model.

## 5. Stage 4 — experimental wrapper (skipped)

The NS9 best genome wrapper produced 20 wrapper-only Int wins
without modification — well above the gate. The optional
`cx2_int_iff_omega_wrapper.json` experimental variant was not
necessary and was not built.

## 6. Combined Int pool (Stage 5)

`scripts/cx2_extract_pool.py` aggregates CX1 + CX2 Int
wrapper-only-vs-NS9 wins by tactic family:

| family | unique | gate | source breakdown |
|---|---:|:---:|---|
| **`iff_omega_pair`** | **10** | ✓ | CX1 cx1_bool_option_int: 2, CX2 4 sets: 8 |
| **`fallback_omega`** | **13** | ✓ | CX1: 1, CX2: 12 |

Detail in `project/evolve/reports/cx2_pool_summary.md`.

### `iff_omega_pair / Int` (10 unique, gate ≥5 ✓)

All emit `exact ⟨fun h => by omega, fun h => by omega⟩`:

```
Int.le_add_one_iff           (CX1)
Int.le_iff_lt_or_eq          (CX1)
Int.le_sub_one_iff           (CX2 iff_omega_easy)
Int.sub_one_lt_iff           (CX2 iff_omega_easy)
Int.le_antisymm_iff          (CX2 iff_omega_easy)
Int.le_iff_eq_or_lt          (CX2 iff_omega_easy)
Int.natCast_nonpos_iff       (CX2 iff_omega_medium)
Int.natCast_ne_zero_iff_pos  (CX2 iff_omega_medium)
Int.lt_toNat                 (CX2 order_arith; iff form caught here)
Int.natCast_eq_zero          (CX2 mixed)
```

### `fallback_omega / Int` (13 unique, gate ≥5 ✓)

All emit bare `omega`:

```
Int.emod_two_eq_zero_or_one  (CX1)
Int.le_of_eq                 (CX2 order_arith)
Int.natAbs_coe_sub_coe_lt_of_lt
Int.le_or_lt
Int.natAbs_coe_sub_coe_le_of_le
Int.zero_le_ofNat
Int.lt_or_lt_of_ne
Int.natAbs_add_of_nonpos
Int.lt_asymm
Int.le_natCast_sub
Int.neg_emod_two
Int.lt_or_le                 (CX2 order_arith)
Int.natCast_pred_of_pos      (CX2 mixed)
```

Output: `project/data/cx2_int_iff_omega_pool_meta.json`.

## 7. Negative control (Stage 6)

CX2 mining used the NS15 router unchanged. Neither the router config,
the NS9 best genome, nor any model checkpoint was modified. All
NS21-validated preservation properties carry over by construction:

- nat_defs_medium: 23/38 routed-raw (NS15 nat_oversample)
- nat_defs_large_v5: 35/65 routed-raw
- demo_v1: 10/15 routed-raw
- ns17_set_extra: 18/30
- ns17_finset_extra: 12/30 (NS15 router) / 15/30 (NS21 router)
- ns14_set_finset_extra: 13/20

The wrapper baselines (NS9 wrap + NS15 router = 37/38 medium, 49/65
large, 11/15 demo) likewise remain — those evals are read-only inputs
to the pool extraction. No re-eval was needed.

## 8. NS22 verdict and recipe (Stage 8)

**Training JUSTIFIED for both Int wrapper-only families.** Two
distinct training targets are available:

### Recommended NS22-A: train `gen_v5_ns22_int_iff_omega`

- **Pool:** 10 unique iff_omega_pair / Int theorems.
- **Tactic:** all emit `exact ⟨fun h => by omega, fun h => by omega⟩`.
- **Oversample factor:** 5× → 50 rows + replay.
- **Init from:** `gen_v5_ns12_balanced` (preserves Set/Finset/demo).
- **Replay:** NS12 balanced full + (optionally) NS15 nat_oversample
  iff_omega Nat rows for cross-family priming.
- **Router update:** add `^Int\.` → `gen_v5_ns22_int_iff_omega` to
  the NS21 router (keep Finset on NS21, Nat on NS15, Set/default on
  NS12).
- **Expected outcome:** model emits the iff-omega pair pattern
  natively on Int goals. Per NS21's transfer-ceiling memory, broad
  transfer is *more* likely here because the base model (NS12) has
  essentially no Int training data — the held-out raw on Int surfaces
  was 1-4/50 vs. baseline NS12's ~28/100 on Finset, indicating Int
  is genuinely a fresh namespace.

### Optional NS22-B: train `gen_v5_ns22_int_omega_combined`

- **Pool:** 23 unique theorems = 10 iff_omega_pair + 13 fallback_omega.
- **Tactic:** mixed — the model learns to emit either pattern
  depending on whether the goal is an `↔` or a closed form.
- **Oversample factor:** 2× (pool > 12 entries) → 46 rows + replay.
- **Init from:** `gen_v5_ns12_balanced`.
- **Pros:** larger pool, broader Int competence.
- **Cons:** mixed-tactic objective is harder to memorize cleanly than
  homogeneous; risk of mode collapse on the more common `omega` (12
  rows) at the expense of the `iff_omega_pair` form (10 rows).

**Recommendation: NS22-A first** (homogeneous, lower-risk), then
evaluate transfer to held-out Int. If transfer is strong, NS22-B
becomes redundant; if narrow, train NS22-B as a follow-up.

## 9. Decision-gate summary

| family | unique | gate | recommended action |
|---|---:|:---:|---|
| iff_omega_pair / Int | 10 | ✓ | **NS22-A training** |
| fallback_omega / Int | 13 | ✓ | NS22-B (optional follow-up) |

Both gates met decisively (2× and 2.6× the 5-win threshold). No
additional Int catalog mining needed before NS22.

## 10. Files

Scripts (committed):

- `scripts/cx2_int_catalog_audit.py`
- `scripts/build_cx2_theorem_sets.py`
- `scripts/cx2_run_eval.sh`
- `scripts/cx2_extract_pool.py`

Configs (committed):

- `project/evolve/routing/cx2_theorem_sets.json`

Catalog metadata (committed):

- `project/data/cx2_int_catalog_audit_meta.json`
- `project/data/cx2_theorem_sets_meta.json`
- `project/data/cx2_int_iff_omega_pool_meta.json`

Reports (committed):

- `project/evolve/reports/cx2_int_catalog_audit.md`
- `project/evolve/reports/cx2_pool_summary.md`
- `project/evolve/reports/cx2_int_iff_omega_mining_report.md` (this file)

`tasks.py` patched with `_load_cx2_sets()`.

Not committed: eval traces, raw logs, model checkpoints. `.gitignore`
extended with CX2 paths.
