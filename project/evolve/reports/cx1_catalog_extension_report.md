# CX1 — Mathlib catalog extension

**Branch:** `cx1-catalog-extension`
**Parent:** consolidation commit `acf1fdf` (Learn-track NS10–NS20).
**Goal:** expand the theorem catalog substantially so future wrapper
mining can find new dense homogeneous proof families.
**Outcome:** **catalog extended 3.5×** (527 → 1817 available
theorems across 8 new/expanded namespaces). The limited eval probe
produced **6 new wrapper-only-vs-NS9 wins across 4 (family,
namespace) buckets**, and — crucially — **the aesop/Finset
training gate is now met at 6 unique wins** (≥5 required for NS21).

## 1. Background

The Learn-track final report concluded that further training was
blocked by three structural constraints. The first — **catalog
exhaustion** — is what CX1 directly addresses. Pre-CX1 catalog had
527 theorems drawn from exactly 3 Mathlib source files:

```
Mathlib/Data/Nat/Defs.lean        208 theorems  (FULLY USED)
Mathlib/Data/Set/Basic.lean        91 theorems  (~50% used)
Mathlib/Data/Finset/Basic.lean    228 theorems  (FULLY USED)
```

The CX1 audit (`cx1_catalog_audit.md`) confirmed Nat / Finset /
List fully exhausted; Bool / Option / Int entirely absent.

## 2. Catalog discovery (Stage 2)

`scripts/cx1_discover_theorems.py` scans 44 additional Mathlib
source files via regex extraction (no LeanDojo invocation). Notable
correctness work during the scan: `section X` was initially being
treated like `namespace X` (which inflated full_names with section
prefixes), and `_root_.X` declarations were being wrongly prepended
with the enclosing namespace. Both bugs were caught by the Stage 3
availability probe and fixed.

**Result:** 3,989 theorem declarations extracted, of which **1,843
have tactic-style proofs**. Per-namespace fresh-with-tactic-proof
distribution:

| namespace | fresh count |
|---|---:|
| Set | 871 |
| Finset | 710 |
| Multiset | 573 |
| List | 541 |
| Nat | 388 |
| Int | 227 (entirely new) |
| Option | 113 (entirely new) |
| Bool | 40 (entirely new) |

Output: `project/discovered_theorems_cx1.json`.

## 3. Availability check (Stage 3)

`scripts/cx1_check_theorem_availability.py` probes 44 source files
× 3 sample theorems = 132 LeanDojo entries (90s timeout, ~5s actual
average). Files with ≥1 successful sample → PRESUMED AVAILABLE.

**Result:** 43/44 files passed. Only `Mathlib/Data/Option/NAry.lean`
came back unavailable. **1,817 theorems usable** at the existing
Dojo commit `29dcec074de168ac2bf835a77ef68bbe069194c5`.

| namespace | available |
|---|---:|
| Finset | 334 |
| Set | 325 |
| List | 276 |
| Nat | 261 |
| Multiset | 260 |
| Int | 120 |
| Option | 47 |
| Bool | 35 |

Output: `project/data/cx1_available_theorems.json`,
`project/evolve/reports/cx1_availability_report.md`.

## 4. Theorem-set construction (Stage 4)

`scripts/build_cx1_theorem_sets.py` partitions the available pool
into six CX1 sets, excluding everything already in prior theorem
sets:

| set | size | composition |
|---|---:|---|
| `cx1_finset_image_filter` | 100 | Finset image/filter/map/card/erase/attach |
| `cx1_nat_gcd_dvd_mod` | 100 | Nat gcd/dvd/mod/div + Nat.ModEq sub-namespace |
| `cx1_list_multiset` | 100 | List 87, List.Chain' 7, Multiset 6 |
| `cx1_bool_option_int` | 80 | Int 43, Bool 35, Option 2 |
| `cx1_mixed_easy` | 37 | stratified easy across namespaces |
| `cx1_mixed_medium` | 50 | stratified medium across namespaces |

Output: `project/evolve/routing/cx1_theorem_sets.json`; loaded via
`tasks.py:_load_cx1_sets()`.

## 5. Eval probe (Stage 5)

Limited per CX1 spec — covered four sets most likely to surface
new signal. Driver: `scripts/cx1_run_matrix.sh`. Each cell uses
`--top-k 8 --max-steps 8` and the NS15 router.

| set | raw | ns9wrap | variant (`finset_aesop_only`) | Δwrap |
|---|---:|---:|---:|---:|
| `cx1_finset_image_filter` | 28/100 | 28/100 | **30/100** | **+2** |
| `cx1_nat_gcd_dvd_mod` | 2/100 | **3/100** | _n/a_ | **+1** |
| `cx1_bool_option_int` | 26/80 | **29/80** | _n/a_ | **+3** |
| `cx1_list_multiset` | 14/100 | 14/100 | _n/a_ | 0 |

Two stratified mixed_* sets deferred (covered ground the four
target sets touch).

## 6. Wrapper-only signal (Stage 6)

Per-(family, namespace) breakdown of the 6 CX1 truly-new wins:

| family | namespace | unique wins | theorems |
|---|---|---:|---|
| aesop | Finset | 2 | `Finset.card_insert_eq_ite`, `Finset.image_id` |
| iff_omega_pair | Int | 2 | `Int.le_add_one_iff`, `Int.le_iff_lt_or_eq` |
| simp_all | Nat | 1 | `Nat.add_mod_of_add_mod_lt` |
| fallback_omega | Int | 1 | `Int.emod_two_eq_zero_or_one` |

Output: `project/data/cx1_wrapper_only_signal_meta.json`,
`project/evolve/reports/cx1_wrapper_only_signal_summary.md`.

### Combined wrapper-only pool (NS18 + NS19 + NS20 + CX1)

Per `scripts/cx1_combined_pool.py` →
`project/data/cx1_combined_pool_meta.json`:

| family | namespace | unique | trainable? | sources |
|---|---|---:|:---:|---|
| **aesop** | **Finset** | **6** | **✔ GATE MET** | NS18: 3, NS19: 1, CX1: 2 |
| simp_all | Nat | 3 | (need 5) | NS18: 2, CX1: 1 |
| iff_omega_pair | Int | 2 | (need 5) | CX1: 2 |
| fallback_omega | Int | 1 | (need 5) | CX1: 1 |

The 6 unique aesop/Finset wins:
- `Finset.coe_insert`, `Finset.cons_eq_insert`,
  `Finset.disjUnion_singleton` (NS18)
- `Finset.coe_cons` (NS19)
- `Finset.card_insert_eq_ite`, `Finset.image_id` (CX1)

All six share the structure "the wrapper emits `aesop` and aesop
closes the goal in one tactic step" — a homogeneous pool ready for
distillation.

## 7. NS21 verdict

**Training JUSTIFIED for the aesop/Finset family.** The pool meets
the 5-unique-win gate at 6 unique wins, all sharing winning_tactic
`aesop`, all in the Finset namespace. This is the first
gate-meeting pool produced since NS15's iff-omega-pair Nat family.

### Recommended NS21 design (informational — not executed in CX1)

1. **Training pairs.** For each of the 6 pool theorems, extract the
   `(initial_state_pp, "aesop")` pair from the corresponding
   wrapper trace under `project/evolve/eval_runs/`. With 10×
   oversampling (per the pool meta's recommended factor), this is
   60 rows + the NS11/NS14 mix-in.

2. **Routing.** Train `gen_v5_ns21_finset_aesop` and update the
   NS15 router so `^Finset\.` points to it. Set/Nat routes stay
   on NS12 balanced / NS15 nat_oversample respectively. The
   stateless router means rollback is trivial if the new model
   regresses.

3. **Anti-forgetting mix-in.** Include the NS12_balanced training
   data at full weight to prevent the NS16 `curriculum_continue`-
   style catastrophic forgetting on Set/Finset baselines. CX1's
   `cx1_bool_option_int` Int wins are NOT mixed in for NS21 —
   they belong to a separate (still sub-gate) iff_omega_pair pool.

4. **Negative-control eval.** Post-train, re-run
   `nat_defs_medium`, `nat_defs_large_v5`, `demo_v1`,
   `ns17_finset_extra`, `ns17_set_extra`, `ns19_finset_aesop_surface`,
   and `cx1_finset_image_filter`. Required: ≥ NS15 raw + NS9 wrapper
   floor on every set; ≥ 6/100 raw wins on `cx1_finset_image_filter`
   (above the current 28/100 raw, since the model should have
   internalized aesop on Finset).

5. **Stop criterion.** Train until train-loss plateau or to a
   max of 20 epochs (whichever first). Pick the checkpoint
   maximizing  raw cx1_finset_image_filter wins − 0.5 × medium
   regressions.

## 8. Other CX1 findings worth noting (not yet trainable)

- **Int/iff_omega_pair (2 wins) is the highest-yield CX1
  discovery.** The iff-omega pattern from NS15 generalized
  directly from Nat to Int with no model retraining. If a
  follow-up CX2 surface — say, the rest of `Int/Defs.lean` or
  `Int/Order.lean` — surfaces 3 more iff_omega Int wins, that
  pool meets the gate. Most attractive next mining target.
- **`cx1_nat_gcd_dvd_mod` baseline is 2/100 raw** — the routed
  NS15 model has essentially zero coverage on this surface.
  Even modest training on the simp_all/Nat pool would lift
  recall here substantially.
- **`cx1_bool_option_int` baseline is 26/80 raw**, mostly from
  Bool theorems the routed default model handles fine. Bool/Int
  baseline coverage is better than feared.

## 9. Files

Scripts (committed):
- `scripts/cx1_catalog_audit.py`
- `scripts/cx1_discover_theorems.py`
- `scripts/cx1_check_theorem_availability.py`
- `scripts/build_cx1_theorem_sets.py`
- `scripts/cx1_run_eval.sh`
- `scripts/cx1_run_matrix.sh`
- `scripts/cx1_extract_signal.py`
- `scripts/cx1_combined_pool.py`

Configs / routing (committed):
- `project/evolve/routing/cx1_theorem_sets.json`

Catalog metadata (committed):
- `project/discovered_theorems_cx1.json` (3989 raw candidates)
- `project/data/cx1_available_theorems.json` (1817 presumed-available)
- `project/data/cx1_catalog_audit_meta.json`
- `project/data/cx1_wrapper_only_signal_meta.json`
- `project/data/cx1_combined_pool_meta.json`

Reports (committed):
- `project/evolve/reports/cx1_catalog_audit.md`
- `project/evolve/reports/cx1_availability_report.md`
- `project/evolve/reports/cx1_wrapper_only_signal_summary.md`
- `project/evolve/reports/cx1_catalog_extension_report.md` (this file)

`tasks.py` patched with `_load_cx1_sets()`.

Not committed: eval traces, raw logs, model checkpoints. The
availability probe log under
`project/evolve/eval_runs/cx1_availability_probe.log` is local-only.
`.gitignore` extended with CX1 paths.
