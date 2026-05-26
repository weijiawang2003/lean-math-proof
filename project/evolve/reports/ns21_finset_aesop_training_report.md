# NS21 — Finset/aesop imitation training

**Branch:** `ns21-finset-aesop-training`
**Parent:** CX1 catalog-extension commit `3ee50fd`.
**Goal:** train a narrow Finset model to imitate the homogeneous
aesop/Finset wrapper-only pool that CX1 elevated past the 5-unique-win
training gate, then route Finset goals to it while preserving Nat (via
NS15) and Set/default (via NS12).
**Outcome:** **the 20x candidate is the chosen router target.** It
absorbs the 6-theorem aesop pool natively (5/6 raw-solvable, all via
`aesop`), preserves all wrapper baselines (37/38 medium, 49/65 large,
11/15 demo) and Nat routing (23/38 medium, 35/65 large), and adds no
Set/demo regression at the router level. Held-out transfer is
**memorization, not generalization**: the held-out Finset surfaces had
already absorbed `aesop` emission via NS12 balanced training, so the
NS21 contribution is bounded to the 5 pool theorems it can now emit
without the wrapper.

## 1. CX1 gate recap

The Finset/aesop pool (per `cx1_combined_pool_meta.json`):

| arc | wins | theorems |
|---|---:|---|
| NS18 | 3 | `Finset.coe_insert`, `Finset.cons_eq_insert`, `Finset.disjUnion_singleton` |
| NS19 | 1 | `Finset.coe_cons` |
| CX1 | 2 | `Finset.card_insert_eq_ite`, `Finset.image_id` |
| **total** | **6 unique** | all winning_tactic = `aesop` |

Gate requirement: ≥5 unique same-family same-namespace wrapper-only
wins. Pool is at 6.

## 2. Training-data construction (Stage 1)

`scripts/build_ns21_training_data.py` extracts the close-row
`(state_pp, "aesop")` pair from each pool theorem's wrapper trace
under `project/evolve/eval_runs/{ns18_aesop_wrapper, ns19_finset_aesop_only,
cx1_finset_aesop_only}_wrapper_*`. All 6 are confirmed to use
`tactic_origin == "tactic_template"` and `tactic == "aesop"`.

Hard exclusions from training:

- Nat simp_all wrapper-only rows (sub-gate, 3 unique only)
- Int iff_omega_pair / fallback_omega rows (sub-gate)
- heterogeneous wrapper-only rows

Replay corpus: `project/data/ns12_train_balanced.jsonl` (7445 rows;
~50% Set+Finset by NS12 design).

Three variants:

| variant | pool rows | replay rows | total | init |
|---|---:|---:|---:|---|
| `ns21_finset_aesop_10x` | 60 (10×6) | 7445 (full NS12) | 7505 | gen_v5_ns12_balanced |
| `ns21_finset_aesop_20x` | 120 (20×6) | 7445 (full NS12) | 7565 | gen_v5_ns12_balanced |
| `ns21_finset_aesop_minimal` | 120 (20×6) | 500 (random) | 620 | gen_v5_ns12_balanced |

Meta committed at `project/data/ns21_finset_aesop_{10x,20x,minimal}_meta.json`.
Full JSONLs gitignored.

## 3. Training (Stage 3)

| ckpt | corpus | epochs | steps | init from |
|---|---|---:|---:|---|
| `gen_v5_ns21_finset_aesop_10x` | 10x | 3 | 2535 | gen_v5_ns12_balanced |
| `gen_v5_ns21_finset_aesop_20x` | 20x | 3 | 2556 | gen_v5_ns12_balanced |
| `gen_v5_ns21_finset_aesop_minimal` | minimal | 5 | 350 | gen_v5_ns12_balanced |

All start from NS12 balanced rather than gen_v5 so the demo/Set/Finset
baseline behavior is preserved on initialization. Logs and ckpt dirs
are gitignored.

## 4. Stage 4 — raw_ckpt evaluation

| set | NS12 base | 10x | 20x | minimal |
|---|---:|---:|---:|---:|
| `demo_v1` | 10/15 | 10/15 | **11/15** | 10/15 |
| `ns14_set_finset_extra` | 13/20 | 11/20 | 12/20 | 10/20 |
| `ns17_set_extra` | 18/30 | 19/30 | 18/30 | 18/30 |
| `ns17_finset_extra` | 12/30 | **15/30** | **15/30** | **15/30** |
| `cx1_finset_image_filter` | 28/100 | **30/100** | **30/100** | **30/100** |
| `ns20_finset_aesop_extra_medium` | 7/16 | 7/16 | 7/16 | 7/16 |

Every NS21 candidate lifts:

- `ns17_finset_extra`: +3 (12 → 15) — the 3 NS18 pool theorems.
- `cx1_finset_image_filter`: +2 (28 → 30) — the 2 CX1 pool theorems.

`Finset.coe_cons` (NS19 pool theorem) was not solved raw by any
candidate; the wrapper had emitted aesop after several priority-template
attempts and only the assist context made aesop work in that trace, so
the raw model is missing the lead-in context — a known limitation of
single-step imitation.

`ns20_finset_aesop_extra_medium` is unmoved: that set's 7 baseline wins
are the surface aesop already cleared via NS12; the pool theorems
aren't in it, so NS21 has nothing to add.

## 5. Router selection (Stage 5)

`scripts/ns21_pick_router.py` scores each candidate as
`finset_wins − 0.5 × set_reg − 0.5 × demo_reg − 0.5 × nat_reg`. Ties
broken by smaller oversample factor.

| ckpt | finset wins | set reg | demo reg | score |
|---|---:|---:|---:|---:|
| 10x | 52 | 2 | 0 | 51.0 |
| **20x** | **52** | **1** | **0** | **51.5** |
| minimal | 52 | 3 | 0 | 50.5 |

**Chosen: `gen_v5_ns21_finset_aesop_20x`.** It alone preserves demo at
11/15 (above the 10/15 floor) and keeps the lone Set regression to 1
theorem on `ns14_set_finset_extra` (12 vs 13 baseline). Router config
at `project/evolve/routing/ns21_router.json` routes:

- `^Nat\.` → `gen_v5_ns15_nat_oversample`
- `^Finset\.` → `gen_v5_ns21_finset_aesop_20x`  ← new
- `^Set\.` → `gen_v5_ns12_balanced`
- default → `gen_v5_ns12_balanced`

### Routed raw matrix (NS21 router)

| set | NS21-routed raw | NS15-routed raw (prior) |
|---|---:|---:|
| `demo_v1` | 10/15 | 10/15 |
| `ns14_set_finset_extra` | **13/20** | 13/20 |
| `ns17_set_extra` | 18/30 | 18/30 |
| `ns17_finset_extra` | **15/30** | 12/30 |
| `nat_defs_medium` | **23/38** | 23/38 |
| `nat_defs_large_v5` | **35/65** | 35/65 |
| `cx1_finset_image_filter` | **30/100** | 28/100 |
| `ns20_finset_aesop_extra_medium` | 7/16 | 7/16 |

The router preserves every Nat / Set / demo baseline exactly because
those namespaces still hit NS15 / NS12. The Finset improvements (+3
ns17, +2 cx1) carry through directly.

## 6. Wrapper compatibility (Stage 6)

NS9 best genome wrapper + NS21 router on the wrapper-floor suites:

| set | NS21 wrap+router | NS9 wrap floor |
|---|---:|---:|
| `nat_defs_medium` | **37/38** | 37/38 ✓ |
| `nat_defs_large_v5` | **49/65** | 49/65 ✓ |
| `demo_v1` | **11/15** | ~10/15 (above floor) ✓ |
| `cx1_finset_image_filter` | **30/100** | n/a (new at CX1) |

The wrapper baselines are preserved exactly. Notably, on
`cx1_finset_image_filter` the wrapper now contributes **zero**
incremental wins over the routed raw NS21 — the model has absorbed
`aesop` emission for the gains the wrapper was producing.

## 7. Transfer-vs-memorization analysis (Stage 7)

`scripts/ns21_compare_finset_transfer.py` →
`project/data/ns21_transfer_analysis.json` +
`project/evolve/reports/ns21_transfer_analysis.md`.

### Pool — 5/6 raw-solvable by every candidate

| theorem | 10x | 20x | min | tactic emitted |
|---|:---:|:---:|:---:|---|
| `Finset.coe_insert` | ✓ | ✓ | ✓ | `aesop` |
| `Finset.cons_eq_insert` | ✓ | ✓ | ✓ | `aesop` |
| `Finset.disjUnion_singleton` | ✓ | ✓ | ✓ | `aesop` |
| `Finset.coe_cons` | — | — | — | (state requires lead-in tactic) |
| `Finset.card_insert_eq_ite` | ✓ | ✓ | ✓ | `aesop` |
| `Finset.image_id` | ✓ | ✓ | ✓ | `aesop` |

### Held-out Finset transfer = 0

Per-set comparison vs `gen_v5_ns12_balanced` on non-pool theorems:

| set | held-out thms | NS12 wins | NS21 wins | gains | losses |
|---|---:|---:|---:|---:|---:|
| `ns17_finset_extra` | 27 | 12 | 12 | 0 | 0 |
| `cx1_finset_image_filter` | 98 | 28 | 28 | 0 | 0 |
| `ns20_finset_aesop_extra_medium` | 16 | 7 | 7 | 0 | 0 |

Tactic breakdown on `cx1_finset_image_filter` held-out wins: NS12
emits `aesop` for 27/28 of its wins; NS21 emits `aesop` for 29/30 of
its wins (the 2 extra are the pool theorems). **NS12 was already
emitting aesop on Finset goals where aesop suffices** — the wrapper's
`aesop`-template contribution was confined to the few goals NS12
hadn't memorized, and NS21 has now memorized exactly those.

### Verdict: pool memorization, no broad transfer

The classifier (`pool_solved=5/6, held_out_gains=0`) marks all three
candidates as **memorization**. This is the *expected* honest outcome
given that:

1. The wrapper-only-vs-NS9 pool is, by construction, the set of
   theorems the routed raw NS15 model fails on. The training
   converts wrapper-driven wins on these exact theorems into native
   raw wins.
2. NS12 balanced already trained on the ample aesop-on-Finset surface
   (3752 Finset rows), so any Finset goal aesop would have closed was
   already being closed natively. The training pool's "new pattern"
   was therefore narrow: 6 specific proof-state shapes where aesop
   was applicable but the model wasn't emitting it.
3. With only 5 of 6 pool shapes successfully memorized and no
   broader pattern in the data, the model cannot generalize beyond
   them.

## 8. Limitations & implications

- **Marginal absolute gain is small.** +5 raw wins (3 + 2) on the
  evaluated Finset surfaces, with a slight Set/Finset trade-off on
  `ns14_set_finset_extra` (-1) — net effective lift on the router
  is +4 wins across the evaluated suite.
- **The training-gate threshold is not the binding constraint.** The
  pool size (6) being just over the gate means the imitation surface
  is narrow. Future pools that approach or exceed ~15-20 unique
  wins would likely produce broader transfer; the present pool is
  large enough to train safely but small enough that the model
  cannot induce a more general rule.
- **The wrapper has been fully absorbed for these 6 theorems** — the
  Finset surface in `cx1_finset_image_filter` shows wrap+router and
  raw_routed at the same 30/100. This validates the AlphaEvolve
  closing-the-loop story end-to-end: search produced wrapper,
  wrapper produced trace data, trace data improved raw model,
  routing preserved domains.
- **Set namespace must stay on NS12.** NS21 raw on Set surfaces
  regresses by 1-3 theorems; routing Finset to NS21 while leaving
  Set on NS12 avoids leakage cleanly.
- **`Finset.coe_cons` resists single-step imitation.** Its wrapper
  trace required priority-template setup before the closing aesop;
  the model only saw the close row in training. Future pools should
  consider mining multi-step traces with the existing `advance_assist`
  role from NS16.

## 9. Recommendation

Pick exactly one of:

1. **More Finset/aesop mining (CX2-Finset).** The CX1 catalog still
   has Finset surfaces unprobed (`ns20_finset_aesop_extra_easy/hard`
   were excluded from this matrix to save eval cost; ~58 theorems).
   At current strike rate, mining the rest of the cx1_finset_image_filter
   neighborhood with NS9 wrap could add 2-4 more aesop wrapper-only
   theorems. With ~10 unique pool theorems, a re-train would have
   substantially better transfer odds.
2. **CX2-Int (Int/iff_omega).** The CX1 secondary finding —
   `iff_omega_pair / Int` at 2 wrapper-only wins — is the
   highest-yield direction. The iff-omega pattern generalized
   directly Nat → Int with no model retraining, and Int's
   `Mathlib/Data/Int/Order.lean` + remaining defs are unprobed (CX1
   sampled only 80 Int theorems). 3 more Int iff_omega wins would
   meet the gate. **This is the most attractive next mining target.**
3. **Stronger aesop wrapper.** If the goal is to recover
   `Finset.coe_cons` and similar multi-step pool theorems, extend
   the wrapper with a `set_intro_then_aesop` template (intro the
   coercion lemma then call aesop) and re-run NS20-style mining.

**Recommended next:** option 2 (CX2-Int). It has the strongest
expected-yield-per-eval ratio, opens a brand-new (family,
namespace) pool, and the iff-omega Nat success in NS15 confirms
the pattern is trainable.

## 10. Files

Scripts (committed):

- `scripts/build_ns21_training_data.py`
- `scripts/ns21_run_eval.sh`
- `scripts/ns21_run_matrix.sh`
- `scripts/ns21_pick_router.py`
- `scripts/ns21_compare_finset_transfer.py`

Configs (committed):

- `project/evolve/routing/ns21_eval_sets.json`
- `project/evolve/routing/ns21_router.json`

Metadata (committed):

- `project/data/ns21_finset_aesop_10x_meta.json`
- `project/data/ns21_finset_aesop_20x_meta.json`
- `project/data/ns21_finset_aesop_minimal_meta.json`
- `project/data/ns21_transfer_analysis.json`

Reports (committed):

- `project/evolve/reports/ns21_finset_aesop_training_report.md` (this file)
- `project/evolve/reports/ns21_transfer_analysis.md`

`.gitignore` extended with NS21 paths. Not committed: model checkpoints,
training JSONLs, eval traces/logs.
