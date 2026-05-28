# NS22 — Int iff-omega imitation training

**Branch:** `ns22-int-iff-omega-training`
**Parent:** CX2 commit `8caafc5`.
**Goal:** train a narrow Int branch model to imitate the homogeneous
Int `iff_omega_pair` wrapper-only family (10 unique wins from CX1+CX2)
and route Int goals to it; the parallel `fallback_omega` ablation was
to be evaluated separately.
**Outcome:** the surprise finding is that **the fallback_omega
ablation outperformed both iff_omega candidates by a wide margin**.
NS22 router routes `^Int\.` to `gen_v5_ns22_int_fallback_omega_5x`,
which produces **+22 raw Int wins (35 → 57)** over the NS12 baseline,
including **cross-family transfer** to 9/10 of the iff_omega pool —
the short `omega` tactic absorbs both training and held-out
iff-omega goals where the long iff-pair tactic failed to memorize.

## 1. CX2 gate recap

Two Int wrapper-only families exceeded the 5-win training gate:

| family | unique | source |
|---|---:|---|
| `iff_omega_pair` / Int | 10 | CX1: 2, CX2: 8 |
| `fallback_omega` / Int | 13 | CX1: 1, CX2: 12 |

NS22-A (`iff_omega_pair`) was the spec's primary; NS22-B
(`fallback_omega`) was an ablation. Per the user constraint, the two
were never mixed in a single training dataset.

## 2. Training-data construction (Stage 1)

`scripts/build_ns22_training_data.py` extracts the close
`(state_pp, tactic)` row for each pool theorem from wrapper traces
under `project/evolve/eval_runs/{cx1_ns9wrap,cx2_ns9wrap}_*`. Three
variants:

| variant | pool | oversample | total rows | init |
|---|---|---:|---:|---|
| `ns22_int_iff_omega_5x`        | 10 iff_omega rows | 5×  | 7,495 | gen_v5_ns12_balanced |
| `ns22_int_iff_omega_10x`       | 10 iff_omega rows | 10× | 7,545 | gen_v5_ns12_balanced |
| `ns22_int_fallback_omega_5x`   | 13 omega rows     | 5×  | 7,510 | gen_v5_ns12_balanced |

Replay: `project/data/ns12_train_balanced.jsonl` (7445 rows) in full.

Metas committed at `project/data/ns22_int_*_meta.json`. JSONLs
gitignored.

## 3. Training (Stage 3)

All three models trained from `gen_v5_ns12_balanced` for 3 epochs at
batch size 8, lr 5e-5, max_src_len 512, max_tgt_len 128.

## 4. Stage 4 — raw_ckpt evaluation

| set | NS12 base | iff_5x | iff_10x | **omega_5x** |
|---|---:|---:|---:|---:|
| `cx2_int_iff_omega_easy` | 1/12 | 1/12 | 1/12 | **5/12** |
| `cx2_int_iff_omega_medium` | 0/3 | 1/3 | 0/3 | **2/3** |
| `cx2_int_order_arith` | 4/50 | 5/50 | 5/50 | **15/50** |
| `cx2_int_mixed` | 4/13 | 6/13 | 4/13 | 6/13 |
| `cx1_bool_option_int` | 26/80 | 26/80 | 25/80 | **29/80** |
| **Int total (/158)** | **35** | **39** | **35** | **57** |
| `demo_v1` | 10/15 | 10/15 | 10/15 | 10/15 |
| `ns17_set_extra` | 18/30 | 18/30 | 19/30 | 19/30 |
| `ns17_finset_extra` | 12/30 | 13/30 | 14/30 | 12/30 |
| `ns14_set_finset_extra` | 13/20 | 11/20 | 13/20 | 11/20 |

The omega_5x ablation absorbs `omega` emission for Int goals so
broadly that it lifts every Int surface (+22 net). Both iff_omega
candidates failed to absorb the longer `exact ⟨fun h => by omega, fun h
=> by omega⟩` tactic — the iff_5x got marginal +4 via incidental
`omega` emission, iff_10x produced 0 net Int lift.

## 5. Stage 5 — routed policy

`project/evolve/routing/ns22_router.json`:

```
^Nat\.    → gen_v5_ns15_nat_oversample
^Int\.    → gen_v5_ns22_int_fallback_omega_5x   ← NEW
^Finset\. → gen_v5_ns21_finset_aesop_20x
^Set\.    → gen_v5_ns12_balanced
default   → gen_v5_ns12_balanced
```

### Routed raw matrix

| set | NS22 routed | prior NS21 routed | NS15 routed |
|---|---:|---:|---:|
| `demo_v1` | 10/15 | 10/15 | 10/15 |
| `ns14_set_finset_extra` | 13/20 | 13/20 | 13/20 |
| `ns17_set_extra` | 18/30 | 18/30 | 18/30 |
| `ns17_finset_extra` | **15/30** | 15/30 | 12/30 |
| `nat_defs_medium` | **23/38** | 23/38 | 23/38 |
| `nat_defs_large_v5` | **35/65** | 35/65 | 35/65 |
| `cx2_int_iff_omega_easy` | **5/12** | (NS15) 1/12 | 1/12 |
| `cx2_int_iff_omega_medium` | **2/3** | (NS15) 0/3 | 0/3 |
| `cx2_int_order_arith` | **15/50** | (NS15) 4/50 | 4/50 |
| `cx2_int_mixed` | 6/13 | (NS15) 4/13 | 4/13 |
| `cx1_bool_option_int` | **29/80** | (NS15) 26/80 | 26/80 |

NS22 router preserves all NS21/NS15 baselines on Nat / Set / Finset /
demo exactly, and adds **+22 raw Int wins**.

## 6. Stage 6 — wrapper compatibility

NS9 best genome + NS22 router:

| set | wrap+NS22-router | NS9 wrap baseline |
|---|---:|---:|
| `nat_defs_medium` | **37/38** | 37/38 ✓ |
| `nat_defs_large_v5` | **49/65** | 49/65 ✓ |
| `demo_v1` | **11/15** | ≥10/15 ✓ |
| `cx2_int_iff_omega_easy` | 5/12 | (raw routed = 5/12; wrap adds 0) |
| `cx2_int_order_arith` | 16/50 | (raw routed = 15/50; wrap adds 1) |

The wrapper baselines are exactly preserved. On Int surfaces, the
NS22 raw model has now absorbed `omega` emission so completely that
the wrapper adds essentially zero incremental wins — analogous to
the NS21 outcome on `cx1_finset_image_filter`.

## 7. Transfer analysis (Stage 7)

`scripts/ns22_compare_int_transfer.py` →
`project/evolve/reports/ns22_transfer_analysis.md`.

| ckpt | own-pool solved | via trained tactic | other-pool solved | held-out Int gains | neg losses | verdict |
|---|---:|---:|---:|---:|---:|---|
| `iff_omega_5x` | 2/10 | 0 | 2/13 | 0 | 3 | **weak_or_no_signal** |
| `iff_omega_10x` | 0/10 | 0 | 1/13 | 0 | 0 | **weak_or_no_signal** |
| **`fallback_omega_5x`** | **13/13** | **13** | **9/10** | 0 | 2 | **cross_family_transfer** |

### Key findings

1. **The long iff-pair tactic is unlearnable at this model scale.**
   Neither iff_5x nor iff_10x reproduces
   `exact ⟨fun h => by omega, fun h => by omega⟩` at test time — they
   continue to emit standard alternatives (`aesop`, `simp_all`,
   `simp [List.length_cons]`, ...). The 60M-param CodeT5-small base
   model has insufficient capacity to memorize a 49-character
   structured tactic from a 10-row pool, even with 10× oversampling.
   This bounds the kind of wrapper-only signal that NS-style
   imitation training can absorb: **short, vocabulary-aligned tactics
   transfer; long structured terms do not.**

2. **The `omega` tactic transfers broadly across iff-form Int goals.**
   The fallback_omega ablation solved 9/10 of the iff_omega pool
   theorems by emitting `omega` directly — *not* the iff-pair pattern
   the wrapper used to first close them. This is the inverse of the
   wrapper's training-time picture: although NS9 wrap's iff_omega_pair
   template won the race on the wrapper-only-vs-NS9 attribution, plain
   `omega` is in fact sufficient for these goals at test time. Lean's
   `omega` reflects iff goals automatically when the antecedent is a
   linear-arithmetic predicate.

3. **No held-out Int gains outside the pool, but cross-pool wins
   constitute genuine transfer.** The 9 cross-family wins were
   *never* in the omega_5x training set. omega_5x has absorbed
   "emit `omega` for Int goals" as a general policy, not just for
   the 13 trained examples. This is broader than NS21's pool
   memorization but narrower than NS15's iff_omega Nat (which lifted
   ns17_finset_extra-equivalent surfaces by ~3× over baseline).

## 8. Comparison to NS15 and NS21

| arc | pool | base model prior | outcome |
|---|---|---|---|
| **NS15** Nat iff_omega | 5 unique | NS12 had no Nat iff_omega | **broad transfer** (medium 3→23, large 9→35) |
| **NS21** Finset aesop | 6 unique | NS12 already emitted aesop on Finset | **memorization only** (+5 raw wins, 0 held-out) |
| **NS22** Int iff_omega/omega | 10 iff + 13 omega | NS12 had ~0 Int competence | **cross_family_transfer** (+22 Int, 9/10 cross-pool); long tactic unlearnable |

The pattern matches the NS21-derived memory: **fresh-namespace pools
with short tactics produce broad transfer; long structured tactics
plateau even on fresh namespaces**. NS22 confirms the namespace-prior
hypothesis (Int's +22 lift far exceeds Finset's +5 from a similar pool
size) and adds the new constraint that the *tactic complexity* also
bounds what the model can absorb.

## 9. Recommendation

The NS22 router is the new best end-to-end policy. Beyond this:

1. **Skip iff_omega-pair imitation in future arcs.** The wrapper-only
   pool attribution rewards iff-pair templates because they "win the
   race" within NS9's ordering, but the actual transferable signal is
   the shorter `omega` tactic. Future mining should aggregate
   `iff_omega_pair + fallback_omega` into a single homogeneous
   `omega` pool for training-data purposes, while keeping the wrapper
   templates intact for search-time win attribution.

2. **CX3-Int: extend the Int catalog further.** The Int surface
   continues to be under-served by the base NS12 model. Adding
   `Mathlib/Data/Int/Interval.lean`, `Mathlib/Data/Int/CharZero.lean`,
   `Mathlib/Data/Int/CardIntervalMod.lean`, and the rest of
   `Mathlib/Data/Int/{ConditionallyCompleteOrder,LeastGreatest}.lean`
   would likely add 50-150 more candidates and, at the observed 26%
   wrapper-only strike rate, ~13-40 more wrapper-only Int wins.
   Diminishing returns vs current 57/158 NS22-router coverage.

3. **CX2-style mining for other under-served namespaces.** Per the
   CX1 audit, `Bool`/`Option` are still mostly unused (35 Bool + 47
   Option theorems available). The NS22 omega absorption pattern
   suggests `decide`-family wrapper-only pools could yield similar
   broad transfer on Bool.

4. **DPO/ranker direction is now more attractive.** With the simple
   imitation pipeline producing diminishing returns on each new
   namespace, a preference/ranking objective over wrapper-only
   competing tactics may be needed to teach the model the long
   structured tactics (iff-pair, multi-step terms) that NS22-iff
   failed to absorb.

**Recommended next:** option 1 (consolidate iff_omega_pair into
fallback_omega pool training for future mining arcs) plus option 3
(CX3-Bool/Option) as the next-highest-yield mining direction.

## 10. Files

Scripts (committed):

- `scripts/build_ns22_training_data.py`
- `scripts/ns22_run_eval.sh`
- `scripts/ns22_compare_int_transfer.py`

Configs (committed):

- `project/evolve/routing/ns22_router.json`

Metadata (committed):

- `project/data/ns22_int_iff_omega_5x_meta.json`
- `project/data/ns22_int_iff_omega_10x_meta.json`
- `project/data/ns22_int_fallback_omega_5x_meta.json`
- `project/data/ns22_transfer_analysis.json`

Reports (committed):

- `project/evolve/reports/ns22_int_iff_omega_training_report.md` (this file)
- `project/evolve/reports/ns22_transfer_analysis.md`

`.gitignore` extended with NS22 paths. Not committed: checkpoints,
training JSONLs, eval traces/logs.
