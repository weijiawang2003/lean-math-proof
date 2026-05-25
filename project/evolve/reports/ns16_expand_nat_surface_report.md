# NS16 — Expand wrapper-only Nat trace training

## Headline (honest)

**The NS15 transfer recipe did not generalize.** Oversampling 19
NS16 wrapper-only Nat rows at 10×/20×/curriculum-continue produced
**zero** new raw wins on the four new NS16 Nat sets, and zero new
wins on `nat_defs_medium` / `nat_defs_large_v5` beyond NS15.
Demo + NS14 transfer was retained. The NS9 wrapper still adds +1
on `demo_v1` and 14 wrapper-only wins on the larger Nat sets.

Net for the routed policy: **NS16 routed = NS15 routed exactly** on
every set we measure.

## Why we tried it

NS15 found that oversampling 10 NS14 wrapper-only Nat rows at 10×
gave 8/8 transfer of the iff-omega pattern. The hypothesis for
NS16 was that scaling this to a broader Nat surface would unlock
more wrapper-only wins. Step 1: enumerate more Nat theorems. Step
2: extract wrapper-only training rows from the wrapper traces.
Step 3: oversample and retrain.

## Stage 1 — expanded Nat theorem surface

[`scripts/build_ns16_theorem_sets.py`](../../../scripts/build_ns16_theorem_sets.py)
walks `project/discovered_theorems.json`, excludes everything in
any prior eval set, and emits four new Nat sets via naming
heuristics:

| set | size | difficulty | rationale |
|---|---:|---|---|
| `ns16_nat_iff_extra` | 17 | easy 14 / medium 3 | iff-shaped goals, wrapper template should fire |
| `ns16_nat_div_mod_extra` | 25 | easy 18 / medium 7 | div / mod / dvd: family tactics + omega after substitution |
| `ns16_nat_order_extra` | 16 | easy 13 / medium 3 | `_lt_` / `_le_`: omega / linarith reach |
| `ns16_nat_mixed_extra` | 28 | easy 21 / medium 7 | everything else |
| **TOTAL** | **86** | easy 66 / medium 20 | |

All 86 are new — no overlap with `nat_defs_medium`,
`nat_defs_large_v5`, `demo_v1`, or `ns14_nat_extra`. Registered in
`tasks.py` via a new `_load_ns16_sets()` loader.

## Stage 2 — raw vs wrapper baseline

Raw = NS15 routed (`gen_v5_ns15_nat_oversample` for Nat); wrapper =
NS9 best genome + NS15 routed:

| set | raw | wrapper | wrapper-only |
|---|---:|---:|---:|
| `ns16_nat_iff_extra` (17) | 1 | 2 | **+1** (`Nat.lt_find_iff`) |
| `ns16_nat_div_mod_extra` (25) | 0 | 1 | **+1** (`Nat.mul_add_mod_of_lt`) |
| `ns16_nat_order_extra` (16) | 3 | 3 | 0 |
| `ns16_nat_mixed_extra` (28) | 0 | 0 | 0 |
| **TOTAL** | **4/86** | **6/86** | **+2** |

The new wrapper-only signal on NS16 is tiny — only 2 wrapper-only
wins on 86 fresh theorems. This is the **first signal** that the
expanded surface yields diminishing wrapper-only returns: the easy
wrapper templates (iff-omega, plain omega) have already absorbed
the obvious wins; the remaining hard theorems are not closed by
the current wrapper genome.

For comparison, the existing `nat_defs_medium`/`nat_defs_large_v5`
runs (where wrapper still has a lot of room over raw NS15) show
14 wrapper-only theorems each:

| set | wrapper-only count |
|---|---:|
| `nat_defs_medium` | 14 |
| `nat_defs_large_v5` | 14 (same theorems — large ⊃ medium) |

So the wrapper-only Nat pool we *can* mine is 14 (medium/large
overlap) + 2 (NS16) = 16 theorems total.

## Stage 3 — wrapper-only training pair extraction

[`scripts/build_ns16_training_data.py`](../../../scripts/build_ns16_training_data.py)
walks the wrapper traces and extracts pairs only for theorems that
are wrapper-proved AND raw-unproved. Same filter pipeline as NS14
(`max_tactic_len=200`, `max_state_len=2500`, allow-listed origins,
no self-reference).

| metric | value |
|---|---:|
| trace files scanned | 6 |
| rows pre-dedup | 35 |
| rows post-dedup | **19** |
| unique theorems | 16 |
| close transitions | 16 |
| advance_assist transitions | 3 |
| NS11-held-out rows (training contamination) | 11 |

By origin:

| origin | rows |
|---|---:|
| `tactic_template` | 12 |
| `family_tactic` | 3 |
| `generative_topk` | 3 |
| `retrieved_premise` | 1 |

By source set:

| set | rows |
|---|---:|
| `nat_defs_medium` | 16 |
| `ns16_nat_iff_extra` | 2 |
| `ns16_nat_div_mod_extra` | 1 |

Note: `nat_defs_large_v5` produced 0 rows post-dedup because every
state hash matched the medium run's (large ⊃ medium).

The dataset is heavily contaminated with NS11-held-out theorems
(11 of 19 rows), which compromises any honest held-out evaluation
on those names. NS11 had set them aside precisely so we could
measure transfer to unseen theorems. NS14 already started eroding
this; NS16 finishes the job. The meta flags
`is_ns11_heldout=true` on every contaminated row so future stages
can choose whether to keep them.

## Stage 4 — training dataset variants

[`scripts/build_ns16_datasets.py`](../../../scripts/build_ns16_datasets.py)
emits three variants:

| variant | rows | init | NS15 base | NS16 copies | oversample factor |
|---|---:|---|---:|---:|---:|
| `oversample_10x` | 6,033 | `gen_v5` | 5,843 | 190 | 10× |
| `oversample_20x` | 6,223 | `gen_v5` | 5,843 | 380 | 20× |
| `curriculum_continue` | 380 | `gen_v5_ns15_nat_oversample` | 0 | 380 | 20× |

`curriculum_continue` is a short fine-tune (lr=5e-6, 3 epochs)
starting from the NS15 nat_oversample checkpoint — the
"continue-from-NS15" curriculum strategy.

## Stage 5 — training

Same hyperparameters NS15 used:

| variant | epochs | lr | output dir | wall time |
|---|---:|---|---|---|
| `oversample_10x` | 3 | 1e-5 | `gen_v5_ns16_oversample_10x` | ~12 min |
| `oversample_20x` | 3 | 1e-5 | `gen_v5_ns16_oversample_20x` | ~12 min |
| `curriculum_continue` | 3 | 5e-6 | `gen_v5_ns16_curriculum_continue` | ~1 min |

## Stage 6 — evaluation

### Raw eval (no wrapper, no router)

Each NS16 sub-model on all 8 sets:

| set | NS15 routed | NS16 10x | NS16 20x | NS16 curriculum |
|---|---:|---:|---:|---:|
| `nat_defs_medium` (38) | 23 | 23 | 22 | 17 |
| `nat_defs_large_v5` (65) | 35 | 35 | 34 | 26 |
| `demo_v1` (15) | 10 | 9 | 9 | 10 |
| `ns14_nat_extra` (20) | 9 | 9 | 8 | 3 |
| `ns14_set_finset_extra` (20) | 13 (n/a — Set/Finset) | — | — | — |
| `ns16_nat_iff_extra` (17) | 1 | 1 | 1 | 0 |
| `ns16_nat_div_mod_extra` (25) | 0 | 0 | 0 | 0 |
| `ns16_nat_order_extra` (16) | 3 | 3 | 3 | 3 |
| `ns16_nat_mixed_extra` (28) | 0 | 0 | 0 | 0 |

Findings:

- **`oversample_10x` matches NS15 nat_oversample exactly** on every
  set (the 190 extra rows didn't move the model meaningfully).
- **`oversample_20x` slightly regresses** on medium / large / NS14
  by 1 theorem each — the heavier oversample mildly distorts the
  Nat distribution.
- **`curriculum_continue` catastrophically forgets**: -6 on
  medium, -9 on large, -6 on NS14. Starting from
  `gen_v5_ns15_nat_oversample` and fine-tuning on just the 380-row
  wrapper-only corpus (lr=5e-6 even) shifts the output distribution
  enough to lose most prior wins. Demo (Set) is preserved at
  10/15 because the demo replay rows are not in the corpus.
- **Zero new raw wins on any NS16 set** for any variant. The
  19-row wrapper-only corpus is too sparse and too varied
  (tactic_template, family_tactic, generative_topk,
  retrieved_premise mixed) to teach the model a new pattern.

### Routed eval (Nat → `oversample_10x`, Set/Finset → `ns12_balanced`)

[`project/evolve/routing/ns16_router.json`](../routing/ns16_router.json)
routes Nat to `oversample_10x` (best NS16 Nat raw) and keeps
NS12's checkpoint for everything else:

| set | NS15 routed | NS16 routed | Δ |
|---|---:|---:|---:|
| `nat_defs_medium` (38) | 23 | 23 | 0 |
| `nat_defs_large_v5` (65) | 35 | 35 | 0 |
| `demo_v1` (15) | 10 | 10 | 0 |
| `ns14_nat_extra` (20) | 9 | 9 | 0 |
| `ns14_set_finset_extra` (20) | 13 | 13 | 0 |
| `ns16_nat_iff_extra` (17) | 1 | 1 | 0 |
| `ns16_nat_div_mod_extra` (25) | 0 | 0 | 0 |
| `ns16_nat_order_extra` (16) | 3 | 3 | 0 |
| `ns16_nat_mixed_extra` (28) | 0 | 0 | 0 |

NS16 routed is *exactly* NS15 routed. The Nat sub-model swap
(NS15 nat_oversample → NS16 oversample_10x) made no measurable
difference — the two checkpoints solve the same theorems on the
same states.

### Wrapper compatibility (NS9 genome + NS16 routed)

| set | NS9 + NS15 routed | NS9 + NS16 routed |
|---|---:|---:|
| `nat_defs_medium` (38) | 37 | **37** |
| `nat_defs_large_v5` (65) | 49 | **49** |
| `demo_v1` (15) | 11 | **11** |
| `ns14_nat_extra` (20) | 9 | 9 |
| `ns14_set_finset_extra` (20) | 13 | 13 |
| `ns16_nat_iff_extra` (17) | 2 | 2 |
| `ns16_nat_div_mod_extra` (25) | 1 | 1 |
| `ns16_nat_order_extra` (16) | 3 | 3 |
| `ns16_nat_mixed_extra` (28) | 0 | 0 |

The wrapper still preserves the NS9 baseline (37/38 medium, 49/65
large), still adds +1 on demo_v1 (11/15), and still gives the same
+2 wrapper-only wins on NS16 sets. Net wrapper-compatible
behavior is unchanged from NS15.

## Stage 7 — theorem-level transfer

[`scripts/ns16_compare_transfer.py`](../../../scripts/ns16_compare_transfer.py)
emits the full table at
[`ns16_transfer_analysis.md`](ns16_transfer_analysis.md).
Headline findings:

- **NS14 wrapper-only Nat wins (8): retained 8/8.** NS16 routed
  proves all the iff-omega theorems NS15 first transferred —
  `Nat.pred_eq_succ_iff`, `Nat.pred_sub`, `Nat.lt_of_lt_pred`,
  `Nat.lt_sub_iff_add_lt'`, `Nat.sub_sub_sub_cancel_right`,
  `Nat.add_sub_sub_cancel`, `Nat.sub_add_sub_cancel`,
  `Nat.sub_lt_sub_iff_right`.
- **NS16 wrapper-only wins (2): retained 0/2 by raw.** The two
  wrapper-only NS16 wins (`Nat.lt_find_iff`,
  `Nat.mul_add_mod_of_lt`) did not become raw wins in any
  variant. They still need the wrapper to close.
- **No medium/large theorem changed status** between NS15 and
  NS16 routed (both solve the same 23 / 35 theorems
  respectively).

The model still emits `exact ⟨fun h => by omega, fun h => by omega⟩`
on the iff goals it learned in NS15 — we verified by reading
per-theorem `tactic` fields. No *new* learned templates beyond
`pt_iff_8`.

## What this answers

1. **Can wrapper-only transfer scale beyond a single template?**
   On the evidence here, **no** — not at 10×/20× / 19-row corpus.
   The model needs the patterns to be *uniform* (NS14: every win
   used the same iff-omega tactic) and the rows to be *abundant
   enough per pattern*. Mixing tactic_template + family_tactic +
   generative_topk in one 19-row corpus does not teach any one
   pattern well enough to break greedy decoding.

2. **Is there a fundamental ceiling on the NS16 Nat sets?**
   Likely yes for the current wrapper genome — wrapper itself
   only proves 6/86 across these sets, so the wrapper-only signal
   to mine is small to begin with. The harder sets (div_mod,
   mixed) are not closeable by any current pattern.

3. **Did we break anything?** Almost nothing. `oversample_10x`
   loses 1 demo theorem (9/15 vs 10/15) when used raw, but the
   router avoids it. `oversample_20x` mildly regresses everywhere
   (-1) and `curriculum_continue` regresses heavily.

4. **Is the NS9 wrapper still load-bearing?** Yes — on
   `nat_defs_medium` it goes 23 → 37, adding 14 wins; on
   `nat_defs_large_v5` it goes 35 → 49, adding 14 wins; on
   `demo_v1` it adds +1 (10 → 11); on NS16 sets it adds +2
   total. The wrapper is doing essentially all the work above
   the raw ceiling on Nat.

## Limitations

- 19 wrapper-only rows is small. With 35 pre-dedup, large eval
  produced 0 rows post-dedup because it shares states with medium.
  Running wrapper on more *distinct* theorems (Multiset, List,
  Option) would yield richer state shapes.
- Some training rows hit NS11-held-out theorems (11/19).
  Held-out eval honesty is now further eroded relative to NS14.
- The two NS16 wrapper-only wins use distinct tactics
  (`exact ⟨fun h => by omega, fun h => by omega⟩` and an `omega`
  after substitution). With only 1 row of each, oversampling
  to 10× gives 10 copies — not enough to overcome 5,500 base
  rows of competing distribution.
- We didn't try mixing more aggressive losses (label-smoothing,
  KL to base) — those could amplify minority-pattern gradient
  without bulk row duplication.
- Only the `nat_defs_*` and NS16 trace-pair extraction was done;
  the wrapper on demo_v1 and `ns14_set_finset_extra` was not
  mined for additional Set/Finset training rows.

## NS17 recommendations

1. **Per-template oversampling, not per-row.** Group rows by the
   *tactic family* (iff-omega, omega-after-divmod, etc.) and
   oversample each family separately to a target frequency
   (e.g., 100 rows per family). 19 rows of mixed origins is too
   diluted.
2. **Expand the wrapper-only theorem pool by running wrapper on
   fresh theorem surface from other Mathlib files** (List, Option,
   Multiset, Bool, Sym). Single-source (Mathlib/Data/Nat/Defs)
   gave us 14 medium/large wrapper-only theorems; reaching
   wider files should add many more.
3. **Try `aesop`-as-wrapper.** The current wrapper's omega
   templates target iff + arithmetic only. A wrapper that invokes
   `aesop` with hints on div_mod / mixed Nat theorems could reveal
   richer wrapper-only patterns to mine.
4. **Reconsider held-out discipline.** If we keep training on
   wrapper traces, we should stop calling the NS11 held-out names
   a held-out set in evaluation. Or deliberately exclude them
   from any future trace mining.
5. **Investigate whether NS16 wrapper-only wins
   (`Nat.lt_find_iff`, `Nat.mul_add_mod_of_lt`) can be unlocked**
   with deeper search (`max_steps=20`) or sample-mode decoding.
   They're in wrapper's reach; raw decode just doesn't choose
   them in the top-k.

## Files

Committed:
- `scripts/build_ns16_theorem_sets.py` — wider Nat theorem-set construction
- `scripts/build_ns16_training_data.py` — wrapper-only trace mining
- `scripts/build_ns16_datasets.py` — training dataset variants
- `scripts/ns16_run_evals.sh` — raw + wrapper baseline driver
- `scripts/ns16_run_raw_evals.sh` — per-ckpt raw eval driver
- `scripts/ns16_run_router_evals.sh` — NS16-router routed + wrapper driver
- `scripts/ns16_compare_transfer.py` — offline transfer analysis
- `project/evolve/routing/ns16_theorem_sets.json` — 86 theorems
- `project/evolve/routing/ns16_router.json` — NS16 router config
- `project/data/ns16_nat_wrapper_only_meta.json`
- `project/data/ns16_train_oversample_10x_meta.json`
- `project/data/ns16_train_oversample_20x_meta.json`
- `project/data/ns16_train_curriculum_continue_meta.json`
- `tasks.py` — added `_load_ns16_sets()` for ns16_* theorem sets
- `project/evolve/reports/ns16_expand_nat_surface_report.md` (this file)
- `project/evolve/reports/ns16_transfer_analysis.md`
- `.gitignore` — NS16 paths

Not committed (gitignored / regeneratable):
- `project/data/ns16_*.jsonl`
- `project/models/gen_v5_ns16_*`
- `project/models/gen_v5_ns16_*_training.log`
- `project/evolve/eval_runs/{gen_v5_ns16,ns16}_*`
