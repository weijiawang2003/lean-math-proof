# gen_v8 — Frontier Pairs Did Not Lift; Net Regression of 1

**Date:** 2026-05-03
**Run dir:** `experiments/gen_v8_20260502_192009/`
**Checkpoint:** `project/models/gen_v8/` (t5-base, 15 epochs, lr 5e-5, batch 8, seed 42, val-split 0.1; identical to gen_v7 except for the augmented pool)
**Eval:** `eval/eval-f41156ca/`
**Train runtime:** 6h 12m on M4 Pro (slower than the brief's 3h estimate; loss converged to 0.394 train / ~0.35 eval).

## Headline

| Checkpoint | Pool | curriculum_all |
|---|---|:-:|
| gen_v7 (baseline) | seq2seq_data_v5 (5,577 ex) | **24/30** |
| **gen_v8** | seq2seq_data_v5 + 126 frontier pairs (5,703 ex) | **23/30** |

**Net change: −1.** All four Finset frontier theorems still fail. `Nat.mul_add_mod'` still fails despite 120+ Nat training pairs added. One previously-solved Set theorem regressed (`Set.inter_subset_left`).

Per the brief's decision tree, this is the **"gen_v8 < 24/30 → regression. Something specific went wrong"** bucket. The Step A prerequisite was already known violated (1 frontier proof, not ≥3), so the negative outcome is consistent with the warning: 96% of the augmented pool was a single theorem (Nat.mul_add_mod'), the augmentation was 2.2% of the total pool by count, and that's not enough to shift t5-base's priors — but it IS enough to introduce noise that knocks one theorem out.

## Per-theorem disposition diff (gen_v7 vs gen_v8)

29 theorems unchanged; 1 lost; 0 gained.

| Theorem | v7 | v8 | Δ | v8 winning tactic |
|---|:-:|:-:|---|---|
| Set.subset_univ | ✗ | ✗ | — | — |
| Set.empty_subset | ✗ | ✗ | — | — |
| Set.ite_univ | ✓ | ✓ | — | `simp [Set.ite]` |
| **Set.inter_subset_left** | ✓ | ✗ | **lost** | (all 8 errored at step 1) |
| Set.inter_subset_right | ✓ | ✓ | — | `simp [Set.subset_def]` |
| Set.subset_union_left | ✓ | ✓ | — | `tauto` |
| Set.subset_union_right | ✓ | ✓ | — | `tauto` |
| Nat.mul_add_mod' | ✗ | ✗ | — | — |
| Finset.mem_insert | ✗ | ✗ | — | — |
| Finset.mem_singleton | ✗ | ✗ | — | — |
| Finset.disjoint_insert_right | ✗ | ✗ | — | — |
| Finset.insert_comm | skip | skip | — | (LeanDojo lookup failure, same as before) |
| (20 other theorems all proved by both) | ✓ | ✓ | — | aesop / tauto / simp |

## Frontier-theorem tactic emission (this is the experiment)

For each of the four evaluable frontier theorems, here are gen_v8's top-8 beam outputs (all errored — none of the canonical / search-found tactics appear):

**Nat.mul_add_mod'** (training pool added 120 pairs, including the canonical proof 11× and `rw [Nat.mul_comm, Nat.mul_add_mod]` 8×):
```
simp [Nat.add_zero], simp [List.append_nil], simp [List.length_cons],
simp [Nat.one_mul], simp [List.nil_append], simp [List.map],
simp [Nat.zero_add], simp [Nat.mul_one]
```
None of these are the trained closers. The beam continues to emit
the v5-pool priors and ignores the new (state, tactic) pairs.

**Finset.mem_insert** (training pool added 4 pairs from Step A — all
`simp [Finset.mem_insert, ...]` partial-progress transitions, none
closing):
```
step 1: aesop  (advances state, doesn't close)
step 2: simp [Set.inter_empty], simp [Set.empty_inter],
        simp [Set.union_self], simp [Set.inter_self],
        simp [Set.ite], tauto, simp_all   — all errored
```
No `Iff.rfl`, no `simp [Finset.mem_insert]`. The 4 partial-progress
pairs added didn't propagate.

**Finset.mem_singleton, Finset.disjoint_insert_right:** same pattern —
no transferred idioms in the beam.

## Why the regression: idiom-preference perturbation

`Set.inter_subset_left` is the load-bearing data point. In gen_v7's
trace, it proved in **two steps**:

```
step 1: simp [Set.subset_def]   (TacticState — partial progress)
step 2: tauto                   (ProofFinished)
```

In gen_v8, `simp [Set.subset_def]` is **not in the top-8 beam** for the
initial state. v8 emits `simp_all, aesop, simp [Set.ext_iff],
simp [Set.union_self], simp [Set.inter_self], tauto, simp [List.map],
simp [*]` — `tauto` is there but never gets applied because step 1
never reaches a state where `tauto` succeeds.

This is the same dilution effect documented in the prior
capacity-isolation writeup. Adding 126 examples (heavily skewed toward
one theorem) shifted the relative frequencies enough to push
`simp [Set.subset_def]` out of the top-8 for this state. The new
content didn't add any frontier capability; it just nudged the
distribution in a way that randomly broke one previously-working chain.

This explains the regression mechanism crisply: t5-base trained on
{v5 + 126 Nat-skewed} is *not* the same model as t5-base trained on v5,
but the differences are noise rather than signal. With only 4-6 pairs
per Finset theorem (none of which actually closed the goal — they were
the circular `simp [Finset.X]` partial-progress records), there was no
real signal to learn for the frontier. The Nat pairs are a single
theorem's worth of content, ~120 examples, ~2.1% of the pool — below
any reasonable threshold for shifting t5-base's beam preferences in a
useful direction.

## Confirmation status

- Step A prerequisite (≥3 frontier proofs): **violated** — only 1.
- Brief's decision tree:
  - +3 or more on frontier → expert iteration works. **No.**
  - +1 to +2 → partial transfer. **No.**
  - 0 on frontier → upsample. **Yes for frontier, but ALSO net regression.**
  - **Net regression** (gen_v8 < gen_v7): something went wrong. **Yes.**

## Recommendation

**Don't repeat this exact experiment.** The augmentation was too small,
too unbalanced, and too low-signal to lift the wall. Three concrete
next moves, in order of cost-effectiveness:

1. **Step A retry first.** The v5 search produced 1 real proof and 0
   Finset proofs. Before retraining again, fix what blocked Step A:
   (a) treat `trivial` as crash-prone in `actions.py` like `exact?`
   and `apply?` already are — the REPL crash at depth 2 capped Finset
   search depth; (b) read the actual Mathlib proof of
   `Finset.mem_insert` at commit 29dcec07 and add the genuinely-closing
   tactic to a `search_v6` (the `Iff.rfl` diagnosis was wrong; v5's
   FINDINGS proves it). If `search_v6` cracks 3-4 Finset frontier
   theorems, *then* retrain on a larger and more balanced pool.
2. **If retrain again, upsample heavily.** 126 pairs in 5,703 = 2.2%.
   At 10× duplication of frontier pairs, that becomes 1,260 pairs in
   6,837 = 18%. That's the level the prior dilution writeup found
   meaningful — the absolute count of `simp [Set.subset_def]` in v6
   was *higher* than v5, but its relative frequency dropped from 1.99%
   to 1.14% and that's where the −2 regression came from. A 10×
   upsample also wouldn't fix the Finset-no-real-proofs problem,
   though, so combine with (1).
3. **Drop the `Nat.mul_add_mod'` proof into the pool without
   retraining.** Beam already finds it instantly. Either of:
   - `rw [Nat.mul_comm, Nat.mul_add_mod]`
   - `simp [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]`
   becomes a free curriculum point if it's appended as a (state, tactic)
   line and used at *inference* (e.g. as a tactic-injection fallback
   policy) rather than via training. This bypasses the
   "training-data composition needs balance" problem entirely for the
   one theorem search already cracked.

The expert-iteration loop is not falsified by this run — but it isn't
going to work on a single-proof, 1-theorem feedback batch on a
5,577-example baseline. It needs ≥3 frontier proofs and ideally
upsampling, both of which the prior brief's prerequisite encoded.

## Files

- Edit: none in canonical files. Built `experiments/gen_v8_20260502_192009/seq2seq_v5_plus_frontier.jsonl` by concatenating `project/seq2seq_data_v5.jsonl` (5,577 ex) and `experiments/gen_v8_20260502_192009/frontier_seq2seq.jsonl` (126 ex from Step A traces).
- Model: `project/models/gen_v8/` (5.8 GB total, 2 retained checkpoints + final). `save_total_limit=2` confirmed before launch.
- Eval artifacts: `experiments/gen_v8_20260502_192009/eval/eval-f41156ca/{config.json,metrics.json,traces.jsonl}`.
- Train + eval logs: `experiments/gen_v8_20260502_192009/{train.log,eval.log}`.

## Disk safety

Free space at end: 194 GB on root. No incident. `save_total_limit=2`
held throughout — checkpoints rotated at 1284, 1926, 3210, 3852, 4494,
5136, 7062, 7704, 8346, 9630, with at most 2 retained at any moment.

## Limitations

- Single seed (42), beam decode only. Sampling-decode results not measured; could plausibly recover one theorem at the cost of variance.
- The "Set.inter_subset_left regression is from idiom-preference perturbation" claim is the most likely mechanism but not conclusively verified. Could also be a t5-base optimization artifact at this exact (data, lr, seed) combination. To pin it down would require a second seed; out of scope for one-shot training.
- gen_v8 was trained on the augmented pool with no curriculum/weighting on the new pairs — they're treated as uniformly-weighted training examples. Any of the upsampling or curriculum-pretrain strategies the brief suggests would change this.
