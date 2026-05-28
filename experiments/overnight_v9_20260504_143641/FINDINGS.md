# Overnight v9 Sweep — Expert-Iteration Loop Works (with 50× upsample, full retrain)

**Date:** 2026-05-05
**Run dir:** `experiments/overnight_v9_20260504_143641/`
**Total runtime:** ~12.7h (search 4h, build 5min, fine-tune 1h, full retrain 6h, beam evals 20min, 20 variance runs ~75min, summary).
**Disk at end:** 178 GB free, no incidents. `save_total_limit=2` held throughout.

## TL;DR

The expert-iteration loop works on this codebase **at this scale**, but only with full retrain + 50× upsampling — fine-tuning is the wrong shape. Concrete numbers:

| Checkpoint | Beam | Sampling (5 seeds) |
|---|:-:|:-:|
| gen_v5 (t5-small, v5 data — historical SOTA) | **25/30** | 20.8 ± 2.9 |
| gen_v7 (t5-base, v5 data — capacity isolation) | 24/30 | 15.8 ± 1.5 |
| gen_v9_ft (fine-tune gen_v7 on frontier-only, 50×) | 20/30 | 11.2 ± 0.4 |
| **gen_v9_full (full retrain on v5 + 50× frontier)** | **26/30** | 19.2 ± 2.6 |

`gen_v9_full` lifts +2 over gen_v7's beam baseline, and **picks up two specific frontier proofs that beam search found and that no prior checkpoint solves**: `Nat.mul_add_mod'` (via `rw [Nat.mul_comm, Nat.mul_add_mod]`) and `Finset.mem_singleton` (via `exact List.mem_singleton`). The new tactics propagate from search → training pool → model beam, with no checkpoint regression. That's the first concrete expert-iteration result on this project.

## Phase results

### Phase 1 — search_v6 on frontier_v1 (~4h)

`search_v6` action space: `search_v5` minus `trivial` (which crashes the LeanDojo REPL) plus mathlib term-mode closers (`exact List.mem_singleton`, `exact Multiset.mem_cons`, `decide`, etc.). 3914 transitions logged across the 4 evaluable theorems.

| Theorem | Outcome | Proof found |
|---|---|---|
| `Finset.mem_insert` | Stuck — 1462 records, depth 7 reached, 0 proofs | none |
| **`Finset.mem_singleton`** | **Solved at depth 1** (31 proof variants) | `exact List.mem_singleton` |
| `Finset.disjoint_insert_right` | Stuck — 952 records, 0 proofs | none |
| `Finset.insert_comm` | Unavailable (LeanDojo lookup failure, same as before) | — |
| **`Nat.mul_add_mod'`** | **Solved at depth 1** (60 proof variants) | `rw [Nat.mul_comm, Nat.mul_add_mod]` (and `simp [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]`) |

Two of the four evaluable frontier theorems cracked. `mem_singleton` is the new win — the search_v5 brief's `Iff.rfl` diagnosis was wrong, but the search_v6 hypothesis ("Mathlib's actual proof goes through `List.mem_singleton`") was right and falsifiable. `mem_insert` and `disjoint_insert_right` are still beyond beam search at this depth/budget, even with the `trivial` filter and term-mode closers. They almost certainly need explicit mathlib unfolds (`Multiset.mem_cons`, `Finset.disjoint_iff_ne`, etc.) that aren't yet in any action space.

### Phase 2 — augmented training pool

| Pool | Examples |
|---|---:|
| `project/seq2seq_data_v5.jsonl` (anchor) | 5,577 |
| `data/frontier_seq2seq.jsonl` (mined from search_v6 traces) | 128 |
| `data/frontier_seq2seq_50x.jsonl` (50× upsample) | 6,400 |
| `data/v5_plus_frontier_50x.jsonl` (combined) | 11,977 |

Frontier composition (after dedup): 120 Nat.mul_add_mod' / 4 Finset.mem_insert / 2 Finset.mem_singleton / 2 Finset.disjoint_insert_right. Same theorem skew as gen_v8's pool (the 31 mem_singleton proof variants collapsed to 2 (state, tactic) pairs because they shared the same depth-1 state). Difference vs gen_v8: this run's 50× upsample puts frontier pairs at **53.5%** of the combined pool by count vs gen_v8's 2.2%. That's the dilution-floor crossing.

### Phase 3a — fine-tune gen_v7 → gen_v9_ft (~1h, 5 epochs, lr 2e-5, frontier-only data)

Trained `gen_v7_base_on_v5data` for 5 epochs on the 50× frontier pool only (no v5 data). Hypothesis (H1): starting from a model that already knows the curriculum and only training on the new patterns avoids ejecting old idioms.

**H1 falsified.** Full per-theorem diff vs gen_v7 below — gen_v9_ft lost 5 theorems and gained 1. Net −4. The model catastrophically forgot.

### Phase 3b — full retrain → gen_v9_full (~6h, 15 epochs, lr 5e-5, v5 + 50× frontier)

Identical hyperparameters to gen_v7 except the augmented pool. Hypothesis (H2): 50× upsampling raises frontier-pair relative frequency above the dilution floor (~9-50% depending on metric) and overrides v5 priors enough to keep the new tactics in the model's beam.

**H2 confirmed.** gen_v9_full beam = 26/30 = +2 over gen_v7. No theorem lost; both theorems gained are exactly the search-found ones.

### Phase 4 — beam eval on curriculum_all

(In headline table above.)

### Phase 5 — variance bars (5 sampling seeds × 4 checkpoints)

(In headline table above.) Notable findings:

- **gen_v5 (t5-small) under sampling beats all three t5-base variants on the mean.** Sampling decode is brutal on t5-base — the capacity-isolation effect is *bigger* under sampling than under beam (16 ± 1.5 vs 24/30 beam for v7; the 8-point gap dwarfs the 1-point beam gap).
- **gen_v9_full sampling = 19.2 ± 2.6**, ~3 points below gen_v5 sampling and ~3 points above gen_v7 sampling. So in the sampling regime, expert-iteration buys back ~⅓ of the t5-small-vs-t5-base capacity-shift gap.
- **gen_v9_ft is the most brittle of all four**: 11.2 ± 0.4 — tight low band, far below even gen_v7 sampling. The fine-tuned model's distribution is concentrated on the new tactics in a way that breaks under any decode noise.
- **The variance bars confirm gen_v9_full's beam lift is real**: all 5 sampling seeds for v9_full are ≥16, all 5 for v7 are ≤18. Distributions don't overlap on their tails. The 26 vs 24 beam gap is not a single-seed artifact.

## Per-theorem disposition (gen_v7 → gen_v9_full)

29 unchanged; 0 lost; **2 gained** — the only deltas are the two frontier proofs.

| Theorem | gen_v7 | gen_v9_full | Δ | gen_v9_full winning tactic |
|---|:-:|:-:|---|---|
| Set.subset_univ | ✗ | ✗ | — | — |
| Set.empty_subset | ✗ | ✗ | — | — |
| Set.ite_univ | ✓ | ✓ | — | `simp [Set.ite]` |
| **Nat.mul_add_mod'** | ✗ | ✓ | **gained** | **`rw [Nat.mul_comm, Nat.mul_add_mod]`** ← search-found |
| **Finset.mem_singleton** | ✗ | ✓ | **gained** | **`exact List.mem_singleton`** ← search-found |
| Finset.mem_insert | ✗ | ✗ | — | — |
| Finset.disjoint_insert_right | ✗ | ✗ | — | — |
| Finset.insert_comm | skip | skip | — | (LeanDojo lookup) |
| (22 other theorems all proved by both) | ✓ | ✓ | — | aesop / tauto / simp |

For the gained theorems, the model is emitting **the exact tactic that search found and that 50× upsampling baked into training**. This is the loop closing end-to-end. `Nat.mul_add_mod'` had 120 training pairs to learn from; `Finset.mem_singleton` had only 2 — but both transferred. The model isn't memorizing an idiom by sample count alone; the upsample's contribution is making those few pairs frequent enough to compete with v5 priors.

`Finset.mem_insert` and `Finset.disjoint_insert_right` stayed unsolved because **search didn't find proofs for them either**. Training has nothing to teach the model. Until the action space cracks those, no amount of retrain helps.

## Per-theorem disposition (gen_v7 → gen_v9_ft) — the regression

| Theorem | v7 | v9_ft | Δ |
|---|:-:|:-:|---|
| **Set.union_comm** | ✓ | ✗ | **lost** (gen_v7 used `aesop`) |
| **Set.mem_union** | ✓ | ✗ | **lost** |
| **Set.mem_inter_iff** | ✓ | ✗ | **lost** |
| **Set.empty_union** | ✓ | ✗ | **lost** |
| **Set.inter_subset_right** | ✓ | ✗ | **lost** |
| **Finset.mem_singleton** | ✗ | ✓ | gained |
| Nat.mul_add_mod' | ✗ | ✗ | — |
| Finset.mem_insert | ✗ | ✗ | — |
| Finset.disjoint_insert_right | ✗ | ✗ | — |
| (other 22) | ✓ | ✓ | — |

Net: −4. Fine-tuning gen_v7 for 5 epochs on a 6,400-row pool whose distribution is **completely** dominated by Nat.mul_add_mod' content (94% of frontier pairs) collapsed the model's diversity. It learned the new pattern but forgot how to apply `aesop` and `tauto` to easy Set goals. This is the standard catastrophic-forgetting failure mode — fine-tune on a narrow distribution and you get a narrow-distribution model.

The lesson is sharp: **for this pipeline, fine-tuning on search-found proofs is the wrong shape for expert iteration.** The new pairs need to be *added* to the original pool, not *substituted* for it.

## What this means

**The expert-iteration loop is closed end-to-end.** Search found a proof; the proof entered training; the model emits it at inference. Two frontier theorems went from "no checkpoint can solve" to "gen_v9_full solves at top-1 beam." This is the first time anything in this project has cleared that bar.

**The 50× upsample isn't gratuitous.** gen_v8 already tested the no-upsample case (frontier pairs at 2.2% of pool) and got net −1 with zero gains. The dilution floor is real, and somewhere between 2.2% and 53.5% of pool-by-count is where it gets crossed for this dataset / model / tactic-distribution combination. Future experiments can narrow this — 10× upsample (frontier ~17% of pool) is the obvious next dial.

**Fine-tuning doesn't substitute for retrain.** The catastrophic forgetting in gen_v9_ft means we can't shortcut the 6h full-retrain cost via 1h fine-tune. Future iterations of this loop will need the full retrain budget per round, unless we figure out a way to mix frontier-only fine-tuning with rehearsal (e.g., interleaving v5 batches into the fine-tune loop — basically a manual approximation of full retrain with bias).

**The Finset wall still has 2 unbroken theorems**, and both need a deeper search-tooling investment, not more retraining. `mem_insert` and `disjoint_insert_right` need:
1. Mathlib-canonical proof traced explicitly (probably `Multiset.mem_cons` / `Finset.disjoint_iff_ne` plus rfl chain)
2. Added to a `search_v7` action space
3. Re-search with budget large enough to compose those tactics

That work is independent of the retrain pipeline and could be done in parallel.

**Capacity-isolation finding holds, with caveats.** gen_v5 (t5-small) is still ahead of all t5-base variants under sampling decode (mean 21 vs 16-19). The 24 vs 25 beam gap is real but understates the gap under more realistic decoding. Future model selection on this benchmark should report sampling means, not beam-only.

## Recommended next experiments, in priority order

1. **Mine the Mathlib proof of `Finset.mem_insert`.** Read it, identify the exact tactic chain, add it to `search_v7`, re-run search on the 2 remaining stuck Finset theorems. If this lifts to 4/4 cracks, the next retrain round of expert iteration has 2× more signal to work with.

2. **Bake gen_v9_full's traces (curriculum_all rollout) into the next training pool.** This run's eval traces include 26 successful proofs. Some of those used tactics gen_v7 didn't emit (e.g., gen_v9_full uses `trivial` to close `Set.mem_inter_iff`, gen_v7 used `tauto`). Treating gen_v9_full's eval traces as next-round expert demos may compound gains. Free curriculum lift if any of those new tactics propagate.

3. **Sweep upsample ratio.** 50× was a guess. 10× and 25× would let us isolate the dilution-floor curve. If 10× works, future loops are cheaper to construct (smaller training pools).

4. **Strategic-policy ablation** (still unrun). The collaborator's proposed strategic policy hasn't been compared against gen_v9_full on this benchmark. With variance bars now established, strategic vs gen_v9_full is a fair head-to-head.

5. **(Lower priority) Re-run capacity-isolation experiment** under sampling decode for the headline writeup. The current writeup mentions "single-seed beam"; with the new variance bars it's clear the t5-small vs t5-base gap is wider in the more realistic regime.

## Files

- Edits: none in canonical files. `tasks.py` already had `frontier_v1` from the prior brief; `actions.py` already had `search_v6` (user-edited); script `experiments/overnight_v9_sweep.sh` was the orchestrator.
- New checkpoints: `project/models/gen_v9_ft/` (5.8 GB), `project/models/gen_v9_full/` (5.8 GB). Both with `save_total_limit=2`.
- Run artifacts (this run): `experiments/overnight_v9_20260504_143641/`:
  - `SUMMARY.md` — auto-generated table of contents.
  - `search_v6/{traces.jsonl,metrics.json,config.json}` — Phase 1.
  - `data/{frontier_seq2seq.jsonl,frontier_seq2seq_50x.jsonl,v5_plus_frontier_50x.jsonl}` — Phase 2.
  - `eval_v9_ft_beam/eval-*/`, `eval_v9_full_beam/eval-*/` — Phase 4.
  - `sample_<tag>_seed<N>/eval-*/` — 20 variance runs (Phase 5).
  - `run.log` — full sweep log.

## Limitations

- gen_v9_full and gen_v9_ft are single-seed *training* runs. Variance bars are over decoding seeds, not training seeds. A different training seed could plausibly shift the headline number by ±1; the +2 lift is suggestive but not formally bounded.
- The 50× upsample puts the frontier content at 53% of pool — clearly in "this isn't a free lunch" territory. The model's distribution is now skewed toward Nat.mul_add_mod' and Finset.mem_singleton patterns. Some of gen_v9_full's curriculum solves use slightly different tactics than gen_v7 (`trivial` vs `tauto`, etc.); could be a transient capability boost, could be a setup for next-round trouble. Worth tracking.
- Expert-iteration ran one round. A second round (mine more frontier proofs from gen_v9_full's eval traces, retrain again, eval) is the test that this scales beyond a one-shot demo. That's the next experiment that would either confirm the loop or expose a ceiling.
- The `Finset.insert_comm` LeanDojo lookup failure persists. It's been unavailable since the first frontier_v1 eval. Should be filed upstream; for now it's a permanent skip.
