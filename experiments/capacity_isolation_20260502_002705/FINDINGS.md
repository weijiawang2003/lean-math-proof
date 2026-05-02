# Capacity vs. Data Shift in T5-Scale Lean Tactic Generation

**Date:** 2026-05-02
**Project:** `dojo_sandbox` (lean-supervised)
**Run artifacts:** `experiments/capacity_isolation_20260502_002705/`
**Status:** Single-seed result; variance bars pending.

## TL;DR

A four-checkpoint controlled experiment decomposes the apparent t5-base regression on `curriculum_all` (gen_v6: 22/30 vs gen_v5: 25/30) into two independent additive effects:

1. A **-1 net capacity-driven idiom reshuffle** that persists across training-data conditions (loses two specific subset facts, gains one Set.ite fact).
2. A **-2 data-shift effect** specific to the v6 training pool, recoverable by retraining on v5's tighter-frequency pool.

Together they reproduce the observed delta exactly. A third finding falls out: five "frontier" theorems (four Finset, one Nat.Defs) are unsolved by every checkpoint and are unaddressed by either capacity or data scaling — the Finset wall is the project's actual frontier, and the next intervention should target it directly.

## Background

The lean-supervised project trains T5-class tactic generators with Lean as the verifier; every (state, tactic) pair entering training has been mechanically checked. Five generative checkpoints had shipped (gen_v1 → gen_v5, all T5-small). The README's "Phase 2" hypothesis was that scaling to T5-base would help — in particular, that the negative result observed when adding retrieved-premise prefixes (gen_ckpt_v6_premise: 19/30 vs gen_v5: 25/30) was a capacity bottleneck and would invert at larger scale.

A T5-base model (`project/models/gen_v6/`) had in fact been trained, on a *different* training pool: `seq2seq_data_v6.jsonl`, 11,802 examples (vs v5's 5,577) with 119 unique tactics (vs v5's 93). Its first-ever evaluation produced **22/30**, *worse* than gen_v5's 25/30. This raised the question that motivated this experiment: is the regression a capacity problem, a data-shift problem, or both?

## Experimental setup

Four checkpoints, evaluated on `curriculum_all` (31 theorems, 30 available, deterministic beam decoding k=8, max-steps 8):

| Checkpoint | Architecture | Training data | Role |
|---|---|---|---|
| `gen_v5` | T5-small (60M) | `seq2seq_data_v5.jsonl` (5,577 ex, 93 tactics) | Reproducibility anchor |
| `gen_ckpt_v6_premise` | T5-small (60M) | `seq2seq_premise_v1.jsonl` (5,577 ex + premise prefix) | Prior negative-result baseline |
| `gen_v6` | T5-base (220M) | `seq2seq_data_v6.jsonl` (11,802 ex, 119 tactics) | First-ever t5-base eval |
| **`gen_v7`** | **T5-base (220M)** | **`seq2seq_data_v5.jsonl`** *(identical to v5's data)* | **Capacity isolation** |

`gen_v7` was trained with all hyperparameters identical to `gen_v5`'s (15 epochs, lr 5e-5, batch 8, seed 42, val-split 0.1) on v5's exact training pool. The *only* difference between `gen_v5` and `gen_v7` is model size. The only difference between `gen_v6` and `gen_v7` is training data.

## Results

| Checkpoint | Curriculum (proved/available) |
|---|---|
| `gen_v5` (T5-small / v5 data) | **25/30** |
| `gen_ckpt_v6_premise` (T5-small + premise) | 19/30 |
| `gen_v6` (T5-base / v6 data) | 22/30 |
| **`gen_v7` (T5-base / v5 data)** | **24/30** |

The per-theorem disposition matrix (subset shown; full data in `eval/eval-cb22336a/metrics.json`):

| Theorem | v5 | premise | v6 | v7 | Diagnosis |
|---|:-:|:-:|:-:|:-:|---|
| `Set.subset_univ` | ✓ | ✗ | ✗ | ✗ | persistent t5-base capacity regression |
| `Set.empty_subset` | ✓ | ✗ | ✗ | ✗ | persistent t5-base capacity regression |
| `Set.ite_univ` | ✗ | ✗ | ✓ | ✓ | t5-base capacity gain |
| `Set.inter_subset_left` | ✓ | ✗ | ✗ | ✓ | v6 data-dilution loss (fixed by v5 data) |
| `Set.inter_subset_right` | ✓ | ✗ | ✗ | ✓ | v6 data-dilution loss (fixed by v5 data) |
| `Nat.mul_add_mod'` | ✗ | ✗ | ✗ | ✗ | curriculum frontier |
| `Finset.mem_insert` | ✗ | ✗ | ✗ | ✗ | curriculum frontier |
| `Finset.mem_singleton` | ✗ | ✗ | ✗ | ✗ | curriculum frontier |
| `Finset.disjoint_insert_right` | ✗ | ✗ | ✗ | ✗ | curriculum frontier |
| `Finset.insert_comm` | ✗ | ✗ | ✗ | ✗ | curriculum frontier |

(Remaining 20 theorems are solved by all four checkpoints.)

## The decomposition

Every theorem-level transition has an attributed cause:

### 1. Persistent t5-base capacity regressions

`Set.subset_univ` and `Set.empty_subset` are solved by t5-small but lost by *both* t5-base configurations. Since `gen_v7` was trained on identical data to `gen_v5` and only model size differs, these losses are pure capacity effects.

In `gen_v5`, both are solved in one step by `simp [Set.subset_def]`. In both `gen_v6` and `gen_v7`, all 8 of the model's beam candidates error on these specific goals; `simp [Set.subset_def]` is no longer in t5-base's top-8 for these states. The larger model's beam preference has shifted to other tactics that are wrong for these particular subset facts.

### 2. T5-base capacity gain

`Set.ite_univ` is solved by *both* t5-base configurations and missed by t5-small. Both `gen_v6` and `gen_v7` emit `simp [Set.ite]`, the canonical proof. T5-small's top-8 doesn't contain this tactic for this goal; t5-base's does. This is the inverse phenomenon to the regressions, with the same explanation: capacity scaling shifts which idioms are surfaced in the beam.

### 3. v6 data-dilution losses

`Set.inter_subset_left` and `Set.inter_subset_right` are solved by `gen_v5` and `gen_v7` (both trained on v5's pool) but lost by `gen_v6` (trained on v6's pool). Since `gen_v6` and `gen_v7` share architecture and differ only in training data, these losses are pure data-shift effects.

This corroborates the relative-frequency dilution hypothesis: v6's pool, despite being 2× larger in absolute terms, dilutes the *fractional* frequency of high-confidence idioms. `simp [Set.subset_def]` appears 111 times in v5 (1.99% of pool) and 135 times in v6 (1.14% of pool) — more often in absolute count, less often relative to everything else. T5-base trained on the diluted pool inherits the shifted distribution; T5-base trained on the original pool does not.

### 4. The Finset wall

Five theorems are unsolved by *every* checkpoint: `Nat.mul_add_mod'`, `Finset.mem_insert`, `Finset.mem_singleton`, `Finset.disjoint_insert_right`, `Finset.insert_comm`. Four are Finset. None of capacity scaling, data scaling, or premise injection moves any of them.

A retriever-quality probe (`experiments/overnight_*/retriever_probe.json`) earlier measured 0% recall@15 across 14 evaluable `Finset.Basic` proofs in `project_state.json` — the static premise catalog in `premise_retriever.py` is missing the lemmas Finset proofs actually cite. This is the same bottleneck appearing at a different layer of the pipeline: the models can't propose tactics referencing premises they've never seen.

## Arithmetic check

The decomposition reproduces the headline numbers exactly:

```
gen_v5     (25/30)   baseline
            -2       capacity regression (subset_univ, empty_subset)
            +1       capacity gain        (ite_univ)
gen_v7     (24/30)   ✓

gen_v7     (24/30)   same architecture as v6, v5 pool
            -2       data-dilution        (inter_subset_left, inter_subset_right)
gen_v6     (22/30)   ✓
```

There are no leftover unexplained gaps.

## What this means

Three claims, each falsifiable via the same experimental design:

**1. Capacity scaling at this scale is not strictly improving.** It is an idiom-preference shift. Small and large models memorize different vocabularies of useful tactics, even on identical training data. On this curriculum the net delta is -1 (the +1 gain doesn't quite compensate for the -2 loss); on a benchmark with more `Set.ite`-style goals and fewer simple subset goals, t5-base would look strictly better. The *direction* of the capacity effect is benchmark-composition-dependent; the *fact* that scaling reshuffles rather than uniformly improves seems robust.

**2. Data scaling without idiom-frequency preservation introduces independent regressions.** Doubling the training pool from 5,577 to 11,802 examples while expanding the tactic vocabulary from 93 to 119 produced -2 theorems on this curriculum, recoverable by training on the smaller, denser pool. The relevant signal is *relative* frequency, not absolute count — the absolute count of `simp [Set.subset_def]` *increased* between v5 and v6 (111 → 135), but its relative frequency dropped (1.99% → 1.14%) and that's what t5-base learned from. For ExIt-style training pool growth, *how* the pool is grown matters as much as how much.

**3. The Finset frontier is upstream of capacity and data scale.** No checkpoint solves any Finset.Basic theorem in this curriculum, regardless of model size or training pool. The bottleneck is in what tactics the training pool exposes and what the retriever surfaces at inference time, not in the gradient signal. This is the highest-leverage open problem in the project right now: solving the Finset wall is +4 theorems on the curriculum *for any model*, larger than any effect measured in this experiment.

## Limitations

This experiment uses a single seed (42) per checkpoint with deterministic beam-k=8 decoding only. The 25 vs 24 gap is small enough that seed variance could plausibly change which checkpoint is on top; the 24 vs 22 gap is large enough to be likely robust but not formally confirmed. A multi-seed sampling sweep was set up (`experiments/overnight.sh`) but interrupted by network failure; rerunning it once network is stable would tighten the conclusions.

The benchmark is one curriculum (30 theorems, dominated by simple Set/Finset/Nat facts). The decomposition could look different on miniF2F or on a Nat.Defs-heavy slice. The +1 / -2 capacity nets to -1 here, but the same shift on a differently-composed benchmark could net positive.

The v6 pool's specific composition (assembled from accumulated project traces) may not generalize. The dilution effect documented here is for *this* pool transition; other pool-construction strategies (human curation, stratified sampling, distillation from a frontier model) could yield different distributional behavior.

`gen_v6` was trained with no `save_total_limit` and saved a full intermediate per epoch, which contributed to a 250 GB disk-fill incident during this work. Both training scripts (`train_tactic_generator.py`, `train_action_classifier.py`) now set `save_total_limit=2`. The original `gen_v6` checkpoint was unaffected; only the in-progress `gen_v7` training was bitten and restarted clean.

## Next experiments, in priority order

**1. Solve the Finset wall.** Highest-leverage move now. The retriever-quality probe identified 0% recall@15 across evaluable Finset proofs — the static premise catalog in `premise_retriever.py` is the obvious gap. Concrete plan: scan the `proof_tactics` strings in `project_state.json` for theorems in `Mathlib/Data/Finset/Basic.lean`, extract the lemma names actually cited, add them to `STATIC_PREMISES["Finset"]`, rebuild the premise index, re-eval the premise-augmented policy. Expected lift: if retrieval was the only bottleneck, the premise-augmented number should move from 19/30 toward or past 25/30.

**2. Idiom-frequency preservation.** Train a fifth t5-base variant on a stratified version of v6's pool: upsample `simp [Set.subset_def]` and the other v5 high-frequency idioms until their relative frequencies match v5's. If this recovers `Set.subset_univ` and `Set.empty_subset`, the so-called "capacity regression" is actually a subtle data dependency and t5-base can be made to match or beat t5-small on the curriculum. Adds a clean fifth point to the experimental design and resolves the capacity question definitively.

**3. Variance bars.** Re-run the four-checkpoint comparison with sampling decoding (temperature 0.8, top-p 0.95) across 5 seeds each, to confirm the -2/+1/-2 decomposition is robust under decoding noise rather than an artifact of one seed's beam. Required for any external write-up but not blocking the next experiment.

The first two together would close out the project's central scaling-vs-data question with publishable rigor.

## Reproduction

All four checkpoints exist on disk under the paths listed above. Eval is reproducible with:

```bash
for ckpt in project/models/gen_v5 project/gen_ckpt_v6_premise \
            project/models/gen_v6 project/models/gen_v7_base_on_v5data; do
  policy=generative
  [[ "$ckpt" == *premise* ]] && policy=premise_augmented
  python eval_rollout_all.py --theorem-set curriculum_all \
    --ckpt-dir "$ckpt" --policy-type "$policy" \
    --top-k 8 --max-steps 8 --decode-mode beam
done
```

Training of `gen_v7` is reproducible with `bash experiments/capacity_isolation.sh` (assumes `save_total_limit=2` in `train_tactic_generator.py`, in place since 2026-05-02).

The frequency-comparison check is reproducible with the script in this run's artifact directory; raw counts:

| Tactic | v5 count | v5 fraction | v6 count | v6 fraction |
|---|---:|---:|---:|---:|
| `simp [Set.subset_def]` | 111 | 1.99% | 135 | 1.14% |
| `simp [Set.ite]` | 52 | 0.93% | 76 | 0.64% |
| `aesop` | 542 | 9.72% | 608 | 5.15% |
| `simp` | 88 | 1.58% | 213 | 1.80% |

(Pool totals: v5 = 5,577 examples; v6 = 11,802 examples.)
