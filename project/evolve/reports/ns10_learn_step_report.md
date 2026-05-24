# NS10 — Learn step: trace-to-training pipeline

The skeleton-evolution arc (NS5–NS9) produced a 17-skeleton compact
genome proving 37/38 medium with the *unchanged* `gen_v5`
checkpoint. NS10 takes the next AlphaEvolve step:

> Search/Evolve produced verified traces; now train a model on
> those traces and see whether raw model performance improves.

This report covers dataset construction, training feasibility, a
small fine-tune (`gen_v5_plus1`), and the raw-model evaluation
result.

## Hard constraints respected

- `gen_v5` checkpoint untouched.
- New checkpoint at `project/models/gen_v5_plus1/` only.
- Original datasets (`project/seq2seq_data_v5.jsonl`, etc.) unchanged.
- NS9 wrapper artifact unchanged.
- Run artifacts (traces, eval dirs, model checkpoints) gitignored.

## Stage 1 — locate verified traces

Traces consumed by `scripts/build_ns10_training_data.py`:

  - `project/evolve/ns9_runs/` (NS9 baseline + sweep traces with
    NS9 retrieval-gate; 17-skeleton best)
  - `project/evolve/ns7_runs/` (NS7 baseline with stable_id
    attribution)
  - `project/evolve/ns6_runs/` (NS6 baseline + sweep with assist
    attribution)
  - `project/evolve/autonomous_runs/` (earlier v5 autonomous loop)

Total: **128 traces.jsonl files** across **5,159 episodes**.

## Stage 2 — supervised dataset

`scripts/build_ns10_training_data.py` walks every trace, groups by
episode, and keeps:

  - **Close transitions** (`proof_finished=True`).
  - **Advance transitions** that lead to a close within K=3
    accepted steps (`advance_assist`).

Filters:
  - Self-reference: tactic contains its own theorem `full_name`.
  - Long tactic (>200 chars) or long state (>2500 chars).
  - Held-out theorems (NS10 default = 9: AM_GM, the v5 priority-
    template wins, `Nat.div_lt_iff_lt_mul'`, `dvd_iff_div_mul_eq`,
    `sqrt_lt`, `pow_lt_pow_iff_left`).
  - Origins outside the allow-set (default: tactic_template,
    family_tactic, generative_topk, term_builder, fallback_tactic;
    retrieved_premise is opt-in via `--include-retrieved` because
    it isn't reproducible without the retriever at inference time).
  - Dedup by `(state_pp, tactic)`.

### Dataset summary

| field | value |
|---|---|
| total pairs | **143** |
| unique theorems | 51 |
| trace files consumed | 128 |
| held-out trace rows skipped | 2,247 |
| roles | close: 103, advance_assist: 40 |
| origins | tactic_template: 46, fallback_tactic: 49, generative_topk: 28, term_builder: 15, family_tactic: 5 |

Files:
  - `project/data/ns10_evolve_train.jsonl` (~30 KB)
  - `project/data/ns10_evolve_train_meta.json`

Each row carries `prompt`, `tactic` (and `completion` alias),
`theorem`, `theorem_set`, `origin`, `source_run`, `state_hash`,
`tactic_hash`, `skeleton_name`, `skeleton_stable_id`, `role`, and
`assist_distance`.

Combined-with-v5 dataset for comparison runs (not used in the
headline fine-tune):

  - `project/data/ns10_combined_train.jsonl` = 5,577 v5 +
    143 ns10 = **5,720 total pairs**.

## Stage 3 — training feasibility

`train_tactic_generator.py` is the existing pipeline; it accepts
any HF model id *or* a local checkpoint directory. Base model:
T5-small (60M params, 242 MB safetensors).

Hardware: CPU + MPS (no CUDA). Smoke test (20 examples × 1 epoch)
finished in **2.7 s**, confirming the pipeline works end-to-end
on this machine.

Estimated time per epoch on full 143-example NS10 set: ~10s on
MPS. Full 5-epoch fine-tune: ~50s + eval/save overhead → ~2 min
total.

## Stage 4 — smoke test

```
python train_tactic_generator.py \
    --data /tmp/ns10_smoke.jsonl \      # first 20 lines of NS10
    --model project/models/gen_v5 \
    --output-dir project/models/gen_v5_ns10_smoke \
    --epochs 1 --batch-size 4 --val-split 0.0
```

Outcome: trained, saved, no errors. `train_runtime=2.675 s`,
`train_loss=0.2565`. Output not committed.

## Stage 5 — full small fine-tune

```
python train_tactic_generator.py \
    --data project/data/ns10_evolve_train.jsonl \
    --model project/models/gen_v5 \              # start from gen_v5
    --output-dir project/models/gen_v5_plus1 \
    --epochs 5 --batch-size 4 --val-split 0.1 \
    --max-src-len 512 --max-tgt-len 64 \
    --lr 1e-5                                    # small lr — preserve base
```

The lr is intentionally low (1e-5 vs the default 5e-5) so the
small fine-tune doesn't *forget* what gen_v5 already knows. The
goal is *additive* learning of the wrapper-discovered tactics on
top of the existing v5 knowledge.

Outcome: completed in ~2 min. 143 examples, 128 train / 14 val
(10% split, seeded). Final eval_loss ~3.4. Output written to
`project/models/gen_v5_plus1/` (gitignored — 242 MB).

## Stage 6 — evaluation

### Raw model comparison

| policy | medium proved | new theorems |
|---|---|---|
| **raw `gen_v5`** | **3/38** | `Nat.lt_iff_add_one_le`, `Nat.pred_eq_of_eq_succ`, `Nat.succ_succ_ne_one` |
| **raw `gen_v5_plus1`** | **4/38** | the same 3, plus `Nat.add_eq_right` (NEW) |

**Δ = +1 theorem proved by the raw model after Learn step**, no
regressions. The new theorem (`Nat.add_eq_right`) was in the NS10
training set with tactic
`exact ⟨fun h => by omega, fun h => by omega⟩` (the canonical
`pt_iff_8` template from NS9 best). The model learned this
emission pattern from the trace data and applied it successfully.

### Held-out theorem evaluation

None of the 5 v5 priority-template held-out wins (`Nat.div_lt_one_iff`,
`Nat.div_pos`, `Nat.div_pos_iff`, `Nat.mul_eq_left`,
`Nat.mul_eq_right`) were proved by raw `gen_v5_plus1`. Expected:
the held-out set was *removed* from training data precisely so the
v5+1 evaluation could honestly test for generalization. With only
143 training pairs and a 60M-param T5-small base, the model didn't
generalize beyond the patterns it saw.

### Generalization read

The +1 lift comes from one in-training-set theorem. The model
learned a specific (state, tactic) mapping but did not yet
generalize the `iff ↔ omega ∧ omega` pattern to siblings of the
training theorem. Stronger generalization would require either:

  - more diverse training pairs (the current 143 are heavily biased
    toward `iff`-shape exact-tuple tactics — 46 of the 143 are
    tactic_template),
  - more epochs at higher lr (but at the cost of forgetting v5),
  - a larger base model with stronger inductive priors.

## Comparison table

| policy | medium | large | wallclock |
|---|---:|---:|---:|
| raw `gen_v5` (baseline) | 3/38 | not run | ~2.5 min |
| raw `gen_v5_plus1` | **4/38** (+1) | not run | ~2.5 min |
| NS9 wrapper + `gen_v5` | **37/38** | 49/65 | ~2.5 / ~5 min |
| NS9 wrapper + `gen_v5_plus1` | not run | not run | — |

The wrapper+gen_v5_plus1 case was not evaluated; with the current
wrapper at 37/38 the only headroom on medium is `Nat.AM_GM`, which
is *held out* from training and not addressed by the +1 lift.

## What this answers

1. **Does raw gen_v5_plus1 improve over raw gen_v5?** Yes, modestly
   (+1 theorem on medium).
2. **Does it natively produce any wrapper-discovered tactics?** Yes
   — the `pt_iff_8` template tactic for `Nat.add_eq_right`.
3. **Does it close any held-out v5 priority-template wins?** No.
   Training set is too small + held-out by design.
4. **Does it regress on anything?** No regressions observed on
   medium.
5. **Does wrapper + gen_v5_plus1 improve over wrapper + gen_v5?**
   Not evaluated; wrapper already at 37/38, room is only `Nat.AM_GM`.

## Limitations and recommendations for NS11

1. **Dataset size.** 143 unique (state, tactic) pairs is small for
   a 60M-param model. Strategies to grow it:
   - Drop the held-out exclusion temporarily for *coverage*
     training; use a separate K-fold style held-out for generalization
     measurement.
   - Include `retrieved_premise` rows; pair them with retrieval
     context so the model learns to emit retrieved-style tactics.
   - Augment with mid-proof states from advance chains rather than
     just close steps.

2. **Larger or domain-pretrained base.** T5-small struggles to
   generalize from 143 pairs. CodeT5-small or a Lean-pretrained
   t5-base would likely show stronger transfer.

3. **Joint training with original v5 dataset.** The combined
   5720-pair dataset is built (`project/data/ns10_combined_train.jsonl`)
   but not yet used. A 3-epoch fine-tune on the combined set would
   test whether NS10 traces *add value* on top of v5 — preventing
   catastrophic forgetting of v5's existing 3-theorem coverage.

4. **Wrapper + gen_v5_plus1 evaluation.** Quick follow-up to verify
   no regression on the wrapper's 37/38.

5. **Curriculum.** The first targeted re-train should focus on a
   single failure class (e.g., the `div` family theorems
   `Nat.div_lt_one_iff`, `Nat.div_pos`, ...) using their wrapper
   traces — testing whether 5-10 examples per theorem is enough
   for a focused Learn step.

## Conclusion

NS10 demonstrates the Learn step works in principle: the
trace-to-training pipeline closes the loop. A 143-pair fine-tune
on top of `gen_v5` produces a model that closes one previously-
unprovable theorem natively, with zero regressions. The
infrastructure is now in place to scale this — bigger datasets,
more epochs, more focused curricula, or a larger base model are
the next levers.

The **AlphaEvolve loop closure is complete in spirit**: Search
(skeleton-evolution NS5-NS9) → Evaluate (Lean) → Train (NS10) →
back to Search. The lift is small at this dataset size; the next
iteration should focus on dataset growth and a stronger base model
before claiming a meaningful production effect.

## Files

Committed:
  - `scripts/build_ns10_training_data.py` — dataset builder
  - `project/data/ns10_evolve_train_meta.json` — dataset metadata
  - `project/evolve/reports/ns10_learn_step_report.md` — this report
  - `.gitignore` — adds `project/models/gen_v5_plus1/`,
    `project/models/gen_v5_ns10_smoke/`,
    `project/data/ns10_*.jsonl`,
    `project/evolve/eval_runs/`,
    `project/evolve/training_runs/`

Not committed (gitignored):
  - `project/data/ns10_evolve_train.jsonl` (regeneratable from traces)
  - `project/data/ns10_combined_train.jsonl` (regeneratable)
  - `project/models/gen_v5_plus1/` (~242 MB checkpoint)
  - `project/models/gen_v5_ns10_smoke/` (~242 MB throwaway)
  - `project/evolve/eval_runs/` (raw eval artifacts)
  - `project/models/gen_v5_plus1_training.log`
