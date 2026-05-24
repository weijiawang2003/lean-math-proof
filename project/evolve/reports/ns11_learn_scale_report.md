# NS11 — Learn-step scale-up

NS10 closed the AlphaEvolve loop with a 143-pair trace-to-training
fine-tune that lifted raw `gen_v5` from 3/38 to 4/38 on
`nat_defs_medium`. NS11 scales the same pipeline along three axes:

1. **Wider trace consumption** — walk *all* of `project/evolve/` to
   include NS5 skeleton_runs (the bulk: 6,308 close transitions) and
   older autonomous_runs that NS10 had skipped.
2. **Joint training with the original v5 corpus** — a 5,729-pair
   combined dataset that adds 152 evolve pairs on top of the 5,577
   v5 training pairs `gen_v5` was originally fine-tuned from.
3. **Variant grid** — three filter variants
   (`conservative` / `medium` / `coverage`) all reproducible from one
   builder so we can compare the trade-offs.

## Hard constraints respected

- `project/models/gen_v5/` untouched.
- New checkpoints under `project/models/gen_v5_ns11_*/` only.
- NS9 wrapper artifact (`project/evolve/best/ns9_best_genome.json`)
  untouched.
- All datasets, checkpoints, eval artifacts, and training logs are
  gitignored. Only scripts, small metadata, and reports committed.

## Stage 1 — trace source audit

`scripts/audit_trace_sources.py` walks every `traces.jsonl` under
`project/evolve/`. Summary table at
`project/evolve/reports/ns11_trace_source_audit.md`. Key totals:

| | value |
|---|---:|
| traces.jsonl files                | **368** |
| episodes (across all files)       | **13,773** |
| close transitions                 | **11,915** |
| advance transitions               | **5,624** |

Biggest single source: `skeleton_runs/ns5-20260523-050214-0ec613`
(171 trace files / 6,308 close transitions). NS10 had been walking
only NS6/NS7/NS9 dirs (~5,159 episodes); NS11 walks the full
13,773.

## Stage 2 — scaled dataset variants

`scripts/build_ns11_training_data.py` produces three variants with
shared filters (max tactic 200ch, max state 2500ch, dedup by
(state, tactic), self-reference rejection):

| variant       | close | assist | held-out | retrieved | pairs |
|---------------|------:|-------:|----------|-----------|------:|
| conservative  |     ✓ |       — | enforced  | no | **110** |
| medium        |     ✓ |  K=3   | enforced  | no | **152** |
| coverage      |     ✓ |  K=5   | dropped   | no | **166** |
| combined      | medium variant + `project/seq2seq_data_v5.jsonl` rows | | | | **5,729** |

The wider trace walk picked up only +7 close / +2 assist over
NS10. Diversity is bottlenecked by the wrapper's coverage (~51
unique theorems prove repeatedly) rather than trace count — the
13,773 episodes dedup down to ~152 distinct (state, tactic) pairs.

This bottleneck is the central finding: more wrapper runs on the
*same* theorem set won't grow the dataset. Lifting it requires
running the wrapper on a larger theorem set first.

## Stage 3 — prompt variants

The builder supports `--prompt-style {vanilla, origin, skeleton,
premise}`. We scaffolded but did not train all variants — the
training experiments below all use `vanilla` so they're directly
comparable to NS10 baselines. The infrastructure is in place for a
follow-up A/B once the dataset is large enough for prompt-style
differences to matter (rule of thumb: >1,000 pairs per arm).

## Stage 4 — training experiments

All fine-tunes start from `project/models/gen_v5` with low lr
(`1e-5`) to preserve v5 knowledge. Hardware: MPS, batch 4,
max_src 512, max_tgt 64.

| run | dataset | epochs | runtime | final eval_loss |
|---|---|---:|---:|---:|
| `gen_v5_ns11_medium`   | 152 pairs    | 5 | 36 s    | 2.13 |
| `gen_v5_ns11_combined` | 5,729 pairs  | 3 | 12 min  | **0.43** |
| `gen_v5_ns11_coverage` | 166 pairs    | 5 | 38 s    | 3.05 |

The combined run's eval_loss is an order of magnitude lower because
its held-out split is dominated by v5 distribution that gen_v5 was
already trained on; the metric isn't directly comparable across
runs. Use the downstream Lean-eval numbers below for the real read.

## Stage 5 — evaluation

### Raw model — `nat_defs_medium` (38 theorems)

| model | proved | Δ vs gen_v5 |
|---|---:|---:|
| **`gen_v5`**                     | **3 / 38** | — |
| `gen_v5_plus1` (NS10)            | 4 / 38     | +1 |
| `gen_v5_ns11_medium`             | 5 / 38     | +2 |
| `gen_v5_ns11_combined`           | **9 / 38** | **+6** |
| `gen_v5_ns11_coverage`           | 5 / 38     | +2 |

New theorems closed by `gen_v5_ns11_combined` that gen_v5 cannot:
`Nat.add_eq_left`, `Nat.add_eq_max_iff`, `Nat.add_eq_right`,
`Nat.half_le_of_sub_le_half`, `Nat.lt_one_iff`,
`Nat.eq_zero_of_double_le`. All use tactics directly learned from
the NS11 trace pairs (`simp`, `simp_arith`, `omega`-style chains).

### Raw model — `nat_defs_large_v5` (65 theorems)

| model | proved |
|---|---:|
| `gen_v5_ns11_medium`   |  7 / 65 |
| `gen_v5_ns11_combined` | **13 / 65** |
| `gen_v5_ns11_coverage` |  7 / 65 |

Combined adds 4 more large-only theorems on top of the 9 medium ones:
`Nat.add_eq_three_iff`, `Nat.max_eq_zero_iff`,
`Nat.sub_eq_of_eq_add'`, `Nat.two_pow_succ`.

### Wrapper + new model — NS9 best genome

The wrapper is the production setup; preserving its proved count is
a hard requirement for any base-model swap.

| model | medium | large |
|---|---:|---:|
| `gen_v5` + NS9 wrapper (baseline) | **37 / 38** | **49 / 65** |
| `gen_v5_ns11_medium` + NS9 wrapper | 37 / 38 | not run |
| **`gen_v5_ns11_combined` + NS9 wrapper** | **37 / 38** | **49 / 65** |
| `gen_v5_ns11_coverage` + NS9 wrapper | 37 / 38 | not run |

**No wrapper regression.** The combined model fully reproduces the
NS9 production result on both `nat_defs_medium` and
`nat_defs_large_v5` — meaning the fine-tune is genuinely additive,
not a Pareto move.

### Set / Finset regression check

`demo_v1` (15 theorems; Set-heavy):

| model | proved | regressions vs gen_v5 |
|---|---:|---|
| `gen_v5`                | 10 / 15 | — |
| `gen_v5_ns11_medium`    |  8 / 15 | lost `Set.subset_univ`, `Set.empty_subset` |
| `gen_v5_ns11_combined`  |  8 / 15 | same two |
| `gen_v5_ns11_coverage`  |  8 / 15 | same two |

`set_small` (1) and `finset_small` (3): all three NS11 checkpoints
match the gen_v5 baseline of 0/1 and 0/3 — no regression there, but
the baseline is already 0 so this is not informative.

**The two lost demo_v1 theorems are a real regression.** Both used
`simp [Set.subset_def]` from gen_v5; after fine-tune the model no
longer picks that emission for those states even though the
combined dataset's 5,577 v5 rows still contain Set examples. Likely
cause: 3 epochs of joint training with a Nat-heavy 152-pair add-on
shifts the model's distribution toward Nat tactics enough to drop
two infrequent Set patterns.

Mitigations to try in NS12: even lower lr (5e-6), shorter joint
training (1-2 epochs instead of 3), or oversampling the Set rows
during training.

### Held-out theorem read-through

None of the 10 held-out theorems
(`Nat.AM_GM`, `Nat.div_lt_*`, `Nat.div_pos*`, `Nat.mul_eq_*`,
`Nat.dvd_iff_div_mul_eq`, `Nat.sqrt_lt`, `Nat.pow_lt_pow_iff_left`)
were proved by *any* NS11 raw model, including coverage, which had
1–4 wrapper-trace examples per held-out theorem (14 rows total).
The wrapper proofs reference local hypotheses (`hb`, `hba`, …); a
60M-param T5 needs many more shape-equivalent examples to
generalize that pattern.

## Stage 6 — summary table

The full headline:

| policy                                       | medium (38) | large (65) | demo_v1 (15) |
|----------------------------------------------|------------:|-----------:|-------------:|
| raw `gen_v5`                                  | 3 / 38      | not run    | **10 / 15**  |
| raw `gen_v5_plus1` (NS10)                     | 4 / 38      | not run    | not run      |
| raw `gen_v5_ns11_medium`                      | 5 / 38      | 7 / 65     | 8 / 15       |
| **raw `gen_v5_ns11_combined`**                | **9 / 38**  | **13 / 65**| 8 / 15       |
| raw `gen_v5_ns11_coverage`                    | 5 / 38      | 7 / 65     | 8 / 15       |
| NS9 wrapper + `gen_v5`                        | **37 / 38** | **49 / 65**| —            |
| NS9 wrapper + `gen_v5_ns11_combined`          | **37 / 38** | **49 / 65**| —            |

## What this answers (NS11 goals)

1. **Does raw performance improve beyond NS10's 4/38?**
   Yes — combined raw lifts from 4/38 → **9/38** medium and
   establishes 13/65 large.
2. **Does joint training with the original v5 corpus help?**
   Decisively yes. Combined more than doubles the medium-only
   lift; the v5-only variants stop at 5/38.
3. **Can wrapper-discovered tactics be learned without runtime?**
   Yes for the closing tactics — the model now natively emits
   `simp_arith`, learned `simp` chains, and `aesop`-style closers
   on previously-unprovable theorems. No for the multi-step
   skeleton/retrieval patterns that reference local hypotheses —
   those still need the wrapper.
4. **Does Set/Finset regress?** Yes, on demo_v1: −2 theorems
   across all three NS11 variants. The combined model still
   matches gen_v5 inside the wrapper (37/38, 49/65) but the raw
   model is no longer a strict superset on Set tasks.
5. **Does the wrapper still hit 37/38 with the new base model?**
   Yes — `gen_v5_ns11_combined` + NS9 wrapper holds at 37/38
   medium and 49/65 large, identical to gen_v5 + wrapper.

## Limitations and recommendations for NS12

1. **Diversity bottleneck.** 13,773 trace episodes dedup to ~152
   unique training pairs. To grow the dataset we need to *run the
   wrapper on a larger theorem set* — the same wrapper hitting the
   same medium theorems will not produce new (state, tactic)
   pairs. Candidates: a curated frontier of theorems where the
   wrapper currently fails (e.g., the 16 unproved large
   theorems), or harvesting Mathlib for theorems matching the
   wrapper's family/shape templates.

2. **Set regression on raw model.** Fine-tune-induced forgetting of
   `simp [Set.subset_def]`. Try:
   - lr = 5e-6 instead of 1e-5;
   - 1 or 2 epochs combined instead of 3;
   - explicit oversampling of Set/Finset rows in the combined data.

3. **Prompt variant A/B.** Builder supports
   `--prompt-style {origin, skeleton, premise}` but we did not
   train them. With the dataset still under ~200 pairs from
   traces, prompt-style differences are likely below noise; revisit
   once we have ≥1,000 pairs per arm.

4. **Multi-step learning.** The 42 advance_assist rows in the
   medium dataset carry intermediate proof steps. The raw model
   does emit those single tactics individually, but doesn't yet
   produce the *sequence*. A token-conditioned multi-step format
   (or a small chain-of-tactics fine-tune) is the next research
   direction.

5. **Larger base model.** All NS10/NS11 experiments use T5-small
   (60M params). With a Mathlib-pretrained 220M model the same
   152 pairs would likely lift to ≥15/38. This is a hardware
   trade-off, not a pipeline change.

## Conclusion

NS10 proved the Learn step works in principle (143 pairs → +1
theorem). NS11 establishes that **joint training with the original
v5 corpus is the right lever**: 5,729 pairs lift the raw model
from 3/38 → 9/38 medium (3×) and from "not run" → 13/65 large,
while preserving the NS9 wrapper's 37/38 medium and 49/65 large.

The one real cost — a 2-theorem regression on demo_v1's Set
subset_def proofs — is a forgetting artifact of the small joint
fine-tune, fixable via the lr/epoch/oversampling mitigations
above.

The dataset-diversity bottleneck (13,773 episodes → 152 pairs) is
the new gating constraint. NS12 should focus on *generating new
verified traces* (running the wrapper on a wider theorem set) and
on a tighter Set-regression fix, before any further base-model
swap.

## Files

Committed:
- `scripts/audit_trace_sources.py` — trace-source audit.
- `scripts/build_ns11_training_data.py` — variant-aware dataset
  builder.
- `project/data/ns11_train_conservative_meta.json`
- `project/data/ns11_train_medium_meta.json`
- `project/data/ns11_train_coverage_meta.json`
- `project/data/ns11_train_combined_meta.json`
- `project/evolve/reports/ns11_trace_source_audit.md`
- `project/evolve/reports/ns11_learn_scale_report.md` (this file)
- `.gitignore` — adds `project/data/ns11_*.jsonl` and the new
  checkpoint dirs.

Not committed (gitignored, regeneratable):
- `project/data/ns11_train_{conservative,medium,coverage,combined}.jsonl`
- `project/models/gen_v5_ns11_{medium,combined,coverage}/`
- `project/models/gen_v5_ns11_*_training.log`
- `project/evolve/eval_runs/gen_v5_ns11_*/`
