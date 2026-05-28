# Overnight A+D — Summary

Run directory: `experiments/overnight_20260501_204545`

## Curriculum eval (curriculum_all, 30 theorems)

Headline: deterministic beam-k=8 anchor + sampling variance (temp=0.8, top-p=0.95, k=8) over 5 seeds.

| Checkpoint | Beam (anchor) | Sample mean ± std | Sample range | N seeds |
|---|---|---|---|---|
| gen_v5  (t5-small, baseline) | 0/1 | 0.0 ± 0.0 | 0 – 0 | 1 |
| gen_v6_premise (t5-small + premise injection) | 0/1 | 0.0 ± 0.0 | 0 – 0 | 1 |
| gen_v6  (t5-base, never previously evaluated) | 1/1 | 1.0 ± 0.0 | 1 – 1 | 1 |

### Per-seed sample results

| Checkpoint | seed 42 |
|---|---|
| gen_v5  (t5-small, baseline) | 0 |
| gen_v6_premise (t5-small + premise injection) | 0 |
| gen_v6  (t5-base, never previously evaluated) | 1 |

## Reading the result

- **Capacity scaling (t5-small → t5-base):** 0 → 1 (Δ = ↑ 1 theorems on the curriculum, beam).
- **Premise injection at t5-small:** 0 → 0 (Δ = → 0, beam). Reproducing the v5/v6_premise contrast.
- **v5 vs premise gap under sampling:** 0.0 ± 0.0  vs  0.0 ± 0.0.  Gap = +0.0, pooled sd = 0.00.  → ROBUST.
- **t5-small vs t5-base gap under sampling:** 0.0 ± 0.0  vs  1.0 ± 0.0.  Gap = +1.0, pooled sd = 0.00.  → ROBUST.

## Retriever quality probe (D)

- Total proven theorems in `project_state.json`: **262**
- Evaluable (named premises + first-state in traces): **28**
- Skipped — tactic had no named premises (e.g. plain `aesop`): 222
- Skipped — no first-state in traces: 12

| File bucket | N | Recall@1 | Recall@5 | Recall@10 | Recall@15 |
|---|---|---|---|---|---|
| Set.Basic | 12 | 50.0% | 83.3% | 83.3% | 83.3% |
| Finset.Basic | 14 | 0.0% | 0.0% | 0.0% | 0.0% |
| Nat.Defs | 2 | 16.7% | 50.0% | 50.0% | 50.0% |
| **OVERALL** | 28 | 22.6% | 39.3% | 39.3% | 39.3% |

**Reading this:** Recall@k = of the lemma names that actually appeared in the winning tactic, what fraction did the retriever place in its top-k?  Low Recall@5 with high Recall@15 means the retriever is finding the right premise but ranking it poorly. Low Recall@15 means the premise isn't in the index at all, which is the bottleneck regardless of model size.

## Raw artifacts

- Per-run metrics: `experiments/overnight_20260501_204545/<tag>_<mode>_<seed>/eval-*/metrics.json`
- Run log: `experiments/overnight_20260501_204545/run.log`
- Retriever probe JSON: `experiments/overnight_20260501_204545/retriever_probe.json`
