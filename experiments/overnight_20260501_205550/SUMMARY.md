# Overnight A+D — Summary

Run directory: `experiments/overnight_20260501_205550`

## Curriculum eval (curriculum_all, 30 theorems)

Headline: deterministic beam-k=8 anchor + sampling variance (temp=0.8, top-p=0.95, k=8) over 5 seeds.

| Checkpoint | Beam (anchor) | Sample mean ± std | Sample range | N seeds |
|---|---|---|---|---|
| gen_v5  (t5-small, baseline) | 25/30 | 18.5 ± 2.1 | 17 – 20 | 2 |
| gen_v6_premise (t5-small + premise injection) | 19/30 | — | — | 0 |
| gen_v6  (t5-base, never previously evaluated) | 22/30 | — | — | 0 |

### Per-seed sample results

| Checkpoint | seed 42 | seed 456 |
|---|---|---|
| gen_v5  (t5-small, baseline) | 17 | 20 |
| gen_v6_premise (t5-small + premise injection) | — | — |
| gen_v6  (t5-base, never previously evaluated) | — | — |

## Reading the result

- **Capacity scaling (t5-small → t5-base):** 25 → 22 (Δ = ↓ 3 theorems on the curriculum, beam).
- **Premise injection at t5-small:** 25 → 19 (Δ = ↓ 6, beam). Reproducing the v5/v6_premise contrast.

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

## Run failures (these did NOT produce metrics)

- `base_sample_seed1024 (no metrics.json — likely crashed)`
- `base_sample_seed123 (no metrics.json — likely crashed)`
- `base_sample_seed42 (no metrics.json — likely crashed)`
- `base_sample_seed456 (no metrics.json — likely crashed)`
- `base_sample_seed789 (no metrics.json — likely crashed)`
- `premise_sample_seed1024 (no metrics.json — likely crashed)`
- `premise_sample_seed123 (no metrics.json — likely crashed)`
- `premise_sample_seed42 (no metrics.json — likely crashed)`
- `premise_sample_seed456 (no metrics.json — likely crashed)`
- `premise_sample_seed789 (no metrics.json — likely crashed)`
- `v5_sample_seed1024 (no metrics.json — likely crashed)`
- `v5_sample_seed123 (no metrics.json — likely crashed)`
- `v5_sample_seed789 (no metrics.json — likely crashed)`

## Raw artifacts

- Per-run metrics: `experiments/overnight_20260501_205550/<tag>_<mode>_<seed>/eval-*/metrics.json`
- Run log: `experiments/overnight_20260501_205550/run.log`
- Retriever probe JSON: `experiments/overnight_20260501_205550/retriever_probe.json`
