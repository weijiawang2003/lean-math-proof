# Overnight v9 Sweep — Summary

Run dir: `experiments/overnight_v9_20260504_143641`

## Phase 1 — search_v6 on frontier_v1

- Transitions logged: 3914
- Seq2seq pairs after filtering: 128
- After 50x upsampling: 6400

## Phase 4 — Beam evals on curriculum_all

| Checkpoint | Score | Notes |
|---|---|---|
| gen_v5 (anchor)   | 25/30 | t5-small, v5 data — historical SOTA |
| gen_v7 (anchor)   | 24/30 | t5-base, v5 data — capacity isolation |
| gen_v9_ft        | 20/30 | regression vs gen_v7 |
| gen_v9_full        | 26/30 | lifts above gen_v7 — partial success |

## Phase 5 — Variance bars (sampling, temp=0.8, top-p=0.95, k=8)

| Checkpoint | Mean ± std | Range | N seeds |
|---|---|---|---|
| gen_v5 | 20.8 ± 2.9 | 17-25 | 5 |
| gen_v7 | 15.8 ± 1.5 | 14-18 | 5 |
| gen_v9_ft | 11.2 ± 0.4 | 11-12 | 5 |
| gen_v9_full | 19.2 ± 2.6 | 16-22 | 5 |

## Hypotheses tested

**H1 (fine-tune avoids dilution):** Fine-tuning gen_v7 on frontier-only
data should preserve gen_v7's curriculum solves while adding the new
ones. → check gen_v9_ft beam score: ≥25/30 supports H1.

**H2 (50x upsampling crosses dilution floor):** Full retrain with
frontier upsampled to ~9% of pool should override v5 priors enough
to keep frontier proofs in beam. → check gen_v9_full beam score:
≥27/30 supports H2.

**H3 (variance bars):** The 24 vs 25 gap between gen_v7 and gen_v5
should be small under sampling noise.  If it's swamped by std,
the entire capacity-isolation finding needs caveating.

## What to do next (decision tree)

- **gen_v9_ft ≥ 27/30:** fine-tuning is the right pattern;
  iterate the loop — search next batch of frontier theorems and fine-tune again.
- **gen_v9_full ≥ 27/30:** upsampling-with-full-retrain works;
  more expensive but cleanly comparable to v7.
- **Both at 24-25:** dilution mechanism is even more robust than thought;
  next move is the strategic-policy ablation, not more data manipulation.
- **Net regression:** something specific went wrong in one of the runs;
  inspect run.log + per-theorem diffs.

Per-theorem disposition matrix and idiom-frequency comparison should be
done as a morning analysis pass via Claude Code.
