# Morning Analysis Brief — for Claude Code

> You are Claude Code, running in `~/dev/dojo_sandbox`. Last night a bash
> orchestrator (`experiments/overnight.sh`) ran 18 rollout evaluations plus a
> retriever probe. Your job is to turn those raw metrics into a focused
> analysis report. Read this whole brief before doing anything.

## Context you need

This is a Lean-supervised tactic-learning project. Three checkpoints were
evaluated last night on `curriculum_all` (30 theorems):

- **gen_v5** — T5-small baseline. Prior known result: 25/30 on beam-k=8.
- **gen_ckpt_v6_premise** — T5-small + retrieved-premise prefix. Prior
  known result: 19/30 on beam-k=8 (the project's "negative result").
- **gen_v6** — **T5-base, never previously evaluated.** Last night was its
  first eval. The headline curriculum number for t5-base is now in the run
  dir, and you are the first to read it.

For each checkpoint, two decoding modes were run:

- One deterministic `beam` run at k=8 — the reproducibility anchor.
- Five `sample` runs (temperature 0.8, top-p 0.95, seeds 42/123/456/789/1024)
  — gives variance bars on the gap between checkpoints.

A retriever-quality probe was also run, computing recall@k of the premise
retriever against ground-truth premises extracted from verified proofs in
`project/project_state.json`, bucketed by mathlib file.

## Where to look

```
experiments/overnight_<TIMESTAMP>/
  SUMMARY.md                          ← read this first
  run.log                             ← full bash log; check for [FAIL] lines
  retriever_probe.json                ← per-theorem retriever results
  v5_beam/eval-*/{config,metrics,traces}.json
  v5_sample_seed42/eval-*/{...}.json
  ... (18 total run directories)
```

The `metrics.json` per run has a `per_theorem` field with `finished`,
`tactics_used`, `winning_tactic`, `error_message`, etc. for every theorem.
The `traces.jsonl` in the same dir has the raw step-by-step transitions.

There may be exactly one prior `experiments/overnight_*` dir from a recent
smoke test — pick the most recent one with a complete `SUMMARY.md`.

## What to produce

Write `experiments/overnight_<TIMESTAMP>/ANALYSIS.md` containing the four
sections below. Total target length 600–1200 words. Be concrete, not
ceremonial. Cite specific theorem names and tactic strings. Avoid
restating SUMMARY.md numbers — link or reference them and add the analysis
SUMMARY.md cannot do.

### 1. Headline read

In one paragraph: does the t5-base eval move the needle relative to t5-small?
Is the v5 vs premise gap robust under sampling noise? Is anything
unexpected? If the beam anchor numbers do *not* reproduce 25/30 and 19/30,
flag that loudly — it would mean something drifted (torch version, model
file, or seed leakage).

### 2. Per-theorem disposition matrix

For all 30 curriculum theorems, build a table:

| Theorem | v5 beam | premise beam | base beam | base sample mean (5 seeds) |
|---|---|---|---|---|

Then identify and discuss:

- Theorems v5 solved that t5-base did NOT — these are scaling regressions
  and are interesting (capacity should not hurt).
- Theorems t5-base solved that v5 did NOT — direct evidence of capacity
  paying off.
- Theorems all three checkpoints failed on — the consistent frontier within
  the curriculum.
- Theorems where premise injection helped (rare per the prior result) vs.
  hurt (the typical case): list the regressions explicitly with the
  winning tactic of each.

### 3. Retriever probe deep-dive

The probe showed Finset.Basic at 0% recall@15 across 14 evaluable theorems
(prior smoke test). That likely still holds — confirm or update from the
actual `retriever_probe.json`. Then:

- For each Finset.Basic theorem in `per_theorem.Finset.Basic`, look at
  `winning_tactic` and `ground_truth_premises`. What lemma names are these
  proofs using? Are they in `STATIC_PREMISES["Finset"]` in
  `premise_retriever.py`? If not, that's the gap.
- One concrete fix proposal: which 5–10 lemma names should be added to the
  Finset static catalog to lift recall? List them.

Keep this section pragmatic — the goal is a fixable diagnosis, not a
treatise.

### 4. What to run next

Given everything above, three concrete experiment proposals, in priority
order. Each should be (a) a one-paragraph hypothesis, (b) the actual
command to run it, (c) the expected runtime on this Mac.

Candidates worth considering — pick whichever the data supports:

- Train `t5-base + premise-augmented` (the missing crossover-plot point) —
  this is the experiment that would publishably resolve the capacity-
  threshold question.
- Run the `strategic_policy` ablation (strategic-on vs strategic-off at
  gen_v5 and at t5-base) — never quantified.
- Eval the t5-base checkpoint on `curriculum_tier3` and on a Nat.Defs
  sample — does the capacity gain transfer beyond the curriculum?
- Fix the Finset retriever gap and re-run premise-augmented eval — does
  the negative result soften when retrieval is better?

## Constraints (read carefully)

You may:
- Read any file in the repo.
- Run analysis subprocesses: `python -c "..."` short scripts.
- Write **new** files under `experiments/` and `experiments/overnight_*/`.

You should NOT:
- Edit `eval_rollout_all.py`, `generative_policy.py`, `policy.py`,
  `model_rollout.py`, `env.py`, `core_types.py`, `tasks.py`, or any
  `train_*.py` script. These are canonical and were just modified last
  night.
- Re-train any model.
- Run any rollout that takes more than ~5 minutes (no big new evals;
  this is analysis).
- Write to `project/project_state.json` or `project/all_traces.jsonl`.
- Edit `MEMORY.md` or any file under `.claude/`.

If you find a bug in one of the canonical scripts during analysis, write
the bug report to `experiments/overnight_<TIMESTAMP>/BUGS.md` instead of
fixing it yourself.

## Style

- Concrete > ceremonial. Cite theorem names and tactic strings.
- Don't restate things SUMMARY.md already says.
- If a result surprises you, say so and propose a check.
- If the data is noisy or inconclusive, say that — don't manufacture a
  conclusion to fill the section.
- Skip emojis, skip thanks-paragraphs, skip "as a language model" framing.

When you finish, print the path to ANALYSIS.md and a 3-line summary of the
top finding. That's it.
