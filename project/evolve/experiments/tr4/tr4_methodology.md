# TR4 — Retrieval-Aware Proof-Action Ranker at Program Level

## Purpose

TR3 ran **4,377 candidate programs** across 92 confirmed RC2 failures to find **13
verified successes** (12 attributed TRUE_DELTA). That is ~0.3 % positive rate — almost
all live Lean work was wasted. TR4 trains a **ranker**, not a prover:

> score(theorem, retrieved lemma, proof program) → P(verified success)

The goal is to **order candidate programs** so a future TR3-style search runs the
likely-winning probes first and stops early — cutting probe budget without losing
successes. TR4 does **not** generate proofs, does **not** touch production routing, and
does **not** promote any model.

## Unit of learning

One **(theorem, program)** pair — e.g. theorem `Set.disjoint_right`, program
`simp [Set.disjoint_left] <;> aesop`, outcome `success`. Each pair carries the
retrieved-lemma context (rank, score), the program family/depth, and symbolic goal
flags.

## Labels

- **label_success = 1** iff the program closed the theorem live (any source).
- **label_credit = 1** iff additionally: literal RC2 failed, the success is attributed
  `TRUE_*_DELTA` / `TRUE_DEF_UNFOLD_SIMP_WIN`, and it is not BASELINE_DUPLICATE /
  PRODUCTION_SUBSUMED / SOURCE_SPECIFIC.

Both labels are kept. The ranker primarily optimizes **label_success** (the live-search
objective) and **label_credit recovery is reported separately** (the
beats-literal-RC2 objective).

## Data sources

- **TR3** depth-program results (primary; ~3,926 run programs, 13 successes).
- **SF5** retrieval probes (adds the definitional-unfold / pair positives).
- **RC4A** gated def_unfold candidate probes (validated positives).

Rows are tagged by `source`; the per-theorem **ranking / budget** evaluation uses the
TR3 program lists (the realistic search-ordering scenario), while SF5/RC4A rows enrich
the positive class for training.

## Evaluation (not ordinary accuracy)

Extreme class imbalance (~0.3–0.5 % positive) makes raw accuracy meaningless. TR4 is
judged on:

- **Ranking:** per-theorem top-1/3/5 success recovery, MRR, NDCG, positive-rank
  percentile — does the ranker put successful programs near the top?
- **Probe budget:** % of programs needed to recover 25/50/75/100 % of successes;
  successes recovered by the top 5/10/20 % — vs random and vs the TR3 original order
  and a heuristic baseline.
- **Generalization:** grouped splits — leave-one-theorem-out, leave-one-cluster-out,
  leave-one-namespace-out, and a source/time split (train SF5+TR3 → test RC4A).
- **Classification (secondary):** PR-AUC / ROC-AUC, precision@k, recall@k.

Leakage controls: grouped-by-theorem CV (a theorem's programs never split across
train/test); an ablation (Part 9) checks whether the model uses retrieval/program
*interactions* or just memorizes family/name cues.

## Decision space

`RANKER_USEFUL_FOR_PROBE_REDUCTION` (≥70 % successes in ≤20 % probes, beats
heuristic) / `RANKER_SIGNAL_FOUND_NEEDS_MORE_POSITIVES` /
`HEURISTIC_BASELINE_SUFFICIENT` / `DATA_TOO_IMBALANCED` / `REJECT_NO_SIGNAL`.

## Guardrails

Protected configs (RC1/RC2 wrappers, ns24, NS9, REL/RC reports, TR1/TR2/SF/TR source
datasets) are read-only. No production change, no RC4 release, no README update, no
model promotion, no commit. All artifacts under `project/evolve/experiments/tr4/` &
`project/evolve/reports/tr4/`, scripts `scripts/tr4_*.py`. No live Lean is run in TR4 —
it consumes existing verified outcomes only.
