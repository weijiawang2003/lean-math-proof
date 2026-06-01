# TR4 — Retrieval-Aware Proof-Action Ranker at Program Level

**Decision: `RANKER_USEFUL_FOR_PROBE_REDUCTION`** (within seen namespaces; cross-namespace
transfer NOT established — see §9). Exploratory; **not promoted**, no production change.

---

## 1. Executive summary

- **Dataset:** 4,737 (theorem, program) rows from TR3 (3,926) + SF5 (800) + RC4A (11).
  **23 success positives / 22 credit positives** (~0.5 % positive rate).
- **Best model:** `HistGradientBoosting` (HGB). Leakage-free OOF (GroupKFold by
  theorem) **PR-AUC 0.52** (vs ~0.005 base rate), ROC-AUC 0.77, **precision@10 0.90**.
  Logistic 0.10, heuristic 0.017, SGD 0.014.
- **Per-theorem ranking** (13 TR3 theorems with a success): HGB recovers **all 13 in
  top-5** (mean first-success rank **2.08** vs TR3 original order **16.46**, random 8.73).
- **Probe-budget:** running **top-5 programs/theorem** by HGB recovers **13/13
  successes (100 %) and 12/12 credited** while running **439 of 4,377 programs =
  88.8 % probe reduction**. Beats the heuristic (top-20 %: 0.92 vs 0.85 success
  recovery).
- **Ablation:** theorem-name-only PR-AUC 0.017 (≈ base rate → **not** name
  memorization); signal lives in program-tactic (0.39) + symbolic/interaction (0.30)
  features, and all-features adds **+0.13** interaction gain.
- **Decision:** `RANKER_USEFUL_FOR_PROBE_REDUCTION`.

---

## 2. Motivation

TR3 ran 4,377 live programs to find 13 successes — ~0.3 % yield; almost all LeanDojo
work was wasted. A ranker that orders candidate programs by P(success) lets a future
TR3-style search run the likely winners first and stop early. TR4 builds that ranker
from existing verified outcomes (no live Lean).

## 3. Dataset

Sources: TR3 depth-program results (primary), SF5 retrieval probes, RC4A gated
candidate probes. **label_success** = program closed the goal live; **label_credit** =
success AND literal-RC2 failed AND attributed TRUE_*_DELTA / TRUE_DEF_UNFOLD_SIMP_WIN
(not baseline-dup / subsumed / source-specific). 4,737 rows, 23 success / 22 credit
positives. Successes by family: def_unfold_simp 14, d1_simp_lemma 4, d2_simp_aesop 3,
d1_aesop 1, aesop_add_simp 1. Strong imbalance, positives concentrated in Set/Finset/
List (Nat has **0** positives). Artifacts: `data/tr4_program_examples.jsonl`,
`data/tr4_dataset_summary.*`.

## 4. Features

4,825 sparse features: theorem-name char/token n-grams, goal TF-IDF (word), lemma-name
n-grams, program-tactic tokens, family/depth one-hot, retrieval rank-bucket + score,
symbolic flags (set/finset/list/nat/iff/subset/disjoint/compl/singleton/card/tofinset),
and interactions (ns-match × family, rank-bucket × family, uses-retrieved × family).
Artifacts: `data/tr4_features.npz`, `data/tr4_feature_metadata.json`,
`data/tr4_vectorizers.joblib`.

## 5. Models

heuristic (rule: winning-family + ns-match + low retrieval-rank + uses-retrieved,
depth penalty), LogisticRegression(balanced), SGDClassifier(log_loss, balanced),
HistGradientBoosting. Full-data models + a leakage-free OOF score array saved for
ranking/budget. Artifacts: `models/`.

## 6. Evaluation design

Headline metric is **leakage-free OOF** (GroupKFold by theorem — a theorem's programs
never split across train/test). Generalization probed by GroupKFold over theorem,
namespace, and cluster. Ranking metrics are per-theorem (first-success rank, top-k
recovery); budget metrics simulate top-B-per-theorem. Leakage controlled by the
grouped split + the ablation.

## 7. Results

**OOF classification (group = theorem):**

| model | PR-AUC | ROC-AUC | prec@10 | recall@50 | credit PR-AUC |
|---|---|---|---|---|---|
| heuristic | 0.017 | 0.44 | 0.10 | 0.04 | 0.017 |
| logistic | 0.104 | 0.81 | 0.10 | 0.35 | 0.108 |
| sgd | 0.014 | 0.74 | 0.00 | 0.00 | 0.014 |
| **hgb** | **0.522** | 0.77 | **0.90** | 0.57 | 0.544 |

**Per-theorem ranking (13 theorems with a success):**

| ordering | mean first-success rank | top1 | top3 | top5 |
|---|---|---|---|---|
| TR3 original order | 16.46 | 0.385 | 0.385 | 0.462 |
| random (expected) | 8.73 | — | — | — |
| heuristic | 5.15 | 0.385 | 0.615 | 0.769 |
| logistic | 2.69 | 0.385 | 0.769 | 0.846 |
| **hgb** | **2.08** | 0.462 | 0.769 | **1.000** |

## 8. Budget simulation

| HGB budget | successes | success frac | credited | programs run | probe reduction |
|---|---|---|---|---|---|
| B=1 | 6/13 | 0.46 | 6/12 | 92 | 97.7 % |
| B=3 | (see json) | | | | |
| **B=5** | **13/13** | **1.00** | **12/12** | **439** | **88.8 %** |
| B=10 | 13/13 | 1.00 | 12/12 | 867 | 77.9 % |
| top10 % | 10/13 | 0.77 | 9/12 | 440 | 88.8 % |
| top20 % | 12/13 | 0.92 | 11/12 | 822 | 79.1 % |

A fixed **B=5 programs/theorem** is the sweet spot: full success recovery at 88.8 %
fewer probes. → `RANKER_USEFUL_FOR_PROBE_REDUCTION`.

## 9. Error analysis

- **False positives:** of the top-50 OOF-ranked failures, only 2 are unknown-name — the
  ranker is not fooled by out-of-scope lemmas; most top-ranked failures are
  plausible-but-failing simp/aesop programs.
- **False negatives:** 10/23 successes rank below global-100 — chiefly SF5/RC4A
  duplicate-context rows and folds with no in-train positives.
- **Generalization / leakage (the key caveat):** OOF PR-AUC is strong by-theorem (0.52)
  and moderate by-cluster (0.34) but **collapses by-namespace (0.008)** — the same gap
  as TR1. The ablation shows it is **not** theorem-name memorization (name-only ≈ base
  rate); rather, the learned "program-family → success" mapping is **namespace-specific**
  because positives exist only in Set/Finset/List. **Within seen namespaces probe
  reduction is real; transfer to an unseen namespace (e.g. Nat) is not established.**
- **Imbalance:** 23 positives is the binding constraint, not model capacity.

Recommendations: collect more positives via RC4B/RC4C before retraining; a scope-aware
retrieval index helps more than further model tuning; route Nat arithmetic (0 positives)
to a depth/search experiment, not retrieval.

## 10. Active probing queue

92-theorem prioritized queue (leakage-free OOF HGB, no live probes): 13 already-won,
70 flagged `candidate_family_validation` (top-program is d2_simp_aesop / Set.disjoint_left
→ **direct RC4B/RC4C support**), 5 high-uncertainty (useful labels), 4
underrepresented-namespace. Feeds a future TR5 live search. Artifacts:
`out/tr4_active_probe_queue.*`.

## 11. Decision

**`RANKER_USEFUL_FOR_PROBE_REDUCTION`** — the HGB program ranker recovers 100 % of
TR3 successes at B=5/theorem (88.8 % probe reduction), leakage-free by theorem, beating
the heuristic and the TR3 original order, using genuine program/retrieval interactions
(not name memorization). **Scope:** demonstrated *within seen namespaces only*;
cross-namespace transfer is not established (by-namespace PR-AUC 0.008) and the 23
positives are few — so this is a usable search accelerator, **not** a general predictor.

## 12. Next steps

- **Use the queue for a TR5 live search** with a B≈5 ranker-ordered budget over the open
  (no-win) theorems — expected ~85–90 % probe savings within Set/Finset/List.
- **Validate RC4B (`Set.disjoint_left`) and RC4C (`d2_simp_aesop`)** with the RC4A
  harness; these also add positives that broaden the ranker.
- **Grow the program-level dataset** (more namespaces with positives) before further
  model tuning; improve retrieval index scope.
- Do **not** tune the model further until positives grow — data, not capacity, is the limit.

## 13. Protected-file confirmation

- RC1 / RC2-release wrappers, ns24 router, NS9, REL/RC reports, TR1/TR2/SF/TR source
  datasets — **untouched**. No production routing change, no RC4 release, no README
  update, **model not promoted**, no commit. `git diff --stat HEAD` over the three
  protected wrappers is empty. Artifacts under `project/evolve/experiments/tr4/` &
  `project/evolve/reports/tr4/`, scripts `scripts/tr4_*.py`. No live Lean run.
