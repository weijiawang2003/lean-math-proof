# TR1 — Failure-to-Action Router from Verified Proof-Search Traces

**Branch:** `sx3-depth2-sequence-search`  ·  **Date:** 2026-05-30  ·  **No commit made.**
A supervised router/ranker that predicts, for a theorem / failure state, **which action family is
worth trying next** (or that it is no-cheap-action / missing-lemma / depth-gap). **Not** a proof
generator; **not** wired into production routing.

---

## 1. Executive summary

| field | value |
|---|---|
| dataset | **57** verified examples, **7** populated labels (2 zero-support), 31 with goal text |
| models | rule baseline, logistic, **sgd (best)**, random_forest |
| best model | **SGD (log-loss linear)** — macro-F1 **0.628**, LOO accuracy **0.877**, top-3 **0.93** |
| beats rule baseline | **Yes** (rule macro-F1 0.436) |
| held-out eval on 27 RC2 failures | **0.852** (23/27), 0 abstained |
| grouped (leave-one-namespace-out) accuracy | **0.386** → generalization gap **0.491** |
| **DECISION** | **`PILOT_ONLY_NEEDS_MORE_DATA`** |

The router learns the **well-supported** triage classes cleanly (BASELINE_DUPLICATE F1 0.94,
MISSING_BRIDGE_LEMMA_CANDIDATE 0.97, SET_ITE_SIMP 0.92, SX3_PRODUCTION_SUBSUMED 0.89) and beats the
handcrafted rule baseline. It produces a usable **next-work queue** over the 27 confirmed RC2 failures.
**But** the large within-distribution → grouped generalization drop (0.877 → 0.386) shows it leans on
namespace/name-surface cues, and two positive classes are singletons. **This is a pilot, not a
deployable router.** No production routing change is made.

---

## 2. Motivation

We train a **router**, not a prover, because the bottleneck exposed by RC3/SF4 is *triage*: deciding
which confirmed RC2 failures are worth which cheap action vs. which are missing-lemma / depth-gap. The
labels are exactly the verified outcomes the SX4 + SF4 pipeline already produces, so a small supervised
model can encode "what to try next" and rank a work queue. It complements RC2 (production), SF4
(failure-first mining), and SX4 (attribution): TR1 consumes their verified labels and prioritizes the
residual frontier. It never credits a candidate or changes routing.

---

## 3. Dataset construction

`scripts/tr1_build_training_dataset.py` — **verified-label discipline** (positives only from production
deltas / minimal-relabel-confirmed wins / accepted RC components; SX3 proxy "wins" enter **only** as the
negative class `SX3_PRODUCTION_SUBSUMED`).

Sources: `rc2_delta_ledger`, `sx2 set2 minimal relabel`, `sx4 reattribution`, `sf4 confirmation /
clusters / probe_results / sx4_attribution / missing_lemma_triage`, with goal text from SF2/SX3 traces.

**Label distribution (57):**

| label | type | n |
|---|---|---|
| MISSING_BRIDGE_LEMMA_CANDIDATE | triage | 19 |
| BASELINE_DUPLICATE | negative | 18 |
| NO_CHEAP_ACTION | triage | 7 |
| SET_ITE_SIMP | positive | 6 |
| SX3_PRODUCTION_SUBSUMED | negative | 5 |
| PROOF_SEARCH_DEPTH_GAP | triage | 1 |
| WX3_MULTISET_INDUCTION | positive | 1 |
| MX2_TOFINSET_AESOP | positive | 0 (in map, zero-support) |
| SOURCE_SPECIFIC_OR_REJECTED | negative | 0 (in map, zero-support) |

Confidence: 27 verified, 30 strong. Leakage controls: grouped (namespace) eval + name-only-vs-goal
ablation in the error analysis; no promotion. One theorem precedence note: confirmed failures in a
bridge cluster take `MISSING_BRIDGE_LEMMA_CANDIDATE` (more specific) over generic `NO_CHEAP_ACTION`.

---

## 4. Features

`scripts/tr1_featurize.py` (scikit-learn, 1600 dims): **name** char n-grams (878) + token n-grams (189)
+ namespace one-hot (6); **goal/failure** TF-IDF word (67) + char (426) over `goal_text ⊕ last_error`;
**boolean symbolic flags** (15: has_set/ite/iff/subset/multiset/tofinset/card/singleton/ext/induction +
parse/recursion symptom flags); **cluster** features (19: SF4 cluster id + coarse goal shape + rc2
status). Fitted vectorizers persisted (joblib) for inference on new theorems.

---

## 5. Models

`scripts/tr1_train_router.py`:
- **A. Rule baseline** (deterministic): Set∧ite→SET_ITE_SIMP, Multiset∧induction→WX3_MULTISET_INDUCTION,
  toFinset→MX2_TOFINSET_AESOP, rc2_failed∧(iff∨subset)→MISSING_BRIDGE, rc2_failed→NO_CHEAP_ACTION,
  rc2_solved→BASELINE_DUPLICATE. Macro-F1 **0.436**, accuracy 0.632.
- **B. Logistic** (multinomial, balanced): LOO acc 0.842, macro-F1 0.579, top-3 0.965.
- **C. SGD** (log-loss linear, balanced) — **best**: LOO acc **0.877**, macro-F1 **0.628**, top-3 0.93.
- **D. RandomForest**: acc 0.789, macro-F1 0.462 (does not help at this size).

Honest CV: **LeaveOneOut** OOF (every example predicted by a model that never saw it) given singleton
classes; plus **leave-one-namespace-out** grouped accuracy as a leakage check.

---

## 6. Evaluation

| model | accuracy (LOO) | macro-F1 | top-3 | grouped (LONO) |
|---|---|---|---|---|
| rule_baseline | 0.632 | 0.436 | — | — |
| logistic | 0.842 | 0.579 | 0.965 | 0.404 |
| **sgd** | **0.877** | **0.628** | 0.93 | 0.386 |
| random_forest | 0.789 | 0.462 | 0.965 | 0.105 |

**Best-model (SGD) per-label:** BASELINE_DUPLICATE 0.94 / MISSING_BRIDGE 0.97 / SET_ITE_SIMP 0.92 /
SX3_PRODUCTION_SUBSUMED 0.89 / NO_CHEAP_ACTION 0.67 — all usable (support ≥3). Singleton classes
(PROOF_SEARCH_DEPTH_GAP, WX3_MULTISET_INDUCTION) F1 0.0 (unlearnable from 1 example under LOO).

---

## 7. RC2 failure predictions + next-work queue

`scripts/tr1_eval_router_on_rc2_failures.py` — held-out predictions on the 27 SF4 confirmed failures
(each retrained leaving it out): **held-out accuracy 0.852 (23/27), 0 abstained.**

`tr1_next_work_queue.{json,md}` ranks by actionability of the predicted label (depth-gap → bridge-lemma
→ specific-action → baseline → no-cheap-action; subsumed last). High-confidence directions:
- **PROOF_SEARCH_DEPTH_GAP** (`Set.pairwiseDisjoint_filter`, `Eq.subset`) → deeper bounded search /
  widen aesop routing.
- **MISSING_BRIDGE_LEMMA_CANDIDATE** (the Set iff-equivalence cluster: `antitoneOn_iff_antitone`,
  `diff_singleton_subset_iff`, `pair_eq_pair_iff`, …) → **SF5 existing-lemma retrieval first**.

---

## 8. Error analysis

`scripts/tr1_error_analysis.py`:
- **Generalization gap 0.491** (within-dist LOO 0.877 vs leave-one-namespace-out 0.386): the model
  relies on namespace/name-surface cues; it does not yet transfer across namespaces. Expected for a
  57-example, Set-dominated corpus.
- **Name dominates goal:** SET_ITE / bridge labels are strongly predictable from the theorem name
  alone; goal text adds little at this size (and only 31/57 have it).
- **Low/zero-support labels:** PROOF_SEARCH_DEPTH_GAP (1), WX3_MULTISET_INDUCTION (1),
  MX2_TOFINSET_AESOP (0), SOURCE_SPECIFIC_OR_REJECTED (0) — unreliable.
- **Data-collection targets:** more verified Multiset-induction / Set.Finite-aesop positives; more
  depth-gap cases; **non-Set namespaces** (corpus is Set-heavy); full goal-text capture for live failures.

---

## 9. Decision

### `PILOT_ONLY_NEEDS_MORE_DATA`

The router is real signal — it beats the rule baseline, cleanly separates the well-supported triage
classes, and yields a useful next-work queue — but the small, Set-dominated corpus, singleton positive
classes, and large grouped-generalization gap mean it is a **pilot**. It must not drive production
routing or any candidate credit.

---

## 10. Next steps

- **Use the router to prioritize SF5 retrieval** over the Set iff-equivalence bridge cluster (queue
  ranks #3+) — but verify existing Mathlib lemmas first (per SF4 triage).
- **Collect more verified RC2-failure labels** (active-learning list in
  `tr1_active_learning_cases.json`: 25 high-uncertainty frontier theorems, not RC2-solved, with
  file_path) and **broaden beyond Set** before any deployment consideration.
- **Test model-guided probe selection** only after the grouped-generalization gap closes on a larger,
  multi-namespace corpus.
- Do **not** promote to production routing; do **not** create RC4.

---

## 11. Protected-file confirmation

`git diff --stat HEAD` for `rc1_production_wrapper.json`, `rc2_release/rc2_production_wrapper.json`,
`ns24_router.json` → **empty (untouched)**. NS9 genome/checkpoints and REL1/RC1/RC2 release reports
untouched. README not modified. No production routing change, no RC4, no candidate promoted. **No commit
made** (see `project/evolve/experiments/tr1/out/protected_files_check.txt`).
