# AX4 — Multiset symbolic-label expansion to Green + learner v2

**Arc type:** broader-catalog mining + dataset expansion to the symbolic-
learning **Green** gate + retrain/evaluate the symbolic-action learner under a
clean reserved held-out protocol. **Branch:** `ax4-multiset-symbolic-green`.
**Router:** `ns24_router`. **Baseline wrapper:** NS9 genome (unmodified).
**Symbolic layer:** AX1/WX3, unchanged. **No NS9/router/AX1/AX2/WX3/AX3
artifacts modified; no checkpoints overwritten.** The trained v2 classifier and
the dataset JSONL are git-ignored.

## 0. AX3 recap

AX3 trained the first symbolic-action learner but landed **YELLOW**: 26 clean
labels (23 `simp_all` + 3 `simp`), CV positive recall ~0.85, NULL-FP ~0.05, but
only **1** held-out symbolic positive — the reserved induction surface was too
thin to evaluate held-out theorem-level wins. AX3's recommendation was to mine
to Green before promoting the learner. AX4 does exactly that.

## 1. Catalog expansion (Stage 1)

`scripts/ax4_multiset_catalog_expand.py`. AX3 had consumed the *confirmed-
available* Multiset surface (260 available, 259 in prior sets). AX4 mines the
broader **discovered** catalog (`discovered_theorems_cx1.json`: 573 Multiset
names scanned from the pinned mathlib). Frontier =

    discovered (573) − available-260 (consumed) − prior sets − labeled = **313**

availability-unconfirmed candidates, bucketed by induction-likelihood (high 71,
medium 119, hard 45, negative 78; 50 cross-surface). Output:
`project/data/ax4_multiset_catalog_expand_meta.json` + report. **Key empirical
finding: frontier availability attrition was ~0** — every mined frontier
theorem loaded under the pinned mathlib (15/15, 22/22, 45/45, 55/55, 28/28,
28/28, 53/53). The "unconfirmed" frontier was in fact fully available; the
cx1 availability probe had simply never sampled these files.

## 2. Theorem sets (Stage 2)

`scripts/build_ax4_theorem_sets.py` → `ax4_theorem_sets.json`,
`tasks._load_ax4_sets`. Seven disjoint sets (**246** candidates), with a fresh
reserved held-out (`heldout2`, 53 unused medium-confidence candidates) added
after the first pass to reach the Green held-out-positive threshold:

| set | n | role |
|---|---:|---|
| `ax4_multiset_induction_high_confidence` | 15 | mine→train |
| `ax4_multiset_cross_surface` | 22 | mine→train (non-Basic files) |
| `ax4_multiset_induction_medium_confidence` | 55 | mine→train |
| `ax4_multiset_induction_hard` | 28 | mine→train |
| `ax4_multiset_induction_heldout` | 45 | reserved held-out |
| `ax4_multiset_negative_control` | 28 | reserved held-out (expected-NULL) |
| `ax4_multiset_induction_heldout2` | 53 | reserved held-out (2nd pass) |

## 3. Mining (Stage 3)

`scripts/ax4_run_mining_guarded.sh` (+ `run_with_timeout.py` guard; wx3ind
scheduled first per set) and `scripts/ax4_extract_probe.py` (metrics with a
traces.jsonl fallback for any timeout-killed cell). raw vs NS9 vs WX3-induction
on `ns24_router`:

| set | raw | NS9 | WX3-ind | WX3-only | symbolic | regr |
|---|---:|---:|---:|---:|---:|---:|
| high_confidence | 3 | 3 | 7 | 4 | 4 | 0 |
| cross_surface | 8 | 8 | 9 | 1 | 0 | 0 |
| induction_heldout | 10 | 10 | 14 | 4 | 4 | 0 |
| medium_confidence | 10 | 11 | 17 | 6 | 6 | 0 |
| induction_hard | 7 | 7 | 7 | 0 | 0 | 0 |
| negative_control | 8 | 8 | 12 | 4 | 4 | 0 |
| induction_heldout2 | 12 | 12 | 19 | 7 | 4 | 0 |

**Totals: +26 WX3-only beyond NS9, 22 `wrapper_symbolic_action` wins (all
`multiset_induction_simp_all`), 0 regressions.** Findings: the induction-on
symbolic family is concentrated in `Multiset/Basic.lean` — cross-surface
(Bind/Dedup/Lattice) and the `hard` bucket yield ~0 symbolic; the `negative`
bucket over-excluded 4 genuinely induction-solvable lemmas (countP_*/count_bind)
which became real labels.

## 4. Minimal relabel (Stage 4)

`scripts/ax4_relabel_minimal_multiset_symbolic.py` (the same NS23/WX 13-tactic
minimal-first battery + `state_pp` capture). Of the 26 WX3-only wins:
**20 clean single-shot symbolic** (18 `simp_all` + 2 `simp`), **0
over-attributed**, 2 multi-step-assisted. The 0 over-attribution rate confirms
these are genuine induction-needed closes, not simp/aesop mislabels.

## 5. Dataset + readiness (Stage 5)

`scripts/ax4_build_multiset_symbolic_dataset.py` → meta JSON + git-ignored
JSONL. Merges WX3 (20) + AX3 (6) + AX4 (20) clean positives, dedups by name,
adds NULL negatives, and splits train/held-out by reserved-set membership.

| | count |
|---|---:|
| **total clean symbolic labels** | **46** (WX3 20 + AX3 6 + AX4 20) |
| `MULTISET_INDUCTION_SIMP[Multiset,simp_all]` | **41** |
| `MULTISET_INDUCTION_SIMP[Multiset,simp]` | 5 |
| Multiset NULL negatives | 305 |
| non-Multiset control NULL | 53 |
| total rows | 404 |
| **train positives / held-out positives** | 34 / **12** |

**Readiness: GREEN.** ≥40 total (46) ✓, ≥30 `simp_all` (41) ✓, ≥10 held-out
positives (12) ✓, negative controls present ✓.

## 6. Symbolic-action learner v2 (Stage 6)

`scripts/ax4_train_symbolic_classifier.py`: TF-IDF (char_wb 3–5) + balanced
logistic regression over the proof-state prompt; classes = `NULL` + the two
Multiset action ids. Trains on `train_candidate`; the reserved held-out is never
trained on. Model → git-ignored `project/models/ax4_multiset_symbolic_clf/`.

**Stratified CV feature-source ablation (all rows):**

| features | top-1 | positive recall | NULL FP |
|---|---:|---:|---:|
| name + state | 0.896 | 0.717 | 0.081 |
| name only | 0.884 | 0.543 | 0.073 |
| **state only** | 0.874 | **0.761** | 0.112 |

The learner keys on the **proof state** (state-only recall 0.76 ≫ name-only
0.54) — it is learning state structure, not memorizing theorem names, which is
the desired inductive bias for a symbolic-action predictor.

## 7. Held-out theorem-level evaluation (Stage 7)

`project/evolve/experiments/ax4/ax4_multiset_symbolic_predictor_v2.json` (NS9
base byte-identical to `ns9_best_genome.json` + WX3 `symbolic_actions` + a
`symbolic_predictor` block, **`enabled: false`**) and
`scripts/ax4_predictor_heldout_eval.py`. The action is additive + namespace-
gated, so predictor vs oracle is computed exactly offline (predictor win =
oracle symbolic win AND classifier fires ≥0.5):

| held-out set | NS9 | oracle-sym | predictor | retain | precision | regr |
|---|---:|---:|---:|---:|---:|---:|
| ax4 induction_heldout | 10 | 4 | 3 | 0.75 | 0.50 | 0 |
| ax4 induction_heldout2 | 12 | 4 | 2 | 0.50 | 0.67 | 0 |
| ax4 negative_control | 8 | 4 | 1 | 0.25 | 1.00 | 0 |
| ax3 induction_heldout | 0 | 1 | 1 | 1.00 | 1.00 | 0 |
| **TOTAL** | | **13** | **7** | **0.538** | | **0** |

Non-Multiset false-positive control — **0 firing at every threshold**:
demo_v1 0/15, nat_defs_medium 0/38, ns17_set_extra 0/30, ns17_finset_extra
0/30. Effective non-Multiset FP after the namespace gate = **0**.

**Promotion criterion — MET:** retain ≥50% (0.538) ✓, 0 regressions ✓,
effective non-Multiset FP = 0 ✓, clear operating threshold (0.5; the sweep
shows lower thresholds add no positives — the 5 missed held-out positives have
NULL as argmax — and 0.6 trims Multiset-NULL FP 7→1). This is the **first time
a learned symbolic-action predictor clears the promotion bar.**

## 8. Decision

- **Green reached, learner v2 promotable.** The cap in AX3 was label volume,
  not method: the broader discovered catalog (with ~0 availability attrition)
  yielded +20 clean labels and a 12-positive held-out surface, lifting the
  dataset over every Green threshold with 0 regressions throughout.
- **But the deterministic WX3 oracle still dominates raw coverage.** For a
  single additive, namespace-gated action, always-emit (oracle) retains
  **13/13** held-out wins at zero model cost; the learned gate retains
  **7/13** (it suppresses ~46% where it scores NULL). The predictor passes the
  bar but cannot beat free-and-total emission in this regime.

**Recommendation (in order):**
1. **Keep the WX3 oracle wrapper as the production Multiset default** — it is
   deterministic, +26/+7 wins, 0 regressions, no model.
2. **Mark the v2 predictor promotion-eligible but leave it off-by-default**
   (config written, `enabled: false`). Its selectivity only pays where emission
   has cost — i.e. **multi-action symbolic search** (choosing among many
   candidate symbolic actions under a step/candidate budget), where always-emit
   is no longer free. The state-only feature signal (recall 0.76, 0
   cross-namespace FP) shows the learned selector is sound enough to drive that.
3. **Sequence-level / multi-action symbolic search is now warranted** — that is
   the regime where a learned selector beats the oracle, and AX4 has produced a
   selector that clears the held-out promotion bar to drive it.

## Artifacts

Scripts: `ax4_multiset_catalog_expand.py`, `build_ax4_theorem_sets.py`,
`ax4_run_mining_guarded.sh`, `ax4_extract_probe.py`,
`ax4_relabel_minimal_multiset_symbolic.py`,
`ax4_build_multiset_symbolic_dataset.py`, `ax4_train_symbolic_classifier.py`,
`ax4_predictor_heldout_eval.py` (reuses `run_with_timeout.py`). Config:
`project/evolve/experiments/ax4/ax4_multiset_symbolic_predictor_v2.json`,
`project/evolve/routing/ax4_theorem_sets.json`. Metadata: `project/data/ax4_*`.
The dataset JSONL, v2 classifier model, and eval traces/logs/run dirs are
git-ignored.
