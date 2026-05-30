# AX3 — Multiset symbolic-action dataset expansion + first learner

**Arc type:** held-out mining + dataset construction + first symbolic-action
learner (smoke training). **Branch:** `ax3-multiset-symbolic-learning`.
**Router:** `ns24_router`. **Baseline wrapper:** NS9 genome (unmodified).
**Symbolic layer:** AX1/WX3, unchanged. **No NS9/router/AX1/AX2/WX3 artifacts
modified; no checkpoints overwritten.** The trained classifier and dataset
JSONL are git-ignored.

## 0. WX3 recap

WX3 opened the Multiset surface with a state-aware quotient wrapper
(`induction {var} using Multiset.induction_on <;> simp_all`), adding **+25
wins beyond NS9, 0 regressions**, and — crucially — **20 clean single-shot
symbolic labels** (vs AX2's 0), dominated by
`MULTISET_INDUCTION_SIMP[Multiset,simp_all]` (18). That made AX3 — the first
attempt to *learn* symbolic actions rather than raw tactic strings — worth
running, with the open question of whether the clean-label pool could be
pushed over the symbolic-learning gate.

## 1. Held-out mining audit (Stage 1)

`scripts/ax3_multiset_heldout_audit.py`. Of the 251 fresh-available Multiset
theorems, WX3 used 165, leaving **86 held-out** (induction 19, simp 31, ext
32, hard 4). Split into four disjoint sets (`scripts/build_ax3_theorem_sets.py`
→ `project/evolve/routing/ax3_theorem_sets.json`, `tasks._load_ax3_sets`):

| set | n | role |
|---|---:|---|
| `ax3_multiset_induction_mine` | 47 | label mining |
| `ax3_multiset_induction_heldout` | 12 | reserved eval |
| `ax3_multiset_mixed_heldout` | 14 | robustness eval |
| `ax3_multiset_negative_control` | 12 | expected-NULL |

(One mining candidate, `Multiset.eq_of_mem_map_const`, was dropped after it
hung the Lean REPL indefinitely — recorded under `excluded_pathological`;
subsequent runs are guarded by `scripts/run_with_timeout.py` since macOS lacks
coreutils `timeout`.)

## 2. Mining probe (Stage 3)

`scripts/ax3_run_*` + `scripts/ax3_extract_probe.py` →
`project/data/ax3_multiset_mining_probe_meta.json` (raw vs NS9 vs WX3-induction
on `ns24_router`):

| set | raw | NS9 | WX3-ind | WX3-only | symbolic | regr |
|---|---:|---:|---:|---:|---:|---:|
| induction_mine | 5 | 5 | 11 | 6 | 6 | 0 |
| induction_heldout | 0 | 0 | 1 | 1 | 1 | 0 |
| mixed_heldout | 0 | 0 | 0 | 0 | 0 | 0 |
| negative_control | 0 | 0 | 0 | 0 | 0 | 0 |

**+7 WX3-only wins, all `multiset_induction_simp_all`, 0 regressions.** The
negative control yields nothing (good — low false-positive surface). The
held-out induction surface is markedly harder than the WX3 sets (1/12).

## 3. Minimal relabeling (Stage 4)

`scripts/ax3_relabel_minimal_multiset_symbolic.py` (same 13-tactic battery as
WX3, plus `state_pp` capture). Of the 7 WX3-only wins: **6 clean single-shot
symbolic** (5 `simp_all` + 1 `simp`), 0 over-attributed, 1 multi-step-assisted.

## 4. Dataset (Stage 5)

`scripts/ax3_build_multiset_symbolic_dataset.py` → meta JSON + git-ignored
JSONL. Merges WX3 (20) + AX3 (6) clean positives and adds NULL negatives
(Multiset states not closed by induction_on; non-Multiset demo/nat states as
false-positive control). `state_pp` recovered from eval traces.

| | count |
|---|---:|
| **total clean symbolic labels** | **26** (WX3 20 + AX3 6) |
| `MULTISET_INDUCTION_SIMP[Multiset,simp_all]` | 23 |
| `MULTISET_INDUCTION_SIMP[Multiset,simp]` | 3 |
| NULL negatives (Multiset + control) | 132 |
| rows total | 158 |
| train positives / held-out positives | 25 / 1 |

## 5. Readiness (Stage 6)

`scripts/ax3_readiness_decision.py` → **YELLOW (smoke training).**
- total clean labels **26** → in the 25–39 Yellow band (< 40 Green).
- dominant `simp_all` **23 ≥ 20** → meets the dominant-label threshold.
- a held-out split exists (1 held-out positive + 75 held-out NULL).

Not Green: total < 40 and the held-out *positive* surface is thin (the
reserved induction held-out yielded only 1 clean label).

## 6. First symbolic-action learner (Stage 7)

`scripts/ax3_train_symbolic_classifier.py`: a deliberately small, deterministic
**TF-IDF (char_wb 3–5) + balanced logistic regression** over the proof-state
prompt (theorem name + state text), classes = the two Multiset action ids +
`NULL`. (A DistilBERT fine-tune is overkill/unstable at 26 labels; the spec
allows the sklearn baseline as the AX3 smoke.) Model → git-ignored
`project/models/ax3_multiset_symbolic_clf/`.

**Stratified 3-fold CV over all 158 rows** (the natural held-out has only 1
positive, too thin for recall):

| metric | value |
|---|---:|
| overall top-1 | 0.924 |
| **positive family recall** (fires induction on a true positive) | **0.846** |
| NULL false-positive rate | 0.053 |
| `simp_all` recall (n=23) | 0.826 |
| `simp` recall (n=3) | 0.667 |

False positives on the **non-Multiset control** states (demo_v1 +
nat_defs_medium): **1.9% (1/53)** raw — and **0 effective**, since the wrapper
namespace-gates emission to `Multiset`.

## 7. Predictor vs oracle (Stage 8)

`project/evolve/experiments/ax3/ax3_multiset_symbolic_predictor.json` (NS9
base byte-identical to `ns9_best_genome.json` + a `symbolic_predictor` block
gating the WX3 induction action on the classifier). Because the symbolic
action is **additive and namespace-gated**, the predictor's only effect vs the
WX3 oracle is suppressing emissions it scores NULL — so A/B/C are computed
exactly offline (`scripts/ax3_predictor_wrapper_eval.py`):

| set | NS9 | oracle-sym | predictor | retained | precision | regr |
|---|---:|---:|---:|---:|---:|---:|
| induction_heldout | 0 | 1 | 1 | 1.00 | 0.33 | 0 |
| mixed_heldout | 0 | 0 | 0 | — | — | 0 |
| negative_control | 0 | 0 | 0 | — | — | 0 |

- **Retained fraction of oracle wins: 1.0** (1/1 on the held-out symbolic
  surface; CV recall 0.85 is the robust estimate).
- **Preservation: 0 regressions** (by construction — predictor only adds a
  gated candidate; all NS9 wins retained). Control: 0 effective emissions on
  non-Multiset.
- Held-out emission precision is 0.33 only because the reserved held-out
  symbolic surface is a single theorem (2 harmless extra emissions); this is
  a data-starvation artifact, not a model failure (CV NULL-FP is 5.3%).

## 8. Decision

- **The first symbolic-action learner works as a smoke**: ~0.85 positive
  recall, ~0.05 NULL false-positive, ~0 effective non-Multiset leakage,
  0 preservation regressions. Symbolic-action *learning* (not raw-tactic SFT)
  is empirically alive on Multiset.
- **But it is YELLOW, not Green**: 26 clean labels (< 40), held-out symbolic
  surface = 1. The learned predictor does not yet beat simply shipping the
  **WX3 oracle wrapper**, which deterministically captures every win at 0 cost.

**Recommendation (in order):**
1. **Keep the WX3 oracle wrapper** (`wx3_multiset_induction_safe`) as the
   production Multiset capability — it is deterministic, +25/+7 wins, 0
   regressions, and needs no model.
2. **Mine to Green before promoting the learner.** The clean pool is
   label-limited, not method-limited: push `simp_all` past the held-out
   surface by (a) mining the broader discovered Multiset catalog
   (573 discovered vs 251 probed-available) and (b) extending the
   `MULTISET_INDUCTION_SIMP` action to other quotient/inductive namespaces
   (Finset/Sym/Quotient) for cross-namespace label transfer. Target ≥40 total
   / ≥20 held-out-eval positives, then retrain and do **live** wrapper
   integration (load the joblib model in the wrapper behind the
   off-by-default `symbolic_predictor` flag).
3. **Sequence-level symbolic search is not yet warranted** — the single-action
   `induction_on` family is still the dominant, unexhausted yield.

## Artifacts

Scripts: `ax3_multiset_heldout_audit.py`, `build_ax3_theorem_sets.py`,
`ax3_run_eval.sh`, `ax3_run_matrix_parallel.sh`,
`ax3_run_remaining_guarded.sh`, `run_with_timeout.py`, `ax3_extract_probe.py`,
`ax3_relabel_minimal_multiset_symbolic.py`,
`ax3_build_multiset_symbolic_dataset.py`, `ax3_readiness_decision.py`,
`ax3_train_symbolic_classifier.py`, `ax3_predictor_wrapper_eval.py`. Config:
`project/evolve/experiments/ax3/ax3_multiset_symbolic_predictor.json`,
`project/evolve/routing/ax3_theorem_sets.json`. Metadata:
`project/data/ax3_*`. The dataset JSONL, classifier model, and eval
traces/logs/run dirs are git-ignored.
