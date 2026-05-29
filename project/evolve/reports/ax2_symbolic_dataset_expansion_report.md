# AX2 — symbolic-action dataset expansion & training-readiness study

**Arc type:** mining + dataset construction + readiness analysis (no
production training). **Branch:** `ax2-symbolic-dataset-expansion`.
**Router:** `ns24_router`. **Baseline wrapper:** NS9 genome (unmodified).
**Symbolic layer:** AX1, unchanged.

## 0. Goal

AX1 built a symbolic-action bridge (`CASES_SIMP[List,simp_all]` etc.) whose
label vocabulary is tiny — 4 labels cover the 27 WX1+WX2 wins — and
recommended growing that dataset before training a symbolic-action
predictor (AX3). AX2 mines fresh Option/List theorems under the AX1
symbolic wrapper, builds a larger symbolic-label dataset, applies the
NS23 minimal-tactic discipline, and decides whether there is enough clean
data to train. **No NS9/router/AX1 artifacts were modified; no checkpoints
written.**

## 1. AX1 recap

The AX1 symbolic config (`project/evolve/experiments/ax1/
ax1_symbolic_option_list_cases.json` = NS9 genome + 5 symbolic actions)
**reproduces WX2 exactly** (Δ=0 on all 6 sets, 0 regressions, 0 emissions
outside gated Option/List). Its symbolic-label dataset = the 27 WX1+WX2
wins → **4 stable labels** (`CASES_SIMP[Option,simp]`×17, `[List,simp]`×6,
`[List,simp_all]`×3, `[Option,simp_all]`×1). All 27 are **single-shot**
closers: one `cases <var> <;> simp[_all]` closes the goal from the initial
state.

## 2. Catalog audit (Stage 1)

`scripts/ax2_symbolic_catalog_audit.py` →
`project/data/ax2_symbolic_catalog_audit_meta.json`. `used` = union of all
registered theorem sets (CX3 / WX1 / WX2 / AX1-equivalence / demo_v1).

| namespace | available | fresh unused | discovered-only (unverified) |
|---|---:|---:|---:|
| Option | 46 | **0** | 0 |
| List | 260 | **76** | 0 |

**Option/Bool/Sum/Prod are fully exhausted** — 0 fresh even in the broader
3989-theorem `discovered_theorems_cx1.json` scan. The only fresh
symbolic-mining surface is **List (76)**, classified (a prior, not ground
truth): `list_cases_simp` 51, `list_induction_simp` 10, `list_hard_unknown`
12, `list_simp_only` 3. This matches the WX2 finding that List is the sole
remaining cases/induction-friendly surface. **Dataset growth is therefore
List-only.**

## 3. Fresh theorem sets (Stage 2)

`scripts/ax2_build_theorem_sets.py` →
`project/evolve/routing/ax2_theorem_sets.json` (loaded by
`tasks._load_ax2_sets`). All 76 fresh List theorems, three disjoint sets;
the spec's Option sets are intentionally **empty** (exhaustion):

| set | n | content |
|---|---:|---|
| `ax2_option_cases_fresh` | 0 | empty (Option exhausted) |
| `ax2_option_simp_fresh` | 0 | empty (Option exhausted) |
| `ax2_list_cases_fresh` | 51 | structural constructor-split candidates |
| `ax2_list_induction_fresh` | 10 | fold/length/sum (induction-leaning) |
| `ax2_option_list_mixed_fresh` | 15 | simp-only (3) + hard/unknown (12) |

## 4. Mining probe: raw vs NS9 vs AX1-symbolic (Stage 3)

`scripts/ax2_run_eval.sh` (9 runs) + `ax2_symbolic_mining_probe_extract.py`
→ `project/data/ax2_symbolic_mining_probe_meta.json`. All on `ns24_router`,
top-k 8, max-steps 8.

| set | avail | raw | ns9 | AX1-sym | sym-only >NS9 | regr |
|---|---:|---:|---:|---:|---:|---:|
| ax2_list_cases_fresh | 46 | 4 | 4 | 7 | **+3** | 0 |
| ax2_list_induction_fresh | 10 | 0 | 0 | 0 | 0 | 0 |
| ax2_option_list_mixed_fresh | 15 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | 71 | 4 | 4 | 7 | **+3** | **0** |

**The AX1 symbolic wrapper adds +3 new wins beyond NS9 on the fresh List
surface, zero regressions.** NS9 is a no-op on List (raw == ns9), as in
WX2. The +3 wins:

| theorem | winning tactic | origin | symbolic action id |
|---|---|---|---|
| `List.headI_dedup` | `cases l <;> simp_all` | wrapper_symbolic_action | `CASES_SIMP[List,simp_all]` |
| `List.tail_dedup` | `cases l <;> simp_all` | wrapper_symbolic_action | `CASES_SIMP[List,simp_all]` |
| `List.zipLeft_nil_right` | `aesop` | generative_topk | — |

Induction (10) and the hard/simp-only mixed set (15) yielded **nothing** —
consistent with WX2 (`induction` adds nothing at this scale; the residual
fresh List is genuinely harder than the WX2-consumed structural lemmas).

## 5. Minimal-symbolic relabeling (Stage 5)

`scripts/ax2_relabel_minimal_symbolic_actions.py` →
`ax2_minimal_symbolic_labels.json`, `ax2_symbolic_family_pools_meta.json`.
Battery (simplest first) from the **initial state**: assumption, rfl,
decide, simp, simp_all, aesop, then symbolic `CASES_SIMP`/`INDUCTION_SIMP`
(vars from the live state), then the raw wrapper tactic.

**Key result — the +3 fresh wins are multi-step, not single-shot:**

| theorem | single-shot closes? | classification |
|---|:---:|---|
| `List.headI_dedup` | no | `MULTISTEP_SYMBOLIC[CASES_SIMP[List,simp_all]]` |
| `List.tail_dedup` | no | `MULTISTEP_SYMBOLIC[CASES_SIMP[List,simp_all]]` |
| `List.zipLeft_nil_right` | no | `MULTISTEP_NONSYMBOLIC` (aesop) |

Verified directly: `cases l <;> simp_all` from the initial state of
`List.headI_dedup` *advances* (`TacticState`) but does **not** close — the
eval proof was a 3-step search (`aesop` + two `cases l <;> simp_all`
applications). So:

- **clean single-shot symbolic (AX3-trainable): 0**
- multistep symbolic-assisted (symbolic action in the winning path, but no
  single tactic closes → weak label): **2**
- multistep non-symbolic (aesop): **1**

Unlike the AX1 27 (single `cases <var> <;> simp` closers), none of the
fresh-List wins reduces to a single clean symbolic-action label. The
single-shot `cases <;> simp` pattern monetizes only the *easiest*
constructor-split lemmas — and those were already consumed by WX2.

## 6. Symbolic-label dataset (Stage 4)

`scripts/ax2_build_symbolic_label_dataset.py` →
`ax2_symbolic_label_dataset_meta.json` (+ tiny gitignored
`ax2_symbolic_label_dataset.jsonl`). Merge AX1 27 (single-shot, trusted) +
3 AX2 fresh wins.

- total examples: **30** (27 AX1 + 3 AX2)
- **clean single-shot symbolic examples: 27** (all AX1; AX2 added 0)
- labels (clean): `CASES_SIMP[Option,simp]` 17, `[List,simp]` 6,
  `[List,simp_all]` 3, `[Option,simp_all]` 1
- by namespace (clean): Option 18, List 9

## 7. Readiness decision (Stage 6)

`scripts/ax2_readiness_decision.py` → `ax2_readiness_meta.json`.

| metric | value |
|---|---|
| clean symbolic examples | **27** |
| unique theorems | 27 |
| labels | 4 |
| max label count | 17 (`CASES_SIMP[Option,simp]`) |
| label entropy | 1.43 bits (balance 0.72) |
| held-out feasible | yes (mechanically) |

Thresholds: Green ≥80 total & a label ≥30; Yellow 40–79 & dominant ≥20;
Red <40.

### Classification: **RED** (27 < 40; top label 17 < 20)

AX2 mined the entire fresh available surface and added **0 clean
single-shot training examples**. The symbolic-label dataset is **capped at
~27 by catalog exhaustion + difficulty**, not by mining effort:
Option/Bool are gone; the easy structural List is gone; the residual fresh
List needs multi-step proofs the single-action label cannot express.

## 8. Smoke classifier (Stage 7)

**Skipped** — gated to Yellow/Green; readiness is RED. Training a per-state
classifier on 27 examples across 4 labels (17/6/3/1) would learn little
beyond the Option-simp majority and is not justified.
(`ax2_smoke_classifier_metrics.json` records the skip.)

## 9. Decision & recommendation

**Recommend WX3 (more mining / wrapper extension), NOT AX3 training.**

The symbolic-action *abstraction* is validated (AX1 Δ=0; AX2 reproduces it
on fresh List, +3/0-regressions) and is a genuinely useful **search-time
wrapper capability**. But a *learned symbolic-action predictor* (AX3) is
not justified: 27 single-shot examples is far below a trainable floor, and
the available catalog cannot grow it. Concretely, WX3 should pursue, in
rough order of likely yield:

1. **Multiset quotient-aware action** (`Multiset.induction_on`-style) — the
   largest untapped fresh surface (~250), but a *new* action type, since
   raw `cases`/`induction` does not apply to a quotient.
2. **Multi-step symbolic action sequences** — the 2 fresh wins
   (`headI_dedup`, `tail_dedup`) are `CASES_SIMP` *followed by* further
   search. A composite/2-step symbolic label would capture this class
   (and unlock the harder residual List), at the cost of a larger action
   space.
3. **Keep the symbolic layer as the canonical cases wrapper** (it equals
   WX2 and is more general) and **defer AX3** until a surface yields enough
   clean single-shot symbolic labels. No short-token raw-SFT family has
   appeared since Int/omega; no clean symbolic-SFT family has reached the
   gate either.

The NS23/minimal-tactic discipline did its job again: it prevented
over-attributing 3 multi-step wins (incl. an `aesop` win) as clean
symbolic-action training labels.

## Artifacts

Scripts: `scripts/ax2_symbolic_catalog_audit.py`,
`ax2_build_theorem_sets.py`, `ax2_run_eval.sh`,
`ax2_symbolic_mining_probe_extract.py`,
`ax2_build_symbolic_label_dataset.py`,
`ax2_relabel_minimal_symbolic_actions.py`, `ax2_readiness_decision.py`,
`ax2_smoke_classifier.py`. Config: `project/evolve/routing/
ax2_theorem_sets.json` + `tasks._load_ax2_sets`. Metadata:
`project/data/ax2_symbolic_catalog_audit_meta.json`,
`ax2_symbolic_mining_probe_meta.json`, `ax2_minimal_symbolic_labels.json`,
`ax2_symbolic_family_pools_meta.json`,
`ax2_symbolic_label_dataset_meta.json`, `ax2_readiness_meta.json`,
`ax2_smoke_classifier_metrics.json`. Report: this file. Eval
traces/logs/run dirs (`ax2_*`) and the per-row label JSONL are gitignored;
no checkpoints or classifier artifacts produced.
