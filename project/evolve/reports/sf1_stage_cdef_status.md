# SF1 Stage C/D/E/F — status report

- seed: `1729`  | branch: `rc1-production-stack`
- scope: additive only; RC1 production stack untouched.

## 1. Executive summary

- Stage A/B previously produced **52 open-frontier declarations** (Set=43, Multiset=3).
- **Stage C (classify): REAL** — deterministic name+namespace(+statement) tagging and candidate-family scoring. 52 low-confidence (statement/type unavailable in artifact-mode catalog).
- **Stage D (batch): REAL** — 7 deterministic batches generated.
- **Stage E (eval): WIRED** — adapter builds routing-format theorem-set files + exact commands; real run = `False` (supported configs ran=0, dry=0, blocked=4, unsupported=4).
- **Stage F (relabel queue): REAL** — 33 candidates queued for NS23-style minimal-sufficient relabeling (no win claimed without confirmation).

## 2. Frontier classification

- namespace histogram: `Set`=43, `Multiset`=3, `Eq`=1, `Function`=1, `Nat`=1, `Prop`=1, `GENERAL_FRONTIER`=1, `traces_from_search`=1
- top tags: `likely_simp`=52, `likely_aesop`=46, `has_set`=43, `has_logic`=26, `has_iff`=18, `likely_cases`=15, `has_eq`=11, `likely_extensionality`=9, `has_subset`=8, `has_singleton`=8, `has_insert`=6, `has_membership`=6, `has_empty`=6, `has_order`=5, `has_inter`=5
- top-candidate-family counts: `future_failure_driven_lemma_candidate`=30, `rc1_production_stack`=13, `broad_set_aesop_rejected`=6, `wx3_multiset_induction`=3
- candidate-family histogram (score ≥ 0.5): `future_failure_driven_lemma_candidate`=30, `broad_set_aesop_rejected`=9, `rc1_production_stack`=8, `ns9_base_wrapper`=6, `wx3_multiset_induction`=3, `ax4_learned_symbolic_selector_off_by_default`=3
- Set frontier: 43  | Multiset holdout: 3  | low-confidence: 52

## 3. Batch generation

| batch | size | dominant ns | dominant family | intended use |
|---|---|---|---|---|
| `sf1_frontier_all` | 52 | `Set`=43, `Multiset`=3, `Nat`=1, `GENERAL_FRONTIER`=1, `Prop`=1 | `future_failure_driven_lemma_candidate`=30, `rc1_production_stack`=13, `broad_set_aesop_rejected`=6, `wx3_multiset_induction`=3 | full open frontier; baseline raw/ns9/rc1 sweep |
| `sf1_set_frontier` | 43 | `Set`=43 | `future_failure_driven_lemma_candidate`=24, `rc1_production_stack`=13, `broad_set_aesop_rejected`=6 | Set-heavy surface; probe mx2 narrow Set.Finite aesop relevance |
| `sf1_multiset_holdout` | 3 | `Multiset`=3 | `wx3_multiset_induction`=3 | Multiset holdout; WX3 induction regression/extension guard |
| `sf1_mx2_candidate` | 0 | _(none)_ | _(none)_ | high mx2 score; targeted Set.Finite/toFinset aesop eval |
| `sf1_wx3_candidate` | 3 | `Multiset`=3 | `wx3_multiset_induction`=3 | high wx3 score; targeted Multiset induction eval |
| `sf1_balanced_mini` | 8 | `Nat`=1, `Prop`=1, `traces_from_search`=1, `Function`=1, `Set`=1 | `future_failure_driven_lemma_candidate`=7, `wx3_multiset_induction`=1 | small deterministic cross-namespace smoke batch (cheap eval) |
| `sf1_failure_driven_seed` | 52 | `Set`=43, `Multiset`=3, `Nat`=1, `GENERAL_FRONTIER`=1, `Prop`=1 | `future_failure_driven_lemma_candidate`=30, `rc1_production_stack`=13, `broad_set_aesop_rejected`=6, `wx3_multiset_induction`=3 | weak productive-family match / low confidence; SF2 lemma-discovery seed |

## 4. Eval wiring

- eval script present: `True`  | `--help` ok: `True`  | detected flags: ['--theorem-set', '--policy-type', '--route-config', '--strategy-config', '--ckpt-dir', '--top-k', '--max-steps', '--out-dir']
- file_path map size (for routing-format adapter): 1868; NS9 config discovered: `project/evolve/ns9_runs/baseline/strategy_config.json`
- policy support:
  - **raw**: supported=`False` — raw baseline requires a confirmed base-policy ckpt + no-wrapper mode; not safely discoverable from SF1 scope
  - **ns9**: supported=`True` — NS9 strategy config discovered: project/evolve/ns9_runs/baseline/strategy_config.json
  - **rc1**: supported=`True` — RC1 production stack (README command)
  - **experimental**: supported=`False` — no safe SF1 experimental candidate config enabled; experimental wrappers off by default this stage
- per-(batch,policy) outcomes: ran=0, dry_run=0, blocked_missing_file_path=4, unsupported=4.
- exact commands: `project/evolve/experiments/sf1/out/real/eval_commands.sh`
- **limitation**: artifact-mode frontier rows lack `file_path`/`statement`; live eval requires backfilling `file_path` from the traced cache (Stage A live-mode TODO). The adapter recovers file_path where possible from `discovered_theorems.json` + routing sets and records `n_missing_file_path` per batch.

## 5. Relabeling queue

- queue size: **33**  | per-theorem eval available: `False`
- by priority: `high`=3, `medium`=30
- by candidate family: `wx3_multiset_induction`=3, `future_failure_driven_lemma_candidate`=30
- high-priority decls: ['Multiset.disjoint_toFinset', 'Multiset.toFinset_eq_singleton_iff', 'Multiset.toFinset_nsmul']
- family-specific ladders are emitted per row (wx3 → `Multiset.induction_on <;> simp_all`; mx2 → `simp/simp_all/aesop/classical; aesop/exact?`; arith → `simp/omega`; ext → `simp/ext/aesop`).
- **connection to NS23**: this queue is the input to minimal-sufficient relabeling — each candidate must be confirmed by the minimal tactic ladder before any family-win attribution.

## 6. Promotion decision

- recommendation: **MINE_MORE**
- rationale: no live RC1/raw/NS9 deltas were confirmed this stage (real_eval_ran=`False`); SF1 produces the queue, not a promotion.
- RC2 promotion remains gated on ALL of: positive delta over RC1, zero regressions, zero off-gate emissions, minimal-sufficient attribution, deterministic reproduction.

## 7. Connection to long-term theorem discovery

- SF1 C/D/E/F is the bridge from proof-search to **failure-driven lemma discovery**: it turns the open frontier into classified, batched, evaluable, relabel-queued units.
- **SF2 — Failure Pattern Miner** (next): RC1 failures → recurring proof-state/goal patterns → missing-lemma templates. The `sf1_failure_driven_seed` batch + the `future_failure_driven_lemma_candidate` family scores are the seed inputs.
- **SF3 — Lemma Inventor**: candidate lemma → proof → downstream-utility test.

## 8. Protected-file confirmation

`git diff --stat HEAD` for protected files (empty = unchanged):

```
(no changes to rc1_production_wrapper.json or ns24_router.json)
```

Working-tree status:

```
M README.md
?? project/evolve/experiments/sf1/
?? project/evolve/reports/sf1_design.md
?? project/evolve/reports/sf1_promotion_report.md
?? project/evolve/reports/sf1_stage_ab_status.md
?? scripts/sf1_classify_frontier.py
?? scripts/sf1_common.py
?? scripts/sf1_eval_matrix.py
?? scripts/sf1_extract_mathlib_catalog.py
?? scripts/sf1_filter_consumed_surfaces.py
?? scripts/sf1_make_batches.py
?? scripts/sf1_minimal_relabel_new_wins.py
?? scripts/sf1_promotion_report.py
```

All SF1 Stage C/D/E/F changes are additive (upgraded `sf1_classify_frontier.py`, `sf1_make_batches.py`, `sf1_eval_matrix.py`, `sf1_minimal_relabel_new_wins.py`, `sf1_promotion_report.py`; new batches + out/real artifacts + this report). RC1 wrapper, NS9 genome, NS24 router, and REL1 reports were not modified.
