# SF1 Live Eval Unblocker — status report

- branch: `rc1-production-stack`  | seed: `1729`  | scope: additive only; protected configs untouched.

## 1. Executive summary

- **Real RC1 smoke eval RAN.** sets=['sf1_multiset_holdout_runnable'] solved/total = 0/3.
- NS9 comparison ran: `False`.
- Remaining blocker: none for RC1 smoke.

## 2. Previous blocker diagnosis & resolution

- **Blocker 1 — missing file_path.** Artifact-mode frontier rows had no `file_path`. Resolved by deterministic exact-name backfill (`scripts/sf1_backfill_frontier_paths.py`) from discovered_theorems.json + routing + tasks + eval_runs metrics.
- **Blocker 2 — `--theorem-set` accepts only registered names** (`choices=list_theorem_sets()`). Resolved with **least-invasive Path A (runtime registration)**: `scripts/sf1_run_eval.py` patches the `get_theorems`/`list_theorem_sets` names that `eval_rollout_all` already imported and registers the SF1 set in-process, then delegates to `eval_rollout_all.main()`. **No edit** to `eval_rollout_all.py`, `tasks.py`, or any protected config; the README RC1 command is unchanged.

## 3. Path backfill results

- total frontier rows: 52
- exact matches: 50  | ambiguous: 0  | unresolved: 2
- cumulative source coverage: `discovered_theorems`=50, `routing`=50, `tasks`=50, `eval_runs_or_final`=50  (eval_runs files scanned: 531)
- resolved by namespace: `Set`=43, `Multiset`=3, `Eq`=1, `Function`=1, `Prop`=1, `GENERAL_FRONTIER`=1
- unresolved by namespace: `Nat`=1, `traces_from_search`=1
- examples unresolved: ['Nat.lt_of_lt_of_le', 'traces_from_search.jsonl']
- batch runnability: {'sf1_balanced_mini': {'size': 8, 'with_path': 6, 'runnable': True}, 'sf1_multiset_holdout': {'size': 3, 'with_path': 3, 'runnable': True}, 'sf1_frontier_all': {'size': 52, 'with_path': 50, 'runnable': True}}

## 4. Runnable theorem sets generated

| name | size | source batch | dropped (no path) | registered | schema |
|---|---|---|---|---|---|
| `sf1_multiset_holdout_runnable` | 3 | `sf1_multiset_holdout` | 0 | runtime (False) | {set_name: [{file_path, full_name, namespace}]} |
| `sf1_balanced_mini_runnable` | 5 | `sf1_balanced_mini` | 2 | runtime (False) | {set_name: [{file_path, full_name, namespace}]} |
| `sf1_frontier_runnable_subset` | 5 | `sf1_frontier_all` | 2 | runtime (False) | {set_name: [{file_path, full_name, namespace}]} |

Files under `project/evolve/experiments/sf1/theorem_sets/`. Registered at run time by `sf1_run_eval.py` (no on-disk registry mutation).

## 5. Eval results

- eval script present=`True` help_ok=`True` flags=['--theorem-set', '--policy-type', '--route-config', '--strategy-config', '--ckpt-dir', '--top-k', '--max-steps', '--out-dir'] | wrapper present=`True`
- ns9 config: `project/evolve/ns9_runs/baseline/strategy_config.json` | path_map size: 50 | run_real=`True`
- outcomes: ran=1 failed=0 blocked=0 unsupported=0
  - `sf1_multiset_holdout_runnable`/`rc1`: status=**ran** rc=0 solved=0/3 metrics=`project/evolve/experiments/sf1/out/real/eval/rc1_smoke_sf1_multiset_holdout_runnable/eval-4e872b94/metrics.json`

- per-theorem (real):
  - `Multiset.toFinset_nsmul` solved=`False` tactic=`aesop`
  - `Multiset.toFinset_eq_singleton_iff` solved=`False` tactic=`None`
  - `Multiset.disjoint_toFinset` solved=`False` tactic=`induction m1 using Multiset.induction_on <;> simp_all`

- exact commands: `project/evolve/experiments/sf1/out/real/eval_commands.sh`
- limitation: live eval needs LeanDojo + the traced Mathlib cache + base policy ckpts (via the NS24 route-config). Where the environment lacks these, the failure is recorded verbatim.

## 6. Relabel queue update

- queue size: 33  | per-theorem eval available: `True`
- by priority: `high`=3, `medium`=30
- by candidate family: `wx3_multiset_induction`=3, `future_failure_driven_lemma_candidate`=30
- high-priority decls: ['Multiset.disjoint_toFinset', 'Multiset.toFinset_eq_singleton_iff', 'Multiset.toFinset_nsmul']
- real RC1 per-theorem status now feeding queue: `True`. No wrapper win is claimed; candidates remain pending NS23 minimal-sufficient relabel.

## 7. Promotion decision

- recommendation: **MINE_MORE**. RC2 promotion remains gated on positive delta over RC1, zero regressions, zero off-gate emissions, minimal-sufficient attribution, deterministic reproduction.

## 8. Implication for SF2

- Real RC1 **failures** now exist on an SF1 set → these seed **SF2 Failure Pattern Miner** (RC1 failures → recurring goal patterns → missing-lemma templates).

## 9. Protected files

`git diff --stat HEAD` for protected configs (empty = unchanged):

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
?? project/evolve/reports/sf1_stage_cdef_status.md
?? scripts/sf1_backfill_frontier_paths.py
?? scripts/sf1_classify_frontier.py
?? scripts/sf1_common.py
?? scripts/sf1_eval_matrix.py
?? scripts/sf1_extract_mathlib_catalog.py
?? scripts/sf1_filter_consumed_surfaces.py
?? scripts/sf1_make_batches.py
?? scripts/sf1_minimal_relabel_new_wins.py
?? scripts/sf1_promotion_report.py
?? scripts/sf1_run_eval.py
```

No commit made. All changes additive; RC1 wrapper, NS9 genome, NS24 router, REL1 reports unmodified.
