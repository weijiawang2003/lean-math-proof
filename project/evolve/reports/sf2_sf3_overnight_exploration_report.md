# SF2/SF3 Overnight Exploration — Truth Repair, Singleton-Iff Deep Dive, Frontier Expansion

- branch `rc1-production-stack` · no production config change · no commit · all live results reproduced from disk.

## 1. Executive summary

- **Corrected Multiset result:** RC1 solved **2/3** of the holdout (`toFinset_nsmul` via aesop; `disjoint_toFinset` via the WX3 induction oracle), 1 genuine failure. The old "0/3" was a metrics-parser bug (read `proof_finished`/`solved`; the real key is `finished`).
- **Singleton-iff genuine failure** `Multiset.toFinset_eq_singleton_iff`: the Part-3 deep ladder closed it **0/13** live — NO new probe solved it ({'proof_failed': 6, 'max_recursion': 6, 'parse_error': 1}).
- **No probe solved the genuine failure.** It needs a multi-step count-extensionality proof; every one-shot `simp_all[...]` (incl. source-inspired lemma sets) hits max recursion, and the WX3 induction oracle strictly hurts.
- **Broader frontier eval:** 3 theorem-set runs recorded (see §4).
- **Genuine RC1 failures found:** 18; clusters: 10.
- **SF3 candidate-lemma queue size:** 1 (key finding: the singleton failure needs NO new lemma — it is a tactic/routing gap).

## 2. Ground-truth correction

`scripts/sf1_eval_matrix.py:parse_metrics` keyed solved on `proof_finished`/`solved`/`proved` (none exist per-theorem); the authoritative key is **`finished`**, and the aggregate uses `proved`/`total_theorems`. Every theorem read as unsolved ⇒ RC1 undercounted as 0/3. Fixed (`_theorem_solved` helper + `--repair-existing-results`); see `sf1_truth_layer_correction.md`. Corrected: **RC1 2/3**. This matters because the WX3 oracle's one held-out win (`disjoint_toFinset`) was being mislabeled a failure.

## 3. Singleton-iff deep dive

- real proof location: `Mathlib/Data/Finset/Basic1th4x_l3.lean (line 2977)` (resolves for Dojo under `Mathlib/Data/Finset/Basic.lean`).
- statement: `s.toFinset = {a} ↔ card s ≠ 0 ∧ s = card s • {a}`
- official proof is a **count-extensionality** argument (`refine` iff-split → `ext'` → `by_cases x = a` → `count_*` rewrites; reverse via `toFinset_nsmul`, `toFinset_singleton`). **No induction.**
- verdict: **reusable_probe (iff-split opener) + too_specialized (count-extensionality closure); NOT a missing lemma**
- residual under the RC1 WX3 oracle: `insert a✝ s✝.toFinset = {a} ↔ a✝ ::ₘ s✝ = (card s✝ + 1) • {a}` (induction makes it worse).
- **probe ladder results (live, single Dojo):**
  - `proof_failed` — `simp`
  - `proof_failed` — `simp_all`
  - `max_recursion` — `constructor <;> intro h <;> simp_all`
  - `max_recursion` — `refine ⟨fun H => ?_, fun H => ?_⟩ <;> simp_all`
  - `proof_failed` — `simp only [Finset.eq_singleton_iff_unique_mem, Multiset.mem_toFi`
  - `max_recursion` — `constructor <;> intro h <;> simp_all [Finset.eq_singleton_iff_un`
  - `proof_failed` — `simp only [Multiset.mem_toFinset, Multiset.mem_singleton]`
  - `max_recursion` — `constructor <;> intro h <;> simp_all [Multiset.mem_toFinset, Mul`
  - `proof_failed` — `simp [Multiset.toFinset_nsmul, Multiset.toFinset_singleton, Mult`
  - `max_recursion` — `constructor <;> intro h <;> simp_all [Multiset.toFinset_nsmul, M`
  - `max_recursion` — `constructor <;> intro h <;> simp_all [Multiset.toFinset_nsmul, M`
  - `proof_failed` — `induction s using Multiset.induction_on <;> simp_all`
  - `parse_error` — `refine ⟨fun H => ⟨fun h => ?_, ?_⟩, fun H => ?_⟩ /   · rw [h, toFi`
- **tactic-gap or lemma-gap?** TACTIC/ROUTING gap. All dependencies exist; the gap is (a) split the iff before ext, (b) multi-step count reasoning a single battery tactic cannot do.

## 4. Expanded frontier eval

| theorem_set | policy | solved/total or status |
|---|---|---|
| sf1_multiset_holdout_runnable | rc1 | 2/3 |
| sf1_balanced_mini_runnable | rc1 | 2/6 |
| sf1_frontier_runnable_subset | rc1 | 5/20 |

> Caveat: SF1 frontier `file_path`s are heuristic namespace→path backfill and the frontier contains mining-artifact rows (e.g. `traces_from_search.jsonl`, `Prop.compl_singleton`). Theorems that fail to resolve are environment/path failures, NOT proof failures — separated in §5.
Commands: `project/evolve/experiments/sf2/out/frontier_expansion/eval_commands.sh` (if emitted) and `scripts/sf1_eval_matrix.py --run-real --policies rc1`.

## 5. Failure clusters

- failures: 20 (genuine 18, junk/unresolved 2); clusters: 10.

| priority | cluster_id | size | capability | next |
|---|---|---|---|---|
| high | `Set|future_failure_driven_lemma_candidate|iff|all_tactics_errored` | 4 | iff decomposition before simp/ext | probe |
| high | `Set|broad_set_aesop_rejected|equality|all_tactics_errored` | 3 | needs source-proof inspection | probe |
| high | `Set|rc1_production_stack|membership|all_tactics_errored` | 3 | needs source-proof inspection | probe |
| high | `Set|rc1_production_stack|equality|all_tactics_errored` | 3 | needs source-proof inspection | probe |
| high | `Set|future_failure_driven_lemma_candidate|membership|all_tactics_errored` | 2 | needs source-proof inspection | probe |
| high | `Multiset|wx3_multiset_induction|equality|all_tactics_errored` | 1 | Multiset induction routing (avoid on membership/iff goals) | probe |
| high | `Function|future_failure_driven_lemma_candidate|equality|all_tactics_errored` | 1 | needs source-proof inspection | probe |
| high | `Set|future_failure_driven_lemma_candidate|equality|all_tactics_errored` | 1 | needs source-proof inspection | probe |
| low | `Prop|future_failure_driven_lemma_candidate|equality|unresolved_or_junk` | 1 | needs source-proof inspection | ignore |
| low | `Eq|future_failure_driven_lemma_candidate|equality|unresolved_or_junk` | 1 | needs source-proof inspection | ignore |

## 6. Relabel queue update

- NS23 minimal-sufficient relabel targets: the corrected Multiset wins (`toFinset_nsmul`, `disjoint_toFinset`) — confirm whether plain `simp`/`omega` also closes them, and the open singleton failure's probe family. See `relabel_queue_updated.jsonl`.
- source-inspired (not yet a claim): the split-iff-then-ext probe family.
- rejected as too theorem-specific: the count-extensionality closure of the singleton theorem.

## 7. SF3 candidate-lemma queue

- size: 1. Honest summary: The official proof (Finset/Dedup.lean:117) uses ONLY pre-existing Mathlib lemmas (toFinset_zero, singleton_ne_empty, mem_toFinset, mem_singleton, toFinset_nsmul, toFinset_singleton). Therefore there is NO genuinely missing lemma — candidate lemmas 1-4 are duplicates/corollaries of existing results. The only non-duplicate, promising item is a TACTIC FAMILY (candidate 5), tested live in Part 3, not a new lemma.
  - `singleton_iff_NO_MISSING_LEMMA` — novelty: duplicate (all deps exist); priority: low_as_lemma_high_as_routing_fix

## 8. Recommendation

- **Do not modify RC1.** No promotion is justified by this run.
- The singleton failure is a genuine open failure with a clear diagnosis (tactic/routing gap, not a missing lemma). If a future multi-step search or a split-iff routing tweak closes it, test on additional singleton/toFinset membership-iff failures and run NS23 minimal-sufficient relabel before any promotion.
- Highest-priority next target: the largest high-priority cluster in §5; the singleton theorem itself needs depth-≥4 search, not a battery tactic.
- Fold the `finished`-key fix into all future SF eval so RC1 is never undercounted.

## 9. Protected-file confirmation

`git diff --stat HEAD` for protected configs (empty = unchanged):
```
(no changes to rc1_production_wrapper.json or ns24_router.json)
```
`git status --short` (working tree; all additive SF1/SF2/SF3 artifacts + scripts):
```
M README.md
?? project/evolve/experiments/sf1/
?? project/evolve/experiments/sf2/
?? project/evolve/experiments/sf3/
?? project/evolve/reports/sf1_design.md
?? project/evolve/reports/sf1_live_eval_unblocker_status.md
?? project/evolve/reports/sf1_promotion_report.md
?? project/evolve/reports/sf1_stage_ab_status.md
?? project/evolve/reports/sf1_stage_cdef_status.md
?? project/evolve/reports/sf1_truth_layer_correction.md
?? project/evolve/reports/sf2_multiset_seed_report.md
?? project/evolve/reports/sf2_sf3_overnight_exploration_report.md
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
?? scripts/sf2_build_failure_cases.py
?? scripts/sf2_cluster_failures.py
?? scripts/sf2_extract_source_context.py
?? scripts/sf2_run_probe_ladder.py
?? scripts/sf3_build_queues.py
?? scripts/sf3_make_overnight_report.py
?? scripts/sf3_run_singleton_iff_probes.py
```
No commit made.