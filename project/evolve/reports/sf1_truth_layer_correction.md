# SF1 Truth-Layer Correction — `finished`-key metrics parser bug

- branch: `rc1-production-stack` · scope: measurement/data repair only · no production config change · no commit
- date: 2026-05-30

## 1. The bug

`scripts/sf1_eval_matrix.py:parse_metrics` decided per-theorem solved status with:

```python
ok = bool(t.get("proof_finished") or t.get("solved") or t.get("proved"))
```

None of `proof_finished` / `solved` / `proved` exist in the live `metrics.json`
`per_theorem` records. The authoritative per-theorem solved flag is **`finished`**
(bool). Because all three legacy keys were always absent, every theorem read as
`solved=False`, so SF1's `eval_matrix_results.json` recorded **RC1 0/3** on the
Multiset holdout even though the run actually proved 2 of 3.

The aggregate block was also mis-keyed: the live writer emits `proved` +
`total_theorems`, not `num_proved` / `num_theorems`.

### metrics.json schema (authoritative, confirmed from the real run)

- top-level aggregate (scalars/dicts): `total_theorems`, `available`, **`proved`**,
  `errored`, `exhausted`, `skipped`, `success_rate`, `proved_by_origin`, …
- `per_theorem[]`: `full_name`, `file_path`, `available`, **`finished`** (solved
  flag), `has_error`, `num_steps`, `tactics_used`, `tactics_used_origins`,
  `winning_tactic`, `winning_tactic_origin`, `winning_tactic_template_source`,
  `error_message`, …

## 2. Old vs new authoritative result

| metric | OLD (buggy parser) | NEW (`finished` key) | source of truth |
|---|---|---|---|
| `sf1_multiset_holdout_runnable` / RC1 solved | **0 / 3** | **2 / 3** | metrics.json `proved=2`, `errored=1` |
| `Multiset.toFinset_nsmul` | unsolved | **PROVED** (`aesop`, generative_topk) | per_theorem.finished=true |
| `Multiset.disjoint_toFinset` | unsolved | **PROVED** (`induction m1 using Multiset.induction_on <;> simp_all`, wrapper_symbolic_action / `MULTISET_INDUCTION_SIMP[Multiset,simp_all]`) | per_theorem.finished=true |
| `Multiset.toFinset_eq_singleton_iff` | unsolved | **FAILED** (genuine; "All top-13 tactics errored at step 4") | per_theorem.finished=false |

There is therefore exactly **one genuine RC1 Multiset failure**:
`Multiset.toFinset_eq_singleton_iff`. The WX3 `Multiset.induction_on` oracle *fired*
on it and did not close it.

## 3. The fix

`scripts/sf1_eval_matrix.py`:

- new helper `_theorem_solved(t)` → reads `finished` first (authoritative); only
  falls back to `proof_finished`/`solved`/`proved` when `finished` is absent, and
  records a per-theorem `parse_warning` when it does (no more silent undercount).
- `parse_metrics` now also reads the correct aggregate keys (`proved`,
  `total_theorems`), emits `aggregate_solved` / `aggregate_total`, cross-checks the
  per-theorem count against the aggregate, and enriches `per_theorem` with
  `file_path`, `num_steps`, `winning_tactic_origin`, `error_message`.
- a SCHEMA NOTE comment documents the keys so this never regresses.
- new `--repair-existing-results` mode (`repair_existing_results()`): re-derives
  `solved`/`total`/`per_theorem` for an existing results file using the corrected
  parser **without re-running eval**, preserving `pre_repair_solved` per result.

## 4. Regenerated artifacts (exact commands)

```bash
# repair existing results in place (original preserved as *.prefix_bug.bak.json)
python3 scripts/sf1_eval_matrix.py \
  --repair-existing-results \
  --results project/evolve/experiments/sf1/out/real/eval_matrix_results.json \
  --out    project/evolve/experiments/sf1/out/real/eval_matrix_results.corrected.json \
  --in-place

# regenerate the minimal-relabel queue + summary from corrected results
python3 scripts/sf1_minimal_relabel_new_wins.py
```

Files written / changed:

- `project/evolve/experiments/sf1/out/real/eval_matrix_results.json` — corrected
  in place (RC1 now solved 2/3; each result carries `pre_repair_solved`,
  `repaired:true`, `parse_warnings`).
- `project/evolve/experiments/sf1/out/real/eval_matrix_results.prefix_bug.bak.json`
  — verbatim copy of the pre-fix file (history preserved, not silently rewritten).
- `project/evolve/experiments/sf1/out/real/eval_matrix_results.corrected.json` —
  explicit corrected copy (matches the in-place file).
- `project/evolve/experiments/sf1/out/real/relabel_queue.jsonl` /
  `relabel_summary.json` — regenerated from corrected per-theorem status.

## 5. Effect on interpretation / decisions

- **Genuine RC1 Multiset failures: 3 → 1.** The two former "failures"
  (`toFinset_nsmul`, `disjoint_toFinset`) are real RC1 **wins**; one of them is a
  WX3-oracle win, confirming the production WX3 Multiset induction action is
  pulling its weight on at least one held-out theorem.
- **Relabel queue corrected.** Multiset entries' `baseline_status` moved from
  `{rc1:false}` (all three) to `{rc1:true}` for `toFinset_nsmul` /
  `disjoint_toFinset` and `{rc1:false}` only for `toFinset_eq_singleton_iff`. The
  three remain high-priority (wx3 score 0.9) but the two wins are now correctly
  framed as NS23 minimal-sufficient *attribution* checks (did plain `simp`/`omega`
  also suffice?), not as failures to fix.
- **No promotion was triggered and none is recommended here.** This is a pure
  measurement repair; the RC1 stack is unchanged.

## 6. Protected-file confirmation

No protected configs were touched. `git diff --stat HEAD` for
`project/evolve/experiments/rc1/rc1_production_wrapper.json` and
`project/evolve/routing/ns24_router.json` is empty (see overnight report §9). The
only code change is `scripts/sf1_eval_matrix.py` (non-protected). No commit made.
