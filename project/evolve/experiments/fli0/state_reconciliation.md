# FLI0 Part 0 — State Reconciliation

_Generated during FLI0 setup. Read-only inspection of the repo; no artifacts modified._

## Current HEAD

```
abd817e (HEAD -> tr5-ranker-guided-live-search, origin/tr5-ranker-guided-live-search)
        Compare validated RC4B and RC4C candidates
```

Recent arc (newest first): `abd817e` RC4BC comparison → `5ef2d1e` RC5V3 safety/maintenance
artifacts → `edd8bef` RC5V3 hardened hybrid scaling benchmark → `cb433e2` RC5H → `fd8cde1` TR7
→ `4f2361f` TR5 → `8f9d08e` RC4B/RC4C → … → `2c0dc72` TR6.

> The compacted prompt that originally opened this thread (describing TR6 as the latest stage)
> was **stale**. The committed project is well past TR6, RC4B/RC4C, RC4D/RC4R, TR5/TR7, and the
> RC5 hybrid line (RC5H → RC5S → RC5V2 → RC5V3).

## Dirty / untracked working tree

`git status --short` shows **142 entries**, all confined to the in-progress RC5V3 run:

- **Modified (tracked):**
  - `scripts/rc5v3_apply_attribution.py`
  - `scripts/rc5v3_run_safe_dynamic_incremental.py`
  - `project/evolve/experiments/rc5_v3/out/rc2_baseline.log`
  - `project/evolve/experiments/rc5_v3/out/rc2_bench/runs/eval-880fcbee/traces.jsonl`
- **Untracked:** the entire `project/evolve/experiments/rc5_v3/{cases,out}/` raw-result set
  (B1/B3/B5 dynamic results + checkpoints, RC2/RC4 baselines, eligibility, plan, retrieval).

**FLI0 does not touch any of this.** RC5V3 raw artifacts are read-only inputs.

## Stage report-status

| stage | final report | status |
|---|---|---|
| RC5V2 | `project/evolve/reports/rc5/rc5v2_hardened_hybrid_fresh_benchmark_report.md` | **PRESENT — complete** |
| RC5V3 | _none_ (`reports/rc5/rc5v3_*` absent) | **PARTIAL_ARTIFACTS_AVAILABLE** |

## RC5V3 artifact status (raw present, analysis missing)

**Present (raw, untracked):** `rc5v3_eval_batch`, `rc5v3_rc2_baseline_results`,
`rc5v3_static_stage_results`, `rc5v3_dynamic_eligible`, `rc5v3_retrieval_results`,
`rc5v3_safe_dynamic_plan`, `rc5v3_b1_dynamic_results`, `rc5v3_b3_dynamic_results`,
`rc5v3_b5_dynamic_results`, `rc5v3_budget_slices`, `rc5v3_dynamic_eligibility_summary`,
`rc5v3_large_fresh_frontier_*`.

**Missing (analysis layer never produced/committed):** `rc5v3_attribution.{json,md}`,
`rc5v3_safety_audit.json`, `rc5v3_system_comparison.json`, `rc5v3_cost_curve.json`,
`rc5v3_namespace_feature_yield.json`, `rc5v3_maintenance_decision.json`,
`rc5v3_dynamic_examples.jsonl`, and the RC5V3 final report.

> **Per task instruction:** RC5V3 final report is missing but raw artifacts exist, so we do
> **not** fabricate an RC5V3 conclusion. RC5V3 = `PARTIAL_ARTIFACTS_AVAILABLE`; FLI0 mines its
> raw per-theorem results directly.

### RC5V3 interruption cause (observed in raw data)

RC5V3 B5 ran into a **network outage**: many B5 records carry
`setup_error: ConnectionError … api.github.com … NameResolutionError` with `live=False`,
`programs_attempted=0`. B1 ran live earlier (108 live records); B5 has only 104 live of 315.
This is infra noise, **not** a math signal — FLI0 separates these `infra_only` cases out of the
clean-failure corpus.

## Failure-universe sizing (computed from raw artifacts)

| | RC5V2 (complete) | RC5V3 (partial) |
|---|---|---|
| dynamic-eligible (CONFIRMED_RC2_FAILURE) | 149 | 318 |
| dynamic successes | 8 | 4 |
| clean live failures | ~141 (6 infra) | **208** (106 infra-only) |
| attribution | committed (`8 FRESH_TRUE_RC5V2_DELTA`, 141 `NO_DYNAMIC_WIN`) | none (derive on the fly) |

**RC5V2 ∩ RC5V3 = 0** — the two stages swept disjoint fresh frontiers, so their failures union
cleanly. Combined clean-failure pool ≈ **343**, far above the 20–40 seed target.

## Safe to read

All RC5V2 `out/`+`cases/` artifacts; all RC5V3 raw `out/`+`cases/` artifacts;
`project/discovered_theorems.json` (dict; `theorems` = 527 records with
`file_path/full_name/has_tactic_proof/num_tactics/difficulty/difficulty_score`).

## Missing / unusable

RC5V3 analysis+report layer (above); per-tactic **residual goal states** are absent from every
stage's artifacts (the dynamic `failures` list records only `{rank, tactic, outcome}`, not the
post-tactic goal) → FLI0 sets `residual_goal_status = MISSING` and reasons from the **theorem
statement + retrieved lemmas + failure outcomes** instead.

## Decision: FLI0 source

**FLI0 uses BOTH RC5V2 (complete) and RC5V3 (partial raw).** RC5V2 supplies a fully-attributed
clean-failure set; RC5V3 supplies a much larger disjoint frontier of live failures. RC5V3 is
treated as `PARTIAL_ARTIFACTS_AVAILABLE` and only its raw per-theorem results are consumed.
