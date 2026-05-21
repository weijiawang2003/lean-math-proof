# evolve/ — an AlphaEvolve-style outer loop for Lean proof search

## What this is

A small evolutionary search layer that sits *outside* the existing pipeline. It
does **not** train neural networks and does **not** modify any checkpoint.
Instead it evolves **proof-search strategies** — configurations that wrap the
rollout machinery.

The conceptual move, borrowed from AlphaEvolve:

```
AlphaEvolve : don't search for one mathematical object;
              search for a program that generates the object.

LeanEvolve  : don't search for one proof script;
              search for a configuration that generates proof-search behaviour.
```

Lean is the evaluator — the same strict, no-partial-credit supervisor the rest
of the project is built around.

## Main result (v3.6)

The same `gen_v5` checkpoint, no retraining, no premise retrieval, on the
same 38 `Mathlib/Data/Nat/Defs.lean` theorems:

| policy                         | proved | rate  | wallclock |
|--------------------------------|-------:|------:|----------:|
| `gen_v5` plain (baseline)      |  3/38  |  7.9% |    ~2.5 min |
| `hybrid_evolved` (this loop)   | **25/38** | **65.8%** |  ~3.5 min |

Δ = **+22 theorems, 0 regressions** (every baseline win is also a hybrid win).
The strategy wrapper composes the gen_v5 model's beam-search top-k with a
deterministic, evolved layer of generic fallback tactics, theorem-name-aware
family tactics (`div`, `mod`, `AM_GM`), and a 1-entry per-theorem deny-list
that suppresses one `(theorem, tactic)` pair known to crash the Dojo REPL.

Win attribution (`proved_by_origin`):
- `fallback_tactic` (`omega`, `simp_all`, …): 18
- `family_tactic` (mod family): 4
- `generative_topk` (gen_v5 model directly): 3

The wrapper generalizes: the 23 unseen theorems added in `nat_defs_medium`
proved at 65.2% — within 1.5 pp of the 15-theorem subset the strategy was
evolved on. See `project/evolve/reports/nat_defs_medium_summary.md` for the
progression v3 → v3.6 and `project/evolve/reports/nat_defs_medium_v3_6.md`
for the paired-comparison run report.

## Reproducing the main result

All commands run from the repo root and assume `project/models/gen_v5/` is
present locally (gitignored binary).

```bash
# 1. Baseline — gen_v5 alone, no wrapper. ~2.5 min on CPU.
python eval_rollout_all.py --theorem-set nat_defs_medium \
    --policy-type generative --ckpt-dir project/models/gen_v5 \
    --top-k 8 --max-steps 8 --out-dir /tmp/gen_v5_baseline_medium

# 2. Hybrid — the v3.6 evolved wrapper. ~3.5 min on CPU.
python -m evolve.run_evolve --theorem-set nat_defs_medium \
    --generations 0 --population-size 1 --survivors 1 \
    --policy-type hybrid_evolved --ckpt-dir project/models/gen_v5

# 3. Paired Markdown report comparing the two runs.
python -m evolve.report_run \
    --hybrid-run project/evolve/runs/<run_id_from_step_2> \
    --baseline-run /tmp/gen_v5_baseline_medium \
    --output project/evolve/reports/nat_defs_medium_<your_label>.md
```

`--generations 0` evaluates only the seed (no mutations) — what you want when
reproducing the published number rather than running fresh evolution.

## The loop

```
  heuristic/LLM mutator        proposes new candidate strategies
            |
            v
  SearchCandidate              policy, ckpt, top_k, max_steps,
                               fallback tactics, prompt, templates
            |
            v
  evaluator                    dry-run fake metrics  OR  eval_rollout_all.py
            |
            v
  scoring + selection          score_metrics() -> select_top()
            |
            v
  next generation              mutate the survivors, repeat
```

## Files

| file                  | role                                                                            |
|-----------------------|---------------------------------------------------------------------------------|
| `candidate.py`        | `SearchCandidate` — the genome (one proof-search strategy)                       |
| `scoring.py`          | `EvalMetrics` + `score_metrics()` scalar fitness                                |
| `population.py`       | `CandidateRecord`, JSONL append/load, `select_top()`                            |
| `evaluator.py`        | `evaluate_candidate()` — dry-run metrics or real Lean eval (subprocess + watchdog) |
| `mutator.py`          | `mutate_candidate()` — deterministic local mutations                            |
| `strategy_wrapper.py` | `StrategyWrapperPolicy` — base policy + fallbacks + family tactics + deny-list   |
| `run_evolve.py`       | CLI that drives the generational loop                                           |
| `report_run.py`       | CLI that turns a run dir into a Markdown report (paired comparison supported)    |

## Quick start (dry-run, no Lean)

Run from the repo root:

```bash
python -m evolve.run_evolve --dry-run --generations 2 --population-size 4 --survivors 2
```

A fuller dry-run on the curriculum:

```bash
python -m evolve.run_evolve --dry-run --theorem-set curriculum_all \
    --generations 5 --population-size 8 --survivors 3
```

In dry-run mode the evaluator returns **deterministic fake metrics** derived
from each candidate's config (peaks at moderate `top_k`/`max_steps`, rewards a
richer fallback/template genome). This is enough to exercise mutation,
scoring, selection and the leaderboard with zero Lean, GPU or API cost.

## Outputs

- `project/evolve/population.jsonl` — every evaluated candidate, one JSON line
  (`generation`, `candidate`, `metrics`, `score`, `run_dir`). Accumulates across
  runs; each line carries its `evolve_run_id` in `candidate.metadata`.
- `project/evolve/runs/<run_id>/` — per-run `config.json`, `summary.json`,
  `best_candidate.json`.

## Connecting to real Lean evaluation

Drop `--dry-run` to make `evaluator.py` shell out to `eval_rollout_all.py` and
parse the `metrics.json` it writes. The wiring is now end-to-end:

1. **CLI mapping** — `evaluate_candidate` passes `--theorem-set`,
   `--policy-type`, `--ckpt-dir`, `--top-k`, `--max-steps`, `--out-dir` and,
   for `hybrid_evolved`, `--strategy-config` pointing at a JSON dump of the
   candidate's `fallback_tactics`, `tactic_templates`,
   `max_extra_tactics_per_state`, `theorem_family_tactics`, `family_budgets`
   and `theorem_tactic_denylist`. Optional `--enable-loop-avoidance` when
   the candidate opts in.

2. **Subprocess watchdog.** Each eval is run as a subprocess with a derived
   timeout (`timeout_per_theorem × n_theorems × 1.05 + 60s`); on timeout
   the process is killed and a `timeout_count = n_theorems` penalty is
   recorded so the score function crushes the candidate.

3. **Outputs.** `metrics.json` carries the standard counts plus
   `proved_by_origin`, `family_activation_counts`, `family_proved_counts`,
   `family_activated_theorems`, `denied_tactic_total`, and per-theorem
   rows with `winning_tactic_origin` / `winning_tactic_family_source` /
   `activated_families` / `denied_tactic_count`. `traces.jsonl` tags each
   step with `tactic_origin`, `tactic_template_source`,
   `tactic_family_source` and (when anti-loop is enabled) `state_hash_*`
   and `loop_detected`.

## Roadmap

- **v1 (done)** — deterministic mutator, dry-run evaluator. Loop works end-to-end.
- **v2 (done)** — wired the real evaluator; added `nat_defs_subset` theorem set;
  subprocess watchdog; pre-flight checkpoint check.
- **v3 (done)** — `hybrid_evolved` strategy-wrapper policy. `fallback_tactics`
  and `tactic_templates` now affect Lean evaluation. v3.1 added per-state Nat
  variable extraction; v3.2 added per-state extras budget + EXHAUSTED
  diagnostics; v3.3 added an opt-in anti-loop / state-aware ranking pass
  (default off — kept as a diagnostic, not enabled by default since it
  doesn't change which theorems close).
- **v3.4 (done)** — theorem-name-aware family tactics (`div`, `mod`, `AM_GM`)
  with per-family budgets; `tactic_family_source` tracing.
- **v3.5 (done)** — library cleanup (drop unknown-tactic AM_GM entries,
  remove `<;>`-chained combinators); scale-out to `nat_defs_medium` (38);
  first generalization result: 25/38 vs baseline 3/38.
- **v3.6 (done, current)** — per-theorem tactic deny-list eliminates the
  residual `DojoCrashError` without removing the winning tactic globally;
  paired-comparison `report_run` script; experiment progression writeup.
- **v4 (deferred)** — premise retriever as a new tactic source within the
  wrapper (most likely lever for the `div` family). LLM mutator to recombine
  the best candidates. Larger theorem sets.
