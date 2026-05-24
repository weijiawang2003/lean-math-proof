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

The same `gen_v5` checkpoint, no retraining, on Mathlib `Nat/Defs.lean`
and a larger second set:

| policy                                       | medium (38) | large (65) | wallclock |
|----------------------------------------------|------------:|-----------:|----------:|
| `gen_v5` plain (baseline)                    | 3/38 (7.9%) | —          | ~2.5 min |
| **`hybrid_evolved` NS9 best (17 skeletons)** | **37/38 (97.4%)** | **49/65 (75.4%)** | ~2.5 / ~5 min |

Δ vs raw model on medium: **+34 theorems, 0 regressions**. The single
residual medium failure (`Nat.AM_GM`) is a model-capability ceiling,
not a skeleton-bag one.

The strategy wrapper composes the gen_v5 model's beam-search top-K with a
small evolved layer of shape-slotted priority templates, theorem-name-aware
family tactics, fallbacks, and shape-aware premise retrieval gated
independently of family-tactic survival (NS9). All emissions are unified
through a `SkeletonBag` (NS4) so each entry has a stable identity (NS7)
and the wrapper's ranked-list output is fully deterministic given a state.

See `project/evolve/reports/skeleton_evolution_executive_summary.md` for
the short version, `skeleton_evolution_final_report.md` for the full
v3→NS9 progression, and `project/evolve/best/README.md` for the current
best genome and reproduce command.

### Earlier milestones

For the chronological record: v3.6 proved 25/38 with a flat fallback/
template + family-tactic genome (no skeleton-bag, no retrieval). NS4
(skeleton-bag refactor) preserved 25; NS5–NS9 compressed the genome
from 48 to 17 enabled skeletons while improving medium to 37/38 and
large to 49/65. See `project/evolve/reports/nat_defs_medium_summary.md`
for the chronological progression.

## Reproducing the NS9 best result

All commands run from the repo root and assume `project/models/gen_v5/`
is present locally (gitignored binary). The best genome ships in-repo
at `project/evolve/best/ns9_best_genome.json`.

```bash
# Medium (~2.5 min, expect 37/38)
python eval_rollout_all.py \
    --theorem-set nat_defs_medium \
    --policy-type hybrid_evolved \
    --ckpt-dir project/models/gen_v5 \
    --top-k 8 --max-steps 8 \
    --strategy-config project/evolve/best/ns9_best_genome.json \
    --out-dir /tmp/ns9_repro_medium

# Large (~5 min, expect 49/65)
python eval_rollout_all.py \
    --theorem-set nat_defs_large_v5 \
    --policy-type hybrid_evolved \
    --ckpt-dir project/models/gen_v5 \
    --top-k 8 --max-steps 8 \
    --strategy-config project/evolve/best/ns9_best_genome.json \
    --out-dir /tmp/ns9_repro_large
```

The proved count lands in `<out-dir>/eval-*/metrics.json` as the
`proved` field. Compare with the baseline `gen_v5` (no wrapper):

```bash
python eval_rollout_all.py --theorem-set nat_defs_medium \
    --policy-type generative --ckpt-dir project/models/gen_v5 \
    --top-k 8 --max-steps 8 --out-dir /tmp/gen_v5_baseline_medium
```

## Reproducing the v3.6 milestone

```bash
python -m evolve.run_evolve --theorem-set nat_defs_medium \
    --generations 0 --population-size 1 --survivors 1 \
    --policy-type hybrid_evolved --ckpt-dir project/models/gen_v5

python -m evolve.report_run \
    --hybrid-run project/evolve/runs/<run_id> \
    --baseline-run /tmp/gen_v5_baseline_medium \
    --output project/evolve/reports/nat_defs_medium_<your_label>.md
```

`--generations 0` evaluates only the seed (no mutations) — what you want
when reproducing the historical v3.6 number rather than running fresh
evolution.

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

## Snapshotting state for ChatGPT

After every local experiment or long Claude Code run, generate a single
compact file with everything needed to bring an outside reviewer (e.g.
ChatGPT) up to speed:

```bash
python scripts/export_context_for_chatgpt.py
```

Writes `project/evolve/reports/chatgpt_context.md` — a ~1–2k-word digest
covering current git state (with a checkpoint-artefact warning if any
model files show as modified), the best result so far on `nat_defs_medium`
and `nat_defs_large_v5`, the most recent autonomous-run cycle, key
reports, open problems pulled from the latest `V5_README.md` / `v5_next_steps.md`,
the changed-file list vs `main`, and a suggested next question to ask
ChatGPT. Reads only small JSON / JSONL / Markdown files — never traces,
subprocess logs, or checkpoint binaries.

## Roadmap

- **v1 (done)** — deterministic mutator, dry-run evaluator.
- **v2 (done)** — wired the real evaluator; subprocess watchdog.
- **v3 (done)** — `hybrid_evolved` wrapper. v3.1 nat-vars; v3.2 budget;
  v3.3 anti-loop; v3.4 family tactics; v3.5 medium scale-out (25/38);
  v3.6 per-theorem deny-list.
- **v4 (done)** — premise retriever in wrapper (v4.1–v4.4); shape gating;
  hypothesis-shape template params; priority_templates (v4.6, 36/38);
  lemma audit jump (v4.7, **37/38**).
- **NS1–NS3.5 (done)** — invariants and wrapper-side fixes:
  specificity ordering, per-theorem failure triage, `any`-as-fallback
  semantics.
- **NS4 (done)** — `SkeletonBag` unified-representation refactor.
  NS4.1 unified family/term-builder/fallback through the bag; NS4.2
  modeled retrieved-premise emissions as dynamic per-state skeletons.
- **NS5 (done)** — archive ledger + six evolutionary mutation operators;
  7.5-hour autonomous sweep compressed 48 → 25 skeletons preserving
  37/49.
- **NS6 (done)** — per-step assist credit; scoped order-changing
  mutations; safe pruning. Compressed 25 → 20.
- **NS7 (done)** — stable skeleton IDs; bag-only pre-flight detector.
  20 enabled; 10 NS6-class regressions still hit Lean.
- **NS8 (done)** — cached `gen_v5` outputs + full ranked-list simulator;
  pre-flight rejects all 10 NS7 Lean regressions; pinned the 20-skeleton
  floor to a single retrieval-gate mechanism.
- **NS9 (done, current)** — `retrieval_requires_family: bool` and
  `retrieval_family_gates: list[str]` decouple retrieval from
  family-tactic survival. Compressed 20 → **17 enabled skeletons**
  preserving 37/49.
- **NS10 (future)** — targeted `gen_v5+1` fine-tune to close
  `Nat.AM_GM` and the ~16 unproved large theorems. The skeleton bag
  is exhausted; the remaining gap is a model-capability issue.
  Multi-step skeleton synthesis (an operator that emits `(rw, rw,
  simp_all)` triples derived from successful retrieval chains in the
  archive) is a complementary direction.
