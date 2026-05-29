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

## Main result — wrapper (NS9) + Learn track (NS15)

Two complementary results sit on top of the same `gen_v5`
training corpus. NS9 is search-only: a 17-skeleton wrapper
around the un-modified `gen_v5` checkpoint. NS15 is the Learn
track: a routed pair of fine-tunes (`gen_v5_ns15_nat_oversample`
for Nat, `gen_v5_ns12_balanced` for Set/Finset) distilled from
the wrapper's successful traces.

| policy                                       | medium (38) | large (65) | demo_v1 (15) |
|----------------------------------------------|------------:|-----------:|-------------:|
| `gen_v5` plain (baseline)                    | 3/38 (7.9%) | —          | 10/15        |
| **`hybrid_evolved` NS9 best (17 skeletons)** | **37/38 (97.4%)** | **49/65 (75.4%)** | 11/15 |
| **NS15 routed raw model (no wrapper)**       | **23/38 (60.5%)** | **35/65 (53.8%)** | **10/15** |
| NS9 wrapper composed on top of NS15 routed   | 37/38 | 49/65 | 11/15 |

Δ vs raw baseline on medium: **+34 with the wrapper, +20 from
training alone**. The NS15 raw lift (3 → 23) came from
fine-tuning on a single 8-pair iff-omega pattern harvested from
NS14 wrapper traces — the smallest possible homogeneous pool
that produced clean transfer.

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

### Post-catalog-extension result — NS22 Int omega branch

NS15 remains the main **Nat** raw-model result. After NS20 the Learn
track looked converged, but NS20 was **old-catalog exhaustion, not
framework exhaustion**: CX1 extended the catalog 527 → 1,817 theorems
across 8 namespaces and immediately re-surfaced trainable signal. CX2
then mined a fresh **Int** surface (26% wrapper-only strike rate) and
NS22 distilled it. NS22 is the main **post-CX result for Int transfer**.

| layer / policy              | Nat medium | Nat large | demo_v1 | Int suite                  |
|-----------------------------|-----------:|----------:|--------:|----------------------------|
| `gen_v5` plain              |       3/38 |         — |   10/15 | —                          |
| NS9 wrapper                 |      37/38 |     49/65 |   11/15 | —                          |
| NS15 routed raw             |      23/38 |     35/65 |   10/15 | —                          |
| NS21 routed raw             |      23/38 |     35/65 |   10/15 | Finset local gains         |
| **NS22 routed raw**         |  **23/38** | **35/65** |   10/15 | **57 vs 35 NS12 baseline** |
| NS9 wrapper + NS22 router   |      37/38 |     49/65 |   11/15 | wrapper adds ~0 on Int     |
| NS24 routed raw             |      23/38 |     35/65 |   10/15 | 58 (NS22 + 1, marginal)    |
| NS9 wrapper + NS24 router   |      37/38 |     49/65 |   11/15 | wrapper adds ~0 on Int     |

NS22 routes `^Int\.` to `gen_v5_ns22_int_fallback_omega_5x` and adds
**+22 raw Int wins (35 → 57) vs the NS12 baseline across the CX1+CX2
Int suite**, while preserving every Nat / Set / Finset / demo baseline
exactly. The branch trained on the short `omega` tactic was intended as
an ablation but became the chosen route: it solved 13/13 of the
`fallback_omega` pool and 9/10 of the `iff_omega` pool **without ever
seeing an iff_omega theorem in training**. The long, structured iff-pair
tactic (`exact ⟨fun h => by omega, fun h => by omega⟩`) did not memorize
at 60M-param scale; the short `omega` tactic is the transferable signal.

The research principle that follows: **train on the shortest sufficient
tactic family, not merely the wrapper-attributed family.** NS9's win
attribution can award a goal to the iff-pair template when plain `omega`
would also have closed it, so future mining should aggregate
`iff_omega_pair` and `fallback_omega` into one minimal `omega` family.
See `project/evolve/reports/post_cx1_ns21_cx2_ns22_update.md`.

NS23 then **proved** that aggregation (9/10 iff_omega Int theorems are
`omega`-minimal; aggregate = 22 unique), and NS24 trained on it. The
NS24 result is a **confirmatory near-null**: the repaired minimal-omega
labels reproduce NS22 (57 → 58/156 Int, +1) rather than reaching the
hoped-for 65–70+. The relabeled iff group is solved 9/9 by **both** NS22
and NS24 — NS22's ablation had already absorbed the `omega` policy, so
the repaired labels had nothing left to teach. The lesson: minimal-tactic
relabeling is the right *attribution/gating* step (it would have saved
NS22's failed iff_5x/iff_10x runs) but only adds wins on a family the
base model has **not** already absorbed. The NS24 router is promoted as a
marginal best; the omega surface is now saturated. See
`project/evolve/reports/ns24_int_minimal_omega_training_report.md`.

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
- **NS10 (done)** — proof-of-concept Learn step. Fine-tuned
  `gen_v5` on 152 wrapper-success pairs harvested from NS9
  runs, producing `gen_v5_plus1`: 3/38 → 4/38 on medium. Lift
  was tiny but confirmed the Learn step works in principle.
- **NS11 (done)** — Learn scale-up. Three datasets
  (`medium` 152 / `combined` 5,729 / `coverage` ~6k). Best
  raw model `gen_v5_ns11_combined`: 3/38 → 9/38 on medium,
  but cost 2/15 demo_v1.
- **NS12 (done)** — anti-forgetting. `gen_v5_ns12_balanced`
  restored 10/15 demo_v1 while keeping 5/38 medium and 6/65
  large. Pareto tradeoff between Nat-specialized and
  Set/Finset-balanced sub-models surfaced clearly.
- **NS13 (done)** — stateless `RoutedGenerativePolicy` matches
  full_name against regex routes. Achieves the oracle union
  across NS11/NS12 checkpoints with zero regressions.
- **NS14 (done)** — fresh theorem surface. Mined 70 fresh
  theorems from `project/discovered_theorems.json`, ran the
  NS9 wrapper, harvested **30 wrapper-only pairs across 24
  theorems** (27% yield vs NS11's 1.1%).
- **NS15 (done, current Learn-track main result)** — trained
  `gen_v5_ns15_nat_oversample` on NS11 + NS14 with 10× over-
  sampling of the iff-omega NS14 rows. NS15 routed (Nat →
  oversample, Set/Finset → ns12_balanced) achieves **23/38
  medium, 35/65 large, 10/15 demo_v1 raw** — a 6.6× lift over
  the gen_v5 baseline from a single homogeneous 8-pair pool.
- **NS16 (done)** — negative-transfer arc. 19 heterogeneous
  wrapper-only Nat rows produced ZERO additional transfer
  over NS15. Confirmed that pool homogeneity, not pool size,
  drives Learn-step success.
- **NS17 (done)** — pattern-family audit. No homogeneous
  family met the NS18 training gate on 114 fresh theorems.
- **NS18 (done)** — wrapper-expansion arc. Six experimental
  wrapper configs probed; 5 truly-new wrapper-only wins
  beyond NS9 across two families
  (aesop on Finset: 3 wins; simp_all on Nat-arith: 2 wins).
  One −1 Set regression introduced.
- **NS19 (done)** — namespace-gated wrapper. New
  `theorem_name_tactic_gates` field on `StrategyWrapperPolicy`
  (filters wrapper-injected entries only — never base-model
  output). Gated aesop variant added +1 Finset/aesop win
  (`Finset.coe_cons`) and eliminated the NS18 Set regression.
  Pool: 4 unique Finset/aesop wins; 2 unique simp_all-Nat
  wins. Catalog-exhaustion finding: 208/208 Nat theorems
  already covered by existing sets.
- **NS20 (done, mining exhaustion)** — evaluated the gated
  aesop variant against the full 74-theorem Finset catalog
  remainder. **0 new wins.** Pool stays at 4 unique.
  Preservation confirmed on all benchmarks (49/65 large_v5,
  37/38 medium, 11/15 demo_v1). The Learn track had converged
  against the *current* Mathlib catalog and 8-step search
  budget — old-catalog exhaustion, not framework exhaustion
  (CX1 reopened the loop).
- **CX1 (done)** — Mathlib catalog extension. Scanned 44 new
  source files and grew the catalog **527 → 1,817 available
  theorems across 8 namespaces** (Int / Option / Bool entirely
  new). A limited eval probe surfaced 6 truly-new wrapper-only
  wins; the **aesop/Finset pool reached 6 unique** (all winning
  tactic `aesop`: `coe_insert`, `cons_eq_insert`,
  `disjUnion_singleton`, `coe_cons`, `card_insert_eq_ite`,
  `image_id`), meeting the NS21 training gate.
- **NS21 (done)** — Finset/aesop imitation training. Best
  checkpoint `gen_v5_ns21_finset_aesop_20x`, routed on `^Finset\.`.
  **Honest memorization / narrow imitation:** 5/6 pool theorems
  solved raw with `aesop`, but **0 held-out gains** because NS12
  already emitted `aesop` on held-out Finset surfaces. Local
  Finset gains only (`ns17_finset_extra` 12→15,
  `cx1_finset_image_filter` 28→30). All Nat/Set/demo routed-raw
  and wrapper baselines preserved exactly.
- **CX2 (done)** — Int iff/omega mining. Extended the Int
  catalog **120 → 216 candidates**; 78 fresh after exclusions.
  **Wrapper-only strike rate 20/78 = 26%** (the NS9 wrapper alone
  produced the signal; no experimental wrapper needed). Two
  homogeneous gates met: **`iff_omega_pair` / Int at 10 unique**
  (all `exact ⟨fun h => by omega, fun h => by omega⟩`) and
  **`fallback_omega` / Int at 13 unique** (all `omega`).
- **NS22 (done, current post-CX main result)** — Int omega
  training. The router sends `^Int\.` to
  `gen_v5_ns22_int_fallback_omega_5x` — intended as an ablation,
  it became the chosen route. **+22 raw Int wins (NS12 baseline
  35 → 57)** across the CX1+CX2 Int suite. `omega_5x` solved
  13/13 of the `fallback_omega` pool and 9/10 of the `iff_omega`
  pool *without seeing iff_omega theorems in training*. The long
  iff-pair tactic is unlearnable at 60M-param scale; the short
  `omega` tactic is the transferable signal. All Nat/Set/Finset/
  demo routed-raw and wrapper baselines preserved exactly.
- **NS23 (done)** — minimal-tactic relabeling / attribution repair.
  Re-ran all 32 wrapper-only-vs-NS9 wins through a 12-tactic
  minimal-sufficient battery. **9/10 Int `iff_omega_pair` theorems
  are `omega`-minimal** — the iff-pair template merely won the NS9
  ordering race. Under repaired labels the Int **omega aggregate =
  22 unique** (21 `omega` + 1 `constructor <;> omega`), the largest
  homogeneous training surface across all arcs. Confirms NS22's
  "cross-family transfer" was single-family omega absorption.
- **NS24 (done, current — marginal)** — Int minimal-omega aggregate
  training. Trained the 22-theorem repaired pool (variants ×5/×10/
  +constructor/from-ns12) from `gen_v5_ns22_int_fallback_omega_5x`.
  Best `gen_v5_ns24_int_minimal_omega_10x`, routed on `^Int\.`.
  **Near-null result: 57 → 58/156 Int (+1).** The relabeled iff
  group is solved 9/9 by **both** NS22 and NS24 — NS22 had already
  absorbed `omega`, so the repaired labels added nothing. Routed
  Nat/Set/Finset/demo and wrapper baselines preserved exactly
  (23/38, 35/65, 10/15; wrap 37/38, 49/65, 11/15). Promoted as a
  marginal best; the Int omega surface is saturated.
- **CX3 (done — negative, mining only)** — Bool/Option short-tactic
  mining. Audited the fresh surface (Bool/Basic was already exhausted
  by CX1; 86 fresh candidates, ~92% Option), built five theorem sets,
  and probed raw-routed vs NS9-wrapper. **Wrapper-only wins = 0:** the
  default model and the wrapper solve an identical 43/83, so the
  wrapper is a no-op on Bool/Option and there is nothing to distill.
  The mandatory minimal-tactic relabel found the only count-meeting
  headroom is a 13-theorem `cases_simp | Option` pool whose minimal
  tactic is the compound, per-theorem-variable `intros <;> cases <v>
  <;> simp_all` — the structured-tactic class NS22 showed won't
  memorize at 60M (plain simp/aesop genuinely fail on all 13). **No
  short-token training gate met.** The relabel did its job: it
  prevented a likely-null NS25.
- **WX1 (done — positive, wrapper expansion)** — state-aware Option
  cases-wrapper. Rather than train on CX3's structured headroom, added a
  namespace-gated, off-by-default `option_cases_skeletons` block to the
  wrapper that reads the case variable from the proof state and emits
  `cases <var> <;> simp_all`. **+19 new Option wins beyond NS9, zero
  regressions** (option surfaces 42 → 61; Bool control + broader Bool
  config add nothing). Minimal-tactic relabel: all wins are the
  state-aware compound `cases <var> <;> simp` (17-theorem
  `option_cases_simp` family) — **wrapper-ready, not short-token
  SFT-ready**, exactly as CX3 predicted. Preservation is by construction
  (gated; byte-identical ranked lists on Nat/Set/Finset, verified
  empirically: nat_medium 37/38, set 18/30, finset 15/30, 0 emissions).
  Stored as an experiment config under `project/evolve/experiments/wx1/`;
  the NS9 genome is unmodified. Recommendation: promote as a wrapper
  capability, do **not** fine-tune.
- **WX2 (done — positive, wrapper generalization)** — consolidated WX1
  into a promoted Option config and tested whether the state-aware cases
  pattern generalizes. Extended the wrapper with per-type namespace
  gates / family labels / notation matching (all backward-compatible).
  **Preservation:** the promoted Option config retains the full +19
  Option gain with 0 non-Option regressions and 0 emissions outside
  Option (Nat 37/37, Set 18/18, Finset 15/15, demo 11/11 = NS9).
  **Generalization:** the only large fresh cases surface is **List**
  (Option/Bool exhausted, Sum absent, Prod tiny, Multiset a quotient —
  not `cases`-able). On fresh List sets the generalized wrapper adds
  **+10 List wins beyond NS9, zero regressions** (NS9 is a no-op on List
  too); all are the state-aware `cases l <;> simp[_all]` family
  (`list_cases_simp`, 6 unique gate-met) — **wrapper-ready, not
  SFT-ready**. `induction`, Prod, and the Bool control add nothing.
  Combined WX1+WX2 = **+29 wins beyond NS9** (Option +19, List +10), all
  wrapper-ready. NS9 genome unmodified; configs under
  `project/evolve/experiments/wx2/`.
- **AX1 (done — prototype, symbolic action layer)** — abstraction layer
  that factors the state-dependent variable out of the cases tactics so
  the *label* becomes SFT-ready. Added `project/evolve/symbolic_actions.py`
  (typed `SymbolicAction`: CASES_SIMP/INDUCTION_SIMP × Option/List/Bool ×
  simp/simp_all/decide, stable id like `CASES_SIMP[List,simp_all]`) +
  `project/evolve/state_vars.py` (coarse-typed state extractor), and an
  off-by-default `symbolic_actions` wrapper block (origin
  `wrapper_symbolic_action`). The AX1 symbolic config **reproduces WX2
  exactly** (Δ=0 on all 6 sets, 0 regressions, 0 emissions outside gated
  namespaces). A symbolic-label dataset prototype shows the 27 WX1+WX2
  wins — all variable-dependent raw tactics — collapse to **4 stable
  symbolic labels**. Recommendation: this validates **AX2 symbolic-action
  training**. No training in AX1; NS9 genome unmodified.
- **AX2 (done — negative for training, mining/readiness study)** — mined
  the fresh Option/List surface under the AX1 symbolic wrapper to grow the
  symbolic-label dataset, then ran the readiness gate. **Audit:**
  Option/Bool/Sum/Prod are exhausted (0 fresh, even in the 3989-theorem
  discovered scan); the only fresh surface is **List (76)**. **Probe** (3
  disjoint fresh List sets, raw vs NS9 vs AX1-symbolic on `ns24_router`):
  the symbolic wrapper adds **+3 wins beyond NS9, 0 regressions**, but
  minimal relabeling shows all 3 are **multi-step** (`cases l <;> simp_all`
  advances but does not close from init; 2 symbolic-assisted + 1 aesop) —
  **0 clean single-shot symbolic labels added**. The dataset stays at 27;
  readiness = **RED** (<40). The single-shot `cases <;> simp` pattern
  monetizes only the easiest constructor-split lemmas, which WX2 already
  consumed. **Decision: do NOT train AX3; recommend WX3** (Multiset
  quotient-aware action, or multi-step symbolic sequences). The symbolic
  layer stays a search-time wrapper capability. No checkpoints; NS9/AX1
  unmodified. See
  `project/evolve/reports/ax2_symbolic_dataset_expansion_report.md`.
- **WX3 (done — wrapper-ready GREEN; symbolic-learning borderline)** — took
  AX2's advice and opened the **Multiset** surface (**251 fresh available**,
  the largest untapped namespace). Extended the AX1 symbolic layer
  additively with a `Multiset` var-type and two new action types:
  **`MULTISET_INDUCTION_SIMP`** (`induction {var} using Multiset.induction_on
  <;> simp[_all]`) and **`EXT_SIMP`** (`ext x <;> simp[_all]`); AX1
  Option/List rendering unchanged. Five disjoint sets (165 thms), raw vs NS9
  vs ind/ext/comb on `ns24_router`: WX3 adds **+25 wins beyond NS9, 0
  regressions, 0 leakage** — the workhorse is `Multiset.induction_on <;>
  simp_all`. Minimal relabeling: **20 clean single-shot symbolic labels**
  (vs AX2's 0), dominated by `MULTISET_INDUCTION_SIMP[Multiset,simp_all]`
  (18; family aggregate 20). Preservation perfect (demo 11/11, medium 37/37,
  Set 18/18, Finset 15/15; by construction, WX3 base == NS9 genome).
  **Gate A (wrapper-ready) MET; Gate B (symbolic-learning) borderline-met by
  the induction_on family aggregate (20 ≥ 20) with held-out surface.**
  Decision: **promote the WX3 induction wrapper**; AX3 is plausible for the
  first time — expand the held-out Multiset induction surface (~86 unused +
  full induction-shape catalog) to ≥40 / ≥20-single-id, then train AX3. No
  checkpoints; NS9/AX1/AX2 unmodified. See
  `project/evolve/reports/wx3_multiset_quotient_wrapper_report.md`.
- **AX3 (done — first symbolic-action learner, YELLOW/smoke)** — mined the 86
  held-out fresh Multiset theorems under the WX3 induction wrapper and trained
  the program's **first learned symbolic-action predictor** (not raw tactic
  SFT). Held-out mining added **+7 WX3-only wins → 6 new clean single-shot
  labels**; combined with WX3 → **26 clean symbolic labels** (23 `simp_all`, 3
  `simp`). Readiness = **YELLOW** (25–39 total, dominant ≥20, held-out split
  exists; < 40 Green). Learner = TF-IDF(char 3–5) + balanced logistic
  regression over the proof state, classes = 2 Multiset action ids + NULL:
  **3-fold CV positive recall 0.85, NULL FP 0.05, non-Multiset control FP 0.02
  (0 effective after the namespace gate)**. Offline predictor-vs-oracle (the
  action is additive+gated, so the predictor only suppresses NULL-scored
  emissions): retains 1.0 of oracle held-out wins, **0 regressions**.
  **Symbolic-action learning is empirically alive on Multiset**, but the pool
  (26) is label-limited → **keep the WX3 oracle wrapper in production** and
  mine to Green (≥40 / ≥20 held-out positives) before promoting the learner to
  live wrapper integration. Classifier model + dataset JSONL git-ignored;
  NS9/router/AX1/AX2/WX3 unmodified. See
  `project/evolve/reports/ax3_multiset_symbolic_learning_report.md`.

### Project state

- **Learn track (fine-tunes):** NS15 (Nat) and NS22 (Int/omega) are the
  positive distillations; NS24 confirmed the Int omega surface is
  **saturated** (57→58, near-null). No short-token SFT family has
  appeared since Int/omega.
- **Wrapper track (NS9 + WX):** NS9 is the base genome; WX1/WX2 add a
  state-aware `cases <var> <;> simp` capability (Option + List, +29
  beyond NS9, 0 regressions, namespace-gated). This is the active
  growth edge — state-dependent headroom is captured at search time.
- **Symbolic bridge (AX):** AX1 shows the three regimes connect — **raw
  tactic SFT** works for short stable tactics (`omega`/`aesop`:
  NS15/NS22); the **state-aware wrapper** works for variable-dependent
  tactics (`cases <var> <;> simp`: WX1/WX2); and **symbolic-action
  training** is the bridge — it makes the variable-dependent family
  SFT-ready *as a label* (`CASES_SIMP[List,simp_all]`) while the wrapper
  instantiates the variable from the state. **AX2 then tested the data
  side and came back RED:** the symbolic-label dataset was capped at ~27
  single-shot examples (Option exhausted; fresh List wins are multi-step).
  **WX3 then broke that cap from the wrapper side:** the new Multiset
  `induction_on` action added **20 clean single-shot symbolic labels** in a
  fresh namespace (+25 wins beyond NS9, 0 regressions). **AX3 then trained the
  first symbolic-action learner** on the WX3+AX3 pool (26 clean labels): a
  TF-IDF+logreg classifier over the proof state reaches **0.85 positive recall
  / 0.05 NULL-FP** in CV and retains the oracle's held-out wins with 0
  regressions — symbolic-action *learning* is empirically alive (YELLOW/smoke).
  But at 26 labels it does not beat shipping the deterministic WX3 oracle
  wrapper, so the learner stays experimental pending a mine-to-Green push
  (≥40 labels / ≥20 held-out positives) and live wrapper integration.
- **Mining protocol:** always run NS23-style minimal-tactic relabeling
  before declaring a training gate met. It distinguishes short-token
  (SFT-ready) from state-aware compound (wrapper-ready) families and has
  twice (CX3, WX1/WX2) correctly routed headroom to the wrapper instead
  of a null fine-tune.

### Recommended next directions

The minimal-tactic principle (NS23/NS24) gates training; the Int omega
surface is saturated; CX3 showed the fresh-namespace short-token thesis
does not carry to Bool/Option; WX1/WX2 captured state-dependent headroom
in a **wrapper**; and AX1 showed that headroom can be made SFT-ready *as
a symbolic label*. In rough order of likely yield:

1. **WX4 / surface expansion, then AX3.** WX3 validated the Multiset
   `induction_on` action — **+25 wins beyond NS9, 20 clean single-shot
   symbolic labels** (the first surface to clear AX2's null result). The
   `MULTISET_INDUCTION_SIMP[Multiset,simp_all]` family sits at 18–20, right
   at the ≥20-in-one-family gate. Next: mine the **held-out Multiset
   induction surface** (~86 fresh unused + the full induction-shape catalog)
   under the WX3 induction wrapper to push clean labels to **≥40 total / ≥20
   in the single `simp_all` action_id**, then train **AX3** on that label
   with a held-out Multiset eval. Keep AX1 as the canonical cases wrapper;
   promote `wx3_multiset_induction_safe` as the canonical Multiset wrapper.
2. **Fold WX1+WX2/AX1 into the canonical wrapper** (genome + router
   sign-off) — the AX1 symbolic config is equivalent to WX2 and more
   general; re-baseline the full matrix.
3. **Extend the symbolic action set to remaining inductive surface.**
   Beyond List the cases-friendly catalog is thin; the largest untapped
   surface is Multiset (251 fresh) but it is a quotient — would need a
   `Multiset.induction_on`-aware action (quotient-specific).
4. **Mine fresh held-out Int** (~50 sub-bitwise/dvd candidates unprobed);
   short-token raw SFT stays gated on a genuinely short-token family.

See `project/evolve/reports/ax1_symbolic_action_layer_report.md` for the
AX1 symbolic action layer,
`project/evolve/reports/wx2_state_aware_cases_generalization_report.md`
for the WX2 cases-wrapper generalization arc,
`project/evolve/reports/wx1_option_cases_wrapper_report.md` for the
`project/evolve/reports/wx1_option_cases_wrapper_report.md` for the
WX1 state-aware Option cases-wrapper arc,
`project/evolve/reports/cx3_bool_option_decide_mining_report.md`
for the CX3 mining arc and
`project/evolve/reports/post_ns24_current_status.md` for the one-page
current status,
`project/evolve/reports/ns24_int_minimal_omega_training_report.md`
for the NS24 arc,
`project/evolve/reports/ns23_minimal_tactic_relabeling_report.md` for
NS23, `project/evolve/reports/post_cx1_ns21_cx2_ns22_update.md`
for the CX1→NS22 update,
`project/evolve/reports/learn_track_final_report_ns10_ns20.md`
for the full Learn-track narrative, and
`project/evolve/reports/learn_track_executive_summary.md` for
the short version.
