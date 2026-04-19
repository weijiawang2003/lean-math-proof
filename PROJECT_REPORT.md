# 1. Project Summary

`lean-supervised` is a lightweight LeanDojo-based experimentation repo for **supervised tactic policy learning** from proof interaction traces, with an explicit loop of: (1) collect/search transitions in Lean, (2) convert transitions into SFT-style labeled examples, (3) train a classifier policy, and (4) run policy rollouts back in Lean for proof attempts. The core object being modeled is a transition-level mapping from `(theorem name + pretty-printed tactic state)` to a discrete tactic label from a predefined action space. The code also keeps a parallel “baseline/search-first” workflow (beam search over handcrafted action sets) and an older/alternative character LM workflow.

Repository state looks like an **active research prototype moving toward a reproducible baseline pipeline**: it has clear script-level orchestration (`run_pipeline.py`), standardized trace schema and run artifacts, and evaluation/compare utilities; but also includes legacy scripts, mixed schema versions in checked-in artifacts, no formal test suite, and global runtime assumptions (e.g., local `clf_ckpt` presence for rollout policy import).

---

# 2. High-Level Architecture

The repository is organized as a script-driven pipeline around a few shared modules:

- **Core schema/utilities** define theorem descriptors, transition records, JSONL I/O, run directory layout, and LeanDojo transition wrappers.
- **Task/action registries** define theorem sets and tactic action spaces used by search and training.
- **Data generation scripts** run either scripted collection or beam-search expansion to produce transition JSONL traces.
- **Dataset/training scripts** filter/transform transitions into classifier-ready JSONL and train a sequence classifier.
- **Rollout/evaluation scripts** apply trained policy in LeanDojo, then summarize per-transition/per-episode performance.
- **Pipeline orchestrator** chains end-to-end commands with dependency prechecks and optional post-eval.

## Data/control flow in the main classifier workflow

1. **Raw inputs**: theorem sets from `tasks.py` + action set from `actions.py` + LeanDojo/mathlib target from `env.py`.
2. **Trace generation** (`search_generate_traces.py`): runs beam search over action space; writes transition rows to JSONL.
3. **Dataset construction** (`build_sft_dataset.py`): filters traces by progress/finish, maps tactic string→label index, writes `{"prompt","label"}` rows.
4. **Training** (`train_action_classifier.py`): trains HuggingFace sequence classifier on those rows; saves model/tokenizer/action-space JSON in `clf_ckpt`.
5. **Inference rollout** (`model_rollout.py` + `policy.py`): loads checkpoint, predicts tactic per state, interacts with Dojo and logs new transitions.
6. **Evaluation** (`evaluate_traces.py`, `compare_runs.py`): aggregates transition/episode/run metrics from JSONL + run metrics files.

## Multiple workflows present

- **Mainline (current)**: `classifier` pipeline in `run_pipeline.py` (search → SFT build → classifier train → rollout, optional eval).
- **Alternative/legacy**: `charlm` pipeline (`collect_traces.py` + `train_sft_char_lm.py`) trains a GRU char LM over concatenated state+tactic text; appears older and less integrated with search/eval conventions.
- **Standalone baseline**: `search_baseline.py` for single-theorem beam stats without trace dumping.

---

# 3. Directory and File Guide

## Core shared modules

- `core_types.py`
  - Purpose: Canonical dataclasses (`TheoremConfig`, `TransitionRecord`) and shared prompt format.
  - When it is used: Everywhere traces or prompt text are produced.
  - Key dependencies: `dataclasses`.
  - Main outputs: In-memory schema + `build_prompt(...)` string format.
  - Notes for future development: Best place to centralize stricter schema validation and IDs.

- `env.py`
  - Purpose: LeanDojo wrappers for repo/theorem creation and transition normalization.
  - When it is used: Any script executing tactics in Dojo.
  - Key dependencies: `lean_dojo`, `core_types`.
  - Main outputs: `TransitionOutcome` with serialized `TransitionRecord` + next state.
  - Notes: Hard-codes mathlib commit; central coupling point for runtime behavior.

- `trace_io.py`
  - Purpose: Append/write/iterate JSONL.
  - When it is used: All trace and dataset scripts.
  - Key dependencies: `json`, `core_types`.
  - Main outputs: JSONL files.
  - Notes: No schema validation beyond JSON parse.

- `experiment_io.py`
  - Purpose: Standard `runs/<run_id>/{config,traces,metrics}` lifecycle.
  - When it is used: collection/search/rollout scripts and evaluator outputs.
  - Key dependencies: `pathlib`, `json`, datetime.
  - Main outputs: run folders and metrics/config files.
  - Notes: This is the reproducibility backbone.

- `actions.py`
  - Purpose: Action-space registry, serialization helpers.
  - When it is used: search, dataset build, classifier train, policy inference.
  - Key dependencies: none external.
  - Main outputs: deterministic action lists; `action_space.json`.
  - Notes: `ACTIONS` alias remains for backward compatibility.

- `tasks.py` / `benchmark_specs.py`
  - Purpose: theorem-set registry and higher-level benchmark presets.
  - When it is used: most scripts take `--theorem-set`; benchmark specs currently advisory.
  - Key dependencies: `core_types`.
  - Main outputs: theorem lists and static config dicts.
  - Notes: no manifest freezing/versioning yet.

## Data collection & search

- `search_generate_traces.py`
  - Purpose: Beam-search theorem attempts that log non-error transitions and write run metrics.
  - When used: first stage of classifier pipeline.
  - Key dependencies: `env.py`, `actions.py`, `tasks.py`, `evaluate_traces.py`.
  - Main outputs: trace JSONL (+ metrics with availability/skip summary).
  - Notes: includes theorem availability precheck and skip/fail modes.

- `collect_traces.py`
  - Purpose: scripted deterministic trace generation from hardcoded `PROBLEMS`.
  - When used: char-LM pipeline and debugging.
  - Key dependencies: `env.py`, `experiment_io.py`.
  - Main outputs: small traces + basic episode metrics.
  - Notes: limited theorem coverage; acts as synthetic seed data path.

- `collect_one_example.py`
  - Purpose: single theorem, single tactic probe with run artifacts.
  - When used: smoke/debug checks.
  - Key dependencies: `tasks.py`, `env.py`.
  - Main outputs: one transition trace and minimal metrics.

- `search_baseline.py`
  - Purpose: baseline beam search stats for one theorem.
  - When used: quick baseline comparisons.
  - Key dependencies: `env.py`, `tasks.py`, `actions.py`.
  - Main outputs: summary metrics JSON only (no transition log).
  - Notes: partly redundant with `search_generate_traces.py`.

## Dataset and model training

- `build_sft_dataset.py`
  - Purpose: convert transition trace JSONL to classifier dataset JSONL (`prompt`,`label`).
  - When used: second stage of classifier pipeline.
  - Key dependencies: `trace_io.py`, `core_types.build_prompt`, `actions.py`.
  - Main outputs: SFT dataset with optional metadata.
  - Notes: has key anti-collapse controls (`min_goal_drop`, dedup, per-label cap).

- `train_action_classifier.py`
  - Purpose: train HuggingFace sequence classifier over SFT rows.
  - When used: third stage of classifier pipeline.
  - Key dependencies: `transformers`, `torch`, `actions.py`.
  - Main outputs: checkpoint dir (model, tokenizer, action-space JSON).
  - Notes: no explicit validation split/eval loop yet.

- `policy.py`
  - Purpose: runtime inference policy loader + `choose_tactic`.
  - When used: `model_rollout.py` imports it.
  - Key dependencies: `transformers`, `torch`, local `clf_ckpt`.
  - Main outputs: predicted tactic string.
  - Notes: model loads at import time; fragile if checkpoint missing.

- `train_sft_char_lm.py`
  - Purpose: legacy alternative char-level GRU language-model over traces.
  - When used: `charlm` pipeline.
  - Key dependencies: `torch`.
  - Main outputs: `char_lm_ckpt.pt`; optional tactic generation.
  - Notes: global constants/hardcoded filenames, no run-dir integration.

## Rollout and evaluation

- `model_rollout.py`
  - Purpose: sequential policy rollout in Dojo for one theorem.
  - When used: final stage of classifier pipeline.
  - Key dependencies: `policy.py`, `env.py`, `tasks.py`.
  - Main outputs: rollout trace + metrics in run directory.

- `evaluate_traces.py`
  - Purpose: aggregate transition, episode, run metrics from trace JSONL.
  - When used: standalone or auto-eval stage.
  - Key dependencies: `trace_io.py`, `experiment_io.py`.
  - Main outputs: in-memory metrics dict and optional JSON file.

- `compare_runs.py`
  - Purpose: normalize and tabulate `runs/*/metrics.json` across heterogeneous methods.
  - When used: quick experiment comparison.
  - Key dependencies: `json`, `pathlib`.
  - Main outputs: printed comparison table.

## Orchestration & docs/artifacts

- `run_pipeline.py`
  - Purpose: one-command pipeline runner with precheck, dry-run, optional auto eval.
  - When used: recommended top-level execution entry.
  - Key dependencies: all stage scripts via subprocess.
  - Main outputs: chained artifacts from selected pipeline.

- `REPRO.md`, `BENCHMARK_WORKFLOW.md`, `TRACE_SCHEMA.md`, `TASK_ROADMAP.md`
  - Purpose: workflow, reproducibility, schema, and implementation guidance.
  - Notes: README is absent; these docs collectively act as operational docs.

- `toy_trace.jsonl`, `traces.jsonl`, `traces_from_search.jsonl`, `sft_dataset.jsonl`, `toy_dataset.jsonl`
  - Purpose: checked-in sample/generated artifacts useful for schema inspection and sanity checks.
  - Notes: schemas differ slightly across older/newer outputs (see Section 5).

---

# 4. End-to-End Pipeline

## A) Main classifier pipeline (current mainline)

1. **Environment setup**
   - Install: `lean_dojo`, `torch`, `transformers`, `accelerate` (per `REPRO.md` / pipeline prechecks).
   - Assumption: LeanDojo has access to mathlib trace artifacts for selected files.

2. **Trace/search generation**
   - Script: `search_generate_traces.py` (or via `run_pipeline.py --pipeline classifier`).
   - Inputs: theorem set, action space, beam width/depth.
   - Outputs: transition JSONL (default `traces_from_search.jsonl`) + run metrics in `runs/search-*`.
   - Failure points: theorem unavailable (`DojoInitError`, missing `.ast.json`), theorem name drift; can skip or fail hard via flags.

3. **SFT dataset build**
   - Script: `build_sft_dataset.py`.
   - Inputs: trace JSONL + action space.
   - Outputs: `sft_dataset.jsonl` with `prompt` and `label` (optional `meta`).
   - Failure points: all rows filtered out (unknown tactic labels, no-progress filter too strict, dedup/label cap too aggressive).

4. **Classifier training**
   - Script: `train_action_classifier.py`.
   - Inputs: SFT dataset, model name, action space.
   - Outputs: checkpoint dir `clf_ckpt/` (model/tokenizer/action-space file).
   - Failure points: empty/too-small dataset, GPU/CPU memory, HuggingFace model download availability.

5. **Rollout/inference**
   - Script: `model_rollout.py`.
   - Inputs: theorem set/index, `clf_ckpt` from previous step.
   - Outputs: rollout transitions + metrics under `runs/rollout-*`.
   - Failure points: import-time failure in `policy.py` if checkpoint missing/incompatible with action space.

6. **Evaluation/compare**
   - Scripts: `evaluate_traces.py` and `compare_runs.py` (optional auto-run with `--auto-eval`).
   - Outputs: metric dicts/files + console table.
   - Interpretation: episode success rate, error episodes, transitions by method.

## B) Char-LM pipeline (alternative)

1. `collect_traces.py` writes simple scripted trace data.
2. `train_sft_char_lm.py --mode train` trains char-level GRU LM.
3. Optional `--mode gen --state-file ...` generates a tactic string.

This path is minimally integrated with run/eval/search conventions and appears exploratory.

---

# 5. Data Structures and Schemas

## `TheoremConfig` (dataclass)
- Fields: `file_path: str`, `full_name: str`.
- Meaning: theorem locator in mathlib repo.
- Produced by: static registries in `tasks.py`.
- Consumed by: collection, search, rollout scripts via `env.make_theorem`.

## `TransitionRecord` (dataclass)
- Core fields:
  - `file_path`, `full_name` (target theorem metadata)
  - `state_pp` (model input text)
  - `tactic` (action/label source)
  - `result_kind` (LeanDojo result class)
  - `proof_finished` (terminal success flag)
- Optional analytic fields:
  - `num_goals_before`, `num_goals_after` (progress signal)
  - `step`, `domain`, `error_message`
  - `run_id`, `episode_id`, `method` (aggregation keys)
- Produced by: `env.run_transition`.
- Consumed by: `build_sft_dataset.py`, `evaluate_traces.py`, downstream analysis.

## Search trace JSONL row
- Usually serialized `TransitionRecord` dict.
- In `search_generate_traces.py`, only **non-error** rows are written; some metrics still account for errors only if present in trace rows.
- Used as supervised source for classifier SFT build.

## SFT dataset row (`build_sft_dataset.py`)
- Required fields:
  - `prompt: str` from `build_prompt(state_pp, full_name)`.
  - `label: int` action index in selected action space.
- Optional:
  - `meta` with theorem/run/episode/method/step and filter config.
- Produced by: `build_sft_dataset.py`.
- Consumed by: `train_action_classifier.SFTRawDataset`.

## Evaluation output schema (`evaluate_traces.py`)
- Top-level sections:
  - `transition_metrics`
  - `episode_metrics`
  - `run_metrics`
  - `rows_by_method`
  - per-theorem summaries (transition + episode).
- Consumed by: humans and `compare_runs.py` normalization.

## Schema inconsistencies observed

1. **Older trace artifacts** (`traces.jsonl`, `toy_trace.jsonl`) omit some newer optional fields (`run_id`, `episode_id`, `method`, goals metadata in some rows).
2. `evaluate_traces.py` treats episode key fallback as theorem identity when run/episode missing, potentially conflating attempts.
3. `collect_trace_nat_mul_add_mod.py` uses ad-hoc dict schema and bypasses shared modules.

---

# 6. Training and Modeling Logic

Current primary model is a **multi-class tactic classifier** using `AutoModelForSequenceClassification` (default `distilbert-base-uncased`) trained on prompt/label pairs. Inputs are prompt text containing theorem name and pretty-printed proof state; targets are integer tactic labels aligned to an action-space list. The supervised dataset is derived from search traces by keeping proof-finishing rows or rows with enough goal reduction (`min_goal_drop`), and optionally deduplicating `(state_pp,tactic)` plus per-label caps to reduce collapse.

Train/validation split is **not implemented**; training runs on all dataset rows with HuggingFace `Trainer` and periodic checkpointing each epoch. Final artifacts include model weights/tokenizer and `action_space.json`, which rollout inference loads to map predicted class indices back to tactic strings. If no checkpoint-specific action-space file exists, `policy.py` falls back to legacy core actions, so mismatch risk exists when training with expanded spaces.

A secondary model (`train_sft_char_lm.py`) is a character-level GRU LM that learns to continue `state_pp + "TACTIC:"` text and emit a tactic line. This appears exploratory and currently separate from the main search/rollout loop.

---

# 7. Search / Rollout / Evaluation Logic

## Search trace generation

`search_generate_traces.py` performs breadth-by-depth beam expansion:
- Node stores Lean state + tactic history + finished flag.
- For each beam node and each action, it executes tactic via `run_transition`.
- Error outcomes are dropped.
- Non-error rows are logged only when goals do not worsen (`goals_after <= goals_before`).
- Beam ranking prioritizes finished nodes, then fewer goals.

Additional robustness:
- Precheck partitions available/unavailable theorems by attempting Dojo init.
- Skip reasons recorded (`missing_trace_artifact`, `theorem_not_found`, etc.).
- Modes: continue-with-warning, fail on unavailable, fail on any skip.

## Baseline search

`search_baseline.py` is similar beam logic but focused on one theorem and only summary metrics; no transition reuse for SFT.

## Model rollout

`model_rollout.py` is **sequential greedy single-action rollout**:
- At each step, classifier predicts one tactic (`argmax` softmax).
- Applies tactic in Dojo and logs transition.
- Terminates on proof finish, first error, or max-step budget.

No beam, backtracking, or fallback policy is currently implemented for learned rollout.

## Evaluation

`evaluate_traces.py` computes:
- transition counts (rows, finished rows, error rows)
- episode metrics (episodes, finished/error episodes, success rate, avg steps)
- run grouping and per-theorem summaries.

Success is episode-level `any proof_finished row`. Error episode is any row with non-empty `error_message`.

Project objective emphasis appears to be **episode proof success under bounded interaction**, with transition quality/progress used to construct supervised labels and reduce noisy samples.

---

# 8. Environment and Dependencies

## Runtime requirements inferred

- Python 3 (exact minor not pinned).
- Core packages: `lean_dojo`, `torch`, `transformers`, `accelerate`.
- Lean target: mathlib4 repo at commit `29dcec074de168ac2bf835a77ef68bbe069194c5` (hard-coded in `env.py` and legacy scripts).
- Requires LeanDojo-traced artifacts for theorem files; missing `*.ast.json` causes theorem unavailability.

## Setup workflow visibility

- Docs suggest pip-based setup only; no `requirements.txt`, `pyproject.toml`, or conda env file.
- No explicit environment variables required by scripts.

## Path assumptions

- Checkpoints default to `clf_ckpt` (`policy.py`, training defaults).
- Many outputs default to root-level files (`traces_from_search.jsonl`, `sft_dataset.jsonl`) plus run dirs.
- `policy.py` loads model at import time, so scripts importing it assume checkpoint exists locally.

## Documentation consistency

- `REPRO.md` and `BENCHMARK_WORKFLOW.md` are mostly aligned with code.
- No central README; operational knowledge is fragmented but present.

---

# 9. Current State of the Project

The project is in a **workable prototype phase with a near-minimal end-to-end path**. The classifier pipeline appears runnable in principle from one command and includes practical controls (dry-run, action-space propagation, theorem availability handling, auto eval). Shared modules (`core_types`, `env`, `trace_io`, `experiment_io`) indicate an ongoing cleanup effort.

At the same time, signs of incompletely consolidated evolution remain: legacy scripts coexist (`collect_trace_nat_mul_add_mod.py`, char-LM route), artifact schemas in checked-in JSONLs vary, and there is no automated test harness or canonical data manifest. The most reliable “mainline” is the `run_pipeline.py --pipeline classifier` path with conservative theorem sets (`nat_single`), while broader theorem sets and expanded action spaces are likely more brittle due LeanDojo artifact availability and label-space dynamics.

---

# 10. Problems, Risks, and Technical Debt

1. **Import-time checkpoint coupling in `policy.py`**
   - Problem: model/tokenizer load on import, hard dependency on `clf_ckpt`.
   - Why it matters: rollout or any import can crash before CLI validation; hard to test/multiplex checkpoints.
   - Fix: lazy-load policy in function/class with explicit `--ckpt-dir` and clear error messages.

2. **No train/val split or training evaluation**
   - Problem: classifier trains on full data without holdout metrics.
   - Why it matters: cannot quantify generalization or regression.
   - Fix: add deterministic split + eval metric logging (accuracy/top-k per label).

3. **Schema drift across scripts/artifacts**
   - Problem: older scripts bypass shared `TransitionRecord`; some rows miss run/episode/method/goal fields.
   - Why it matters: evaluation grouping and reproducibility weaken.
   - Fix: deprecate legacy writers; enforce schema validation in `trace_io`.

4. **Search and baseline duplication**
   - Problem: `search_baseline.py` duplicates beam mechanics separate from `search_generate_traces.py`.
   - Why it matters: behavioral divergence risk and maintenance overhead.
   - Fix: extract reusable beam engine; make baseline a mode/config.

5. **Task/theorem availability fragility**
   - Problem: theorem names/files are static and may drift across trace states; availability depends on local LeanDojo caches.
   - Why it matters: runs unpredictably skip large fractions.
   - Fix: add theorem manifest validation script with explicit artifact preflight report.

6. **Documentation fragmentation**
   - Problem: no single canonical runbook/README, guidance split across multiple MD files.
   - Why it matters: onboarding friction and parameter mismatch risk.
   - Fix: consolidate into one pipeline doc + per-script CLI examples.

7. **No automated smoke tests**
   - Problem: no CI checks for parser/schema/pipeline wiring.
   - Why it matters: refactors can silently break end-to-end behavior.
   - Fix: add tiny local smoke test (one theorem, one step, dry-run + schema check).

---

# 11. Recommended Next Steps

## Highest Priority

- Centralize policy loading and checkpoint config:
  - add `Policy` class with lazy init and explicit checkpoint/action-space parameters.
- Add deterministic dataset split + classifier eval metrics (train/val) in `train_action_classifier.py`.
- Create one canonical `docs/pipeline.md` combining classifier flow, artifact contracts, and failure handling.
- Add schema validator (`validate_trace.py`) to enforce required `TransitionRecord` fields pre-SFT.
- Add one end-to-end smoke command in CI/local script: search on `nat_single` with tiny depth + SFT build + trainer dry sanity.

## Medium Priority

- Refactor beam logic into shared module and make `search_baseline.py` a thin wrapper.
- Introduce explicit output naming conventions to avoid root clutter (prefer run-dir scoped artifacts).
- Add theorem availability preflight script producing machine-readable report.
- Version benchmark/task manifests (`benchmarks/<name>.json`) and pin random seeds centrally.

## Nice to Have

- Add richer rollout strategies (top-k sampling / beam rollout / fallback tactics on error).
- Add action-space analytics (label frequency, entropy, class imbalance reports).
- Integrate char-LM path or archive it into `legacy/` with clear status.

---

# 12. Claude Handoff Brief

Claude, this repo’s core goal is to learn a supervised Lean tactic policy from search-generated transitions and test it via LeanDojo rollout. Read in this order: `run_pipeline.py` (orchestration), `search_generate_traces.py` (data source), `build_sft_dataset.py` (label shaping), `train_action_classifier.py` + `policy.py` (model loop), and `evaluate_traces.py` (metrics contract). 

Top next actions: (1) decouple `policy.py` import-time checkpoint loading, (2) add train/val evaluation in classifier training, (3) enforce one canonical transition schema with validator, (4) unify duplicate beam search logic, (5) write one canonical pipeline doc + smoke test. Biggest pitfalls are LeanDojo theorem availability, action-space mismatch between train/infer, and silent schema drift in legacy artifacts. Fastest productivity path: run classifier pipeline on `nat_single` first, inspect generated traces/SFT rows manually, then harden one module at a time with tiny reproducible runs.

---

# 13. Output Requirements

This report is derived from direct inspection of repository code, docs, and tracked artifacts, including scripts, schema docs, and sample JSONL outputs. Where uncertain (e.g., which path is “current mainline”), conclusions are marked as inference based on orchestration defaults and script coupling.

## Runbook (practical command order)

```bash
# 0) Install deps
pip install lean_dojo torch transformers "accelerate>=1.1.0"

# 1) Dry-run full classifier pipeline
python run_pipeline.py --pipeline classifier --dry-run --theorem-set mixed_easy_v2 --action-space search_v2

# 2) Stable minimal run
python run_pipeline.py --pipeline classifier --theorem-set nat_single --action-space core_v1 --auto-eval

# 3) Manual metric checks
python evaluate_traces.py --in traces_from_search.jsonl
python compare_runs.py --runs-dir runs
```

## Quickest Path for Claude

1. Run `python run_pipeline.py --pipeline classifier --dry-run` to confirm command wiring.
2. Execute a minimal real run on `nat_single` with `core_v1` action space.
3. Inspect `traces_from_search.jsonl` and `sft_dataset.jsonl` for label quality and class balance.
4. Verify `clf_ckpt/action_space.json` matches training action-space and rollout expectations.
5. Refactor `policy.py` to lazy-load checkpoint and add explicit CLI checkpoint argument in `model_rollout.py`.
6. Add train/val split + evaluation logging to `train_action_classifier.py`.
7. Introduce schema validation and retire ad-hoc legacy trace writers.
