# Lean-Supervised Tactic Policy Learning

A research project where a language model proposes Lean proof steps and the Lean theorem prover supervises every one. Nothing enters the training set unless Lean has verified it.

## The one idea

A theorem prover is the cheapest, most honest supervisor we could ever give a model. Lean checks every tactic mechanically — it never grades with partial credit, never hallucinates approval, never gets tired. "Did this tactic work?" is a free, perfectly calibrated signal, and the whole pipeline is built around it.

Three consequences follow:

- **Search.** We can cheaply generate supervised data by letting Lean grade thousands of beam-search attempts.
- **Learn.** We train seq2seq T5-small tactic generators on the survivors. Five checkpoints shipped so far (`gen_v1` → `gen_v5`).
- **Curriculum.** We stage difficulty (easy → medium → hard), measure zero-shot transfer, retrain, repeat.

## Current state (April 2026)

| Metric | Number | What it measures |
|---|---|---|
| Curriculum success | **25 / 30 (83%)** | `gen_v5` top-k=8 rollout on the 30-theorem curriculum |
| Premise-augmented | **19 / 30 (63%)** | `gen_v6` with retrieved lemmas injected into the prompt — net-negative at T5-small scale |
| Full-corpus coverage | **200 / 555 (36%)** | Theorems with any beam-search-verified proof, across three mathlib files |
| Training pool | 103,650 transitions | 96,027 from beam search + selfplay residue |
| Distinct tactics observed | 171 | Across all verified transitions |

Coverage by file (oracle beam search):

- `Mathlib/Data/Finset/Basic.lean` — 132 / 231 (57%)
- `Mathlib/Data/Set/Basic.lean`    —  62 / 116 (53%)
- `Mathlib/Data/Nat/Defs.lean`     —   6 / 208  (3%)  ← current frontier

`Nat.Defs` is where induction / `ring` / `linarith`-style proofs actually matter, and where beam search rarely finds a proof on its own.

## The policy family

| Policy | Architecture | Status |
|---|---|---|
| Classifier | DistilBERT over 179 fixed tactics | Superseded baseline |
| Generative v5 | T5-small seq2seq (~60M params) | Current SOTA — 25/30 (83%) |
| Premise-augmented v6 | T5-small + retrieved lemmas in prompt | Negative result — 19/30 (63%) |
| Strategic | Backward reasoning + cross-field translation + premise templates | Planned, waiting on a t5-base base model |

Every policy is loadable at rollout time via `--policy-type`; the same eval harness runs against all of them.

### Why premise retrieval hurt at this scale

On the same curriculum, injecting a `Relevant premises: …` prefix into the prompt dropped performance from 25/30 to 19/30. Six Set-subset lemmas v5 solved in one step (`aesop` / `simp`) fell over in v6: `Set.subset_univ`, `Set.empty_subset`, `Set.subset_union_left`, `Set.subset_union_right`, `Set.inter_subset_left`, `Set.inter_subset_right`.

Our read: capacity, not information, is the current bottleneck. T5-small has ≈60M parameters. ReProver — whose result inspired this experiment — used 299M. The extra ≈40 tokens of premise prefix push a small model's representation budget past what it can decode cleanly. The prediction is that at t5-base (220M) the same experiment flips from net-negative to net-positive.

## How it fits together

```
  Search          Build              Train                 Rollout
  ──────          ─────              ─────                 ───────
  Beam over  →  Filter by      →  T5-small seq2seq  →  Top-k=8 fallback;
  171-tactic    progress;         (~60M params)         Lean grades each step
  space in     (state → tactic)
  Lean          pairs

       ↑ premise retriever (ReProver-style) feeds both Train and Rollout
```

Every transition written to `runs/<run_id>/traces.jsonl` is a `TransitionRecord` — state before, tactic applied, Lean's verdict, goal count deltas, episode/run IDs. Schema: see `TRACE_SCHEMA.md`.

## Worked example

Theorem `Nat.mul_add_mod'`: for naturals `a, b, c`, `(a·b + c) mod b = c mod b`. The state Lean shows the model is approximately:

```
a b c : ℕ
⊢ (a * b + c) % b = c % b
```

The model emits `simp [Nat.add_mul_mod_self_left, Nat.add_comm]`, Lean accepts, and this `(state, tactic)` pair becomes a labeled training example.

## Project layout

```
core_types.py            TransitionRecord schema
env.py                   LeanDojo environment wrapper
actions.py               tactic action space (classifier label set)
policy.py                classifier policy
generative_policy.py     T5-small seq2seq policy (current)
hybrid_policy.py         generative + fallback blend
strategic_policy.py      backward / cross-field / premise-template wrapper
premise_retriever.py     token-overlap + embedding fallback
trace_io.py              TransitionRecord I/O
experiment_io.py         runs/<id>/{config,metrics,traces} layout
project_state.py         persistent ProjectState across runs

build_sft_dataset.py             classifier SFT pairs
build_seq2seq_dataset.py         generative SFT pairs
build_premise_augmented_dataset.py  with retrieved premises prepended
train_action_classifier.py       DistilBERT classifier trainer
train_tactic_generator.py        T5-small seq2seq trainer
train_decoder_policy.py          decoder-only (GPT-2) trainer

search_baseline.py               beam search baseline
search_generate_traces.py        beam search data collection
collect_traces.py                one-off trace collector
model_rollout.py                 top-k policy rollout + Lean verification
eval_rollout_all.py              evaluate a policy across a theorem set
compare_runs.py                  diff two run directories

run_pipeline.py                  single search → train → eval run
run_curriculum.py                classifier curriculum (tier 1 → 2 → 3)
run_generative_curriculum.py     same harness, generative policy
run_incremental.py               crash-safe incremental pipeline
run_scale.py                     full-corpus expansion

project/                         data and checkpoints (gitignored)
  models/gen_v1 … gen_v5         T5-small checkpoints
  gen_ckpt_v6_premise            premise-augmented checkpoint
  seq2seq_data_v*.jsonl          training pairs
  seq2seq_premise_v1.jsonl       with retrieved premises
  all_traces.jsonl               aggregated verified transitions
  premise_index.json             retriever index
  project_state.json             cross-run state

runs/<run_id>/                   per-run outputs
  config.json                    exact run config
  traces.jsonl                   TransitionRecords
  metrics.json                   aggregate + per-theorem results
curriculum_runs/<stage>/         curriculum stage outputs
```

## Running things

Environment setup (assumes LeanDojo and a mathlib checkout are already configured):

```
pip install -e .
```

One-off rollout against a theorem set:

```
python eval_rollout_all.py \
  --policy-type generative \
  --ckpt-dir project/models/gen_v5 \
  --theorem-set curriculum_all \
  --top-k 8 --max-steps 8
```

Rerun the curriculum (classifier or generative):

```
python run_generative_curriculum.py
```

Grow the training pool via beam search:

```
python search_generate_traces.py --file Mathlib/Data/Set/Basic.lean \
  --beam-width 24 --beam-depth 6
```

Train the next generative checkpoint on the accumulated data:

```
python train_tactic_generator.py --data project/seq2seq_data_v5.jsonl \
  --base-model t5-small --output project/models/gen_v6
```

The per-run outputs are crash-safe: `run_incremental.py` and `ProjectState` track what's already been searched/proved so you can resume.

## What's next — three phases

**Phase 1 — Grow the oracle (weeks 1–4).** Run `strategic_v5`-style search (induction / cases templates, expanded premise hints) on the 355 theorems that have no verified proof yet. Biggest opportunity is `Nat.Defs`, currently 6/208.

**Phase 2 — Scale the base model (weeks 4–8).** Retrain from scratch on the Phase-1 data pool using t5-base (220M). Re-run the v6 premise-augmentation experiment. Prediction: with ~4× the capacity, the prompt-prefix cost is absorbed and injection turns net-positive.

**Phase 3 — Strategic + backward reasoning at scale (weeks 8+).** Layer the strategic policy (backward `suffices`/`have`, cross-field translation, premise templates) on top of the larger base. Target: open-set proofs that beam search alone never finds.

The crosscut constraint across all phases: Lean stays in the loop. No synthetic data, no self-labeled rollouts in the training pool. The supervision signal stays cheap and perfectly calibrated.

## Open questions

- Cost/benefit of the t5-base retrain — GPU hours vs. expected uplift.
- Neural retriever vs. the hand-written one — our current retriever is token-overlap + embedding fallback; ReProver used a learned DPR-style retriever.
- `Nat.Defs` proof templates — is the gap there mostly about tactics the policy has never seen (`induction`, `ring`, `omega`), or about search depth, or about goal decomposition?
- "aesop" dominance — are we learning tactics or learning when to call `aesop`?

## Reference documents

- `TRACE_SCHEMA.md` — the `TransitionRecord` schema and run/episode conventions.
- `PROJECT_REPORT.md` — full project report.
- `TASK_ROADMAP.md` / `HARDENING_PLAN.md` / `TAKEOVER_NOTES.md` — operational notes.
- `BENCHMARK_WORKFLOW.md` — how the curriculum and eval harness are wired up.
- `lean_supervised_progress.pptx` — slide deck summarizing current state and plan.
