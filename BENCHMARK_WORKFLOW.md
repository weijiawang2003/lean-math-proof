# Minimal Reproducible Benchmark Workflow

This project intentionally stays lightweight. To compare methods reliably, use this minimal protocol.

## 0) One-command pipeline

## Environment setup

Install pipeline dependencies first:

- `pip install lean_dojo torch transformers "accelerate>=1.1.0"`

For a full default classifier workflow (search -> SFT dataset -> classifier training -> rollout):

- `python run_pipeline.py --pipeline classifier`

（默认使用 `nat_single`；如需扩展集合可传 `--theorem-set nat_more` 或 `--theorem-set mixed_easy_v2`）

For a dry-run preview of commands:

- `python run_pipeline.py --pipeline classifier --dry-run --theorem-set mixed_easy_v2 --action-space search_v2`


## 1) Fix benchmark task set and budget

Pick one theorem set from `tasks.py` and keep fixed budgets:
- rollout: `max_steps`
- search: `beam_width`, `max_depth`
- action space: `action_space`（例如 `core_v1` / `search_v2`）

Record these in `config.json` (automatic when using scripts with `--out-dir`).

Note: `run_pipeline.py` now propagates `--action-space` to both `build_sft_dataset.py` and `train_action_classifier.py` so label spaces stay aligned.

## 2) Run methods into separate run folders

Examples:
- `python model_rollout.py --theorem-set nat_single --max-steps 5 --out-dir runs`
- `python search_generate_traces.py --theorem-set toy_search --beam-width 16 --max-depth 4 --action-space core_v1 --out-dir runs`
- `python search_generate_traces.py --theorem-set mixed_easy_v2 --beam-width 24 --max-depth 6 --action-space search_v2 --out-dir runs`

If a theorem cannot be initialized because LeanDojo trace artifacts are missing (e.g. missing `*.ast.json`), `search_generate_traces.py` now skips that theorem with a warning and continues.

Each run should create:
- `runs/<run_id>/config.json`
- `runs/<run_id>/traces.jsonl`
- `runs/<run_id>/metrics.json`

## 3) Evaluate with episode/run aggregation

If needed, re-run evaluator explicitly:
- `python evaluate_traces.py --in runs/<run_id>/traces.jsonl --out-metrics runs/<run_id>/metrics.json`

Primary comparison metrics:
- episode success rate
- avg steps per episode
- error episode rate

## 4) Smallest missing pieces for stronger reproducibility

- Fixed random seed handling in all stochastic methods.
- Frozen task-set manifests (explicit theorem lists per benchmark version).
- One aggregate comparison script over multiple `runs/*/metrics.json` files.

These are intentionally small, explicit additions and avoid framework overhead.


## Strict mode for theorem coverage

If you want the run to fail whenever any theorem is skipped during search (instead of warning-and-continue), use:

- `python run_pipeline.py --pipeline classifier --theorem-set toy_search --action-space core_v1 --fail-on-skip`
