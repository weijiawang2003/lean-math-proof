# Minimal Reproducible Benchmark Workflow

This project intentionally stays lightweight. To compare methods reliably, use this minimal protocol.

## 0) One-command pipeline

## Environment setup

Install pipeline dependencies first:

- `pip install lean_dojo torch transformers`

For a full default classifier workflow (search -> SFT dataset -> classifier training -> rollout):

- `python run_pipeline.py --pipeline classifier`

(默认使用 `nat_single`，更稳妥；需要更大集合时再显式传 `--theorem-set toy_search`)

For a dry-run preview of commands:

- `python run_pipeline.py --pipeline classifier --dry-run`


## 1) Fix benchmark task set and budget

Pick one theorem set from `tasks.py` and keep fixed budgets:
- rollout: `max_steps`
- search: `beam_width`, `max_depth`

Record these in `config.json` (automatic when using scripts with `--out-dir`).

## 2) Run methods into separate run folders

Examples:
- `python model_rollout.py --theorem-set nat_single --max-steps 5 --out-dir runs`
- `python search_generate_traces.py --theorem-set toy_search --beam-width 16 --max-depth 4 --out-dir runs`

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
