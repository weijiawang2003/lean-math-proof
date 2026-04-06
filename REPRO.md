# Reproducible Quickstart

This file captures the exact minimal commands to reproduce a successful classifier run.

## 0) Environment

```bash
python --version
pip install lean_dojo torch transformers "accelerate>=1.1.0"
```

## 1) Dry-run first (sanity check command chain)

```bash
python run_pipeline.py --pipeline classifier --dry-run
```

Expected command chain:
1. `search_generate_traces.py`
2. `build_sft_dataset.py`
3. `train_action_classifier.py`
4. `model_rollout.py`

## 2) Run full classifier pipeline (stable default)

```bash
python run_pipeline.py --pipeline classifier
```

Default theorem set is `nat_single`, which is intentionally safer than `toy_search`.

## 3) Verify outputs

You should see:
- `sft_dataset.jsonl` updated,
- `clf_ckpt/` created,
- run artifacts under `runs/search-*/` and `runs/rollout-*/`.

Optional summaries:

```bash
python evaluate_traces.py --in traces_from_search.jsonl
python compare_runs.py --runs-dir runs
```

## 4) Expand to larger theorem set

After the stable run works, try:

```bash
python run_pipeline.py --pipeline classifier --theorem-set toy_search
```

If some theorems cannot initialize due missing LeanDojo `*.ast.json` artifacts, the search step will warn and skip those theorems (unless strict mode is enabled).
