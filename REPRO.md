# Reproducible Quickstart

This file captures the exact minimal commands to reproduce a successful classifier run.

## 0) Environment

```bash
python --version
pip install lean_dojo torch transformers "accelerate>=1.1.0"
```

## 1) Dry-run first (sanity check command chain)

```bash
python run_pipeline.py --pipeline classifier --dry-run --theorem-set mixed_easy_v2 --action-space search_v2 --auto-eval
```

Expected command chain:
1. `search_generate_traces.py`
2. `build_sft_dataset.py`
3. `train_action_classifier.py`
4. `model_rollout.py`

## 2) Run full classifier pipeline (stable default)

```bash
python run_pipeline.py --pipeline classifier --theorem-set nat_single --action-space core_v1 --min-goal-drop 1 --auto-eval
```

Default theorem set is `nat_single` (safer). For expanded search labels, pass `--action-space search_v2`; this now propagates to SFT building and classifier training.

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


## 5) Stabilize theorem availability

If you want the run to fail when availability precheck filters any theorem:

```bash
python run_pipeline.py --pipeline classifier --theorem-set mixed_easy_v2 --action-space search_v2 --fail-on-unavailable
```
