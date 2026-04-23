# Hardening Plan — lean-supervised

## Stage 0: Unblock Basic Usability

### 0.1 Lazy-load policy checkpoint
- **Files:** `policy.py`, `model_rollout.py`
- **Change:** Replace module-level model/tokenizer load with a `Policy` class that loads on first call. Accept `--ckpt-dir` in `model_rollout.py`.
- **Breaking:** Yes — any code doing `from policy import choose_tactic` must change to instantiate `Policy(ckpt_dir)` first, or use a convenience wrapper.
- **Test:** `python -c "import policy; print('ok')"` must succeed with no checkpoint present.

### 0.2 Fail loudly on action-space mismatch
- **Files:** `policy.py`
- **Change:** After loading the model, assert `model.config.num_labels == len(POLICY_ACTIONS)`. Remove the silent `idx = 0` fallback; raise `ValueError` instead.
- **Breaking:** No — only changes error behavior that was previously silent and wrong.
- **Test:** Manually create a checkpoint with 12 labels, load with 42-action space, confirm it raises.

## Stage 1: Eliminate Correctness / Configuration Hazards

### 1.1 Add train/validation split
- **Files:** `train_action_classifier.py`
- **Change:** Add `--val-split` (default 0.1). Use `torch.utils.data.random_split` with a fixed seed. Pass `eval_dataset` to `Trainer` and set `eval_strategy="epoch"`. Log eval accuracy.
- **Breaking:** No.
- **Test:** Run training on `sft_dataset.jsonl`, confirm eval metrics appear in logs each epoch.

### 1.2 Make run IDs collision-proof
- **Files:** `experiment_io.py`
- **Change:** Already partially fixed — `search_generate_traces.py`, `model_rollout.py`, `collect_traces.py` etc. already use `uuid.uuid4().hex[:8]`. The only risk is calling `make_run_dir` without a pre-generated `run_id`. Add a uuid suffix in the default path of `make_run_dir`.
- **Breaking:** No.
- **Test:** Call `make_run_dir("runs", "test")` twice in rapid succession; confirm different directories.

### 1.3 Save action-space hash into checkpoint metadata
- **Files:** `train_action_classifier.py`, `policy.py`
- **Change:** At train time, write `action_space_hash = hashlib.sha256(json.dumps(actions).encode()).hexdigest()[:12]` into `clf_ckpt/training_meta.json`. At load time, verify hash matches.
- **Breaking:** No (old checkpoints just skip the check with a warning).
- **Test:** Train, tamper with `action_space.json`, attempt rollout — should error.

### 1.4 Add determinism controls
- **Files:** `train_action_classifier.py`, `search_generate_traces.py`, `run_pipeline.py`
- **Change:** Accept `--seed` (default 42). Call `torch.manual_seed(seed)`, `random.seed(seed)`. Pass seed to `TrainingArguments`.
- **Breaking:** No.
- **Test:** Two identical training runs produce identical loss curves.

## Stage 2: Make Experiments Trustworthy

### 2.1 Enforce schema on trace write
- **Files:** `trace_io.py`
- **Change:** In `append_jsonl`, when receiving a dict (not `TransitionRecord`), validate required keys: `file_path, full_name, state_pp, tactic, result_kind, proof_finished`. Warn on missing optional keys.
- **Breaking:** Will break `collect_trace_nat_mul_add_mod.py` (intentionally — it needs to use shared modules).
- **Test:** Attempt to write a dict missing `state_pp`; confirm it raises.

### 2.2 Normalize episode_id construction
- **Files:** `search_generate_traces.py`, `collect_traces.py`, `model_rollout.py`, `collect_one_example.py`
- **Change:** Standardize to `f"{full_name}:{run_id_suffix}"` everywhere. Add a helper in `experiment_io.py`.
- **Breaking:** No (old data still parseable; new data more consistent).
- **Test:** Run search + rollout, verify `episode_id` format matches across trace files.

### 2.3 Add JSONL error tolerance in training
- **Files:** `train_action_classifier.py`
- **Change:** Wrap `json.loads` in try/except, log and skip malformed lines with a count.
- **Breaking:** No.
- **Test:** Insert a corrupt line in sft_dataset.jsonl, confirm training still runs and reports 1 skipped.

### 2.4 Add CLI argument validation
- **Files:** `run_pipeline.py`, `search_generate_traces.py`, `model_rollout.py`
- **Change:** Add `argparse` validators: `beam_width >= 1`, `max_steps >= 1`, `max_depth >= 1`, `theorem_index >= 0` and `< len(theorems)`.
- **Breaking:** No.
- **Test:** `python model_rollout.py --max-steps -1` should print a clean error.

## Stage 3: Clean Up Architecture / Remove Dead Branches

### 3.1 Archive char-LM path
- **Files:** `train_sft_char_lm.py`, `collect_traces.py` (char-LM usage only), `collect_trace_nat_mul_add_mod.py`, `char_lm_ckpt.pt`
- **Change:** Move to `legacy/` directory. Remove `charlm` pipeline from `run_pipeline.py`. Keep a one-line note in docs.
- **Breaking:** Yes for anyone using `--pipeline charlm`.
- **Test:** `run_pipeline.py --help` no longer shows `charlm`.

### 3.2 Unify beam search implementations
- **Files:** `search_generate_traces.py`, `search_baseline.py`
- **Change:** Extract beam logic into `beam_engine.py`. Make `search_baseline.py` a thin wrapper that calls the engine without trace logging. Keep `search_generate_traces.py` as the main entry with trace output.
- **Breaking:** No.
- **Test:** Both scripts produce identical beam expansion on `nat_single`.

### 3.3 Move default outputs into run directories
- **Files:** `run_pipeline.py`, `build_sft_dataset.py`, `search_generate_traces.py`
- **Change:** Default `--search-out` and `--sft-out` to paths inside the run directory instead of repo root.
- **Breaking:** Yes — existing scripts assuming root-level files need flag updates.
- **Test:** After a pipeline run, no new files appear in repo root.

### 3.4 Add a consolidated README
- **Files:** New `README.md`
- **Change:** Single entry point doc with: install, quickstart, pipeline diagram, link to detailed docs.
- **Breaking:** No.
- **Test:** A new contributor can run a dry-run pipeline from the README alone.
