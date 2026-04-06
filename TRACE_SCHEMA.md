# Trace and Episode Schema

This project stores proof interaction as JSONL rows. Each line is one **transition** (`state` + `tactic` + `result`).

## TransitionRecord fields

`TransitionRecord` is defined in `core_types.py`.

Required fields:
- `file_path: str` — Lean file path.
- `full_name: str` — theorem full name.
- `state_pp: str` — pretty-printed tactic state before action.
- `tactic: str` — tactic/action applied.
- `result_kind: str` — LeanDojo result type name.
- `proof_finished: bool` — whether theorem was proved at this step.

Optional fields (used for analysis/training):
- `num_goals_before: int | None`
- `num_goals_after: int | None`
- `step: int | None` — step index in episode.
- `domain: str | None` — optional tag for dataset split/domain.
- `error_message: str | None` — Lean error text for failed actions.
- `run_id: str | None` — identifier for one script run.
- `episode_id: str | None` — identifier for one theorem attempt trajectory.
- `method: str | None` — e.g. `policy_rollout`, `beam_search`, `manual`.

## Episode and run interpretation

- A **run** is identified by `run_id`.
- An **episode** is identified by `(run_id, episode_id)`.
- For reliable analysis, each script run should write transitions with stable `run_id`, and per-theorem attempts should have unique `episode_id`.

If these fields are unavailable in older data, episodes can be approximated by:
- grouping by theorem (`file_path`, `full_name`) and
- ordering by `step` where present.

## Output layout convention

Experiment scripts should write to:

- `runs/<run_id>/config.json` — run configuration snapshot,
- `runs/<run_id>/traces.jsonl` — transition records,
- `runs/<run_id>/metrics.json` — run/episode evaluation summary.

This convention is implemented by `experiment_io.py` and used in rollout/collection/search scripts.

## Sufficiency review for future training workflows

Current schema is sufficient for:
- supervised action prediction (`state_pp` -> `tactic`),
- filtering by progress (`num_goals_before/after`),
- error-aware analysis (`error_message`, `result_kind`),
- run/episode/method-level aggregation (`run_id`, `episode_id`, `method`).

Smallest likely additions when scaling:
- `split` (train/val/test) and `source` tags for data provenance,
- `parent_transition_id` for explicit search-tree replay,
- stable theorem ID/hash for cross-version deduplication.

Keep these optional to preserve script simplicity.
