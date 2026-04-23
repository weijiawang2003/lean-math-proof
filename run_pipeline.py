"""One-command research pipeline for this repo.

Examples:
  python run_pipeline.py --dry-run
  python run_pipeline.py --theorem-set toy_search --auto-eval
  python run_pipeline.py --theorem-set nat_single --action-space core_v1 --seed 123
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
import sys

from actions import list_action_spaces


class PipelinePrecheckError(RuntimeError):
    pass


class PipelineCommandError(RuntimeError):
    pass


def _missing_modules(modules: list[str]) -> list[str]:
    missing: list[str] = []
    for module in modules:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return missing


def _ensure_modules(modules: list[str], pipeline_name: str) -> None:
    missing = _missing_modules(modules)
    if not missing:
        return

    module_list = ", ".join(missing)
    suggested = " ".join(sorted(set(missing)))
    raise PipelinePrecheckError(
        f"Missing required Python module(s) for pipeline '{pipeline_name}': {module_list}.\n"
        f"Install them first (example): pip install {suggested}"
    )


def _run(cmd: list[str], dry_run: bool) -> None:
    print("$", shlex.join(cmd))
    if dry_run:
        return
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise PipelineCommandError(
            f"Command failed (exit={exc.returncode}): {shlex.join(cmd)}"
        ) from exc


def classifier_pipeline(args: argparse.Namespace) -> None:
    """Search -> SFT dataset -> classifier train -> model rollout."""
    if not args.dry_run:
        _ensure_modules(["lean_dojo", "torch", "transformers", "accelerate"], pipeline_name="classifier")

    # 1) Search / trace generation
    _run(
        [
            sys.executable,
            "search_generate_traces.py",
            "--theorem-set", args.theorem_set,
            "--beam-width", str(args.beam_width),
            "--max-depth", str(args.max_depth),
            "--action-space", args.action_space,
            "--out", args.search_out,
            "--out-dir", args.out_dir,
            *(["--fail-on-skip"] if args.fail_on_skip else []),
            *(["--fail-on-unavailable"] if args.fail_on_unavailable else []),
        ],
        args.dry_run,
    )

    # 2) SFT dataset build
    sft_cmd = [
        sys.executable,
        "build_sft_dataset.py",
        "--in", args.search_out,
        "--out", args.sft_out,
        "--action-space", args.action_space,
        "--min-goal-drop", str(args.min_goal_drop),
        "--max-per-label", str(args.max_per_label),
        *([] if args.dedup_state_action else ["--no-dedup-state-action"]),
    ]
    if args.include_metadata:
        sft_cmd.append("--include-metadata")
    _run(sft_cmd, args.dry_run)

    # 3) Classifier training
    _run(
        [
            sys.executable,
            "train_action_classifier.py",
            "--sft-path", args.sft_out,
            "--action-space", args.action_space,
            "--output-dir", args.ckpt_dir,
            "--seed", str(args.seed),
            "--val-split", str(args.val_split),
        ],
        args.dry_run,
    )

    # 4) Policy rollout
    rollout_set = args.rollout_theorem_set or args.theorem_set
    if args.eval_all:
        _run(
            [
                sys.executable,
                "eval_rollout_all.py",
                "--theorem-set", rollout_set,
                "--max-steps", str(args.rollout_max_steps),
                "--top-k", str(args.top_k),
                "--out-dir", args.out_dir,
                "--ckpt-dir", args.ckpt_dir,
            ],
            args.dry_run,
        )
    else:
        _run(
            [
                sys.executable,
                "model_rollout.py",
                "--theorem-set", rollout_set,
                "--theorem-index", str(args.rollout_theorem_index),
                "--max-steps", str(args.rollout_max_steps),
                "--out-dir", args.out_dir,
                "--ckpt-dir", args.ckpt_dir,
            ],
            args.dry_run,
        )

    # 5) Optional evaluation
    if args.auto_eval:
        _run([sys.executable, "evaluate_traces.py", "--in", args.search_out], args.dry_run)
        _run([sys.executable, "compare_runs.py", "--runs-dir", args.out_dir], args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the classifier pipeline end-to-end.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only; do not execute.")

    # Output / env
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--ckpt-dir", default="clf_ckpt", help="Classifier checkpoint directory.")

    # Search stage
    parser.add_argument("--theorem-set", default="nat_single")
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--action-space", default="search_v3", choices=list_action_spaces())
    parser.add_argument("--fail-on-skip", action="store_true",
                        help="Fail if any theorem is skipped in search.")
    parser.add_argument("--fail-on-unavailable", action="store_true",
                        help="Fail if theorem availability precheck filters any theorem.")
    parser.add_argument("--search-out", default="traces_from_search.jsonl")

    # SFT build stage
    parser.add_argument("--sft-out", default="sft_dataset.jsonl")
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--min-goal-drop", type=int, default=1,
                        help="SFT filter: keep rows with (goals_before-goals_after)>=this.")
    parser.add_argument("--max-per-label", type=int, default=64,
                        help="SFT label cap (0 = unlimited).")
    parser.add_argument("--dedup-state-action", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether SFT builder deduplicates (state,tactic).")

    # Training stage
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training.")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data for validation (0 to disable).")

    # Rollout stage
    parser.add_argument("--rollout-theorem-set", default="",
                        help="Default: follow --theorem-set when empty.")
    parser.add_argument("--rollout-theorem-index", type=int, default=0)
    parser.add_argument("--rollout-max-steps", type=int, default=5)

    # Rollout strategy
    parser.add_argument("--top-k", type=int, default=5,
                        help="Try up to k tactics per step before declaring failure.")

    # Evaluation
    parser.add_argument("--eval-all", action="store_true",
                        help="Rollout on ALL theorems in the set (instead of one --theorem-index).")
    parser.add_argument("--auto-eval", action="store_true",
                        help="Run evaluate_traces and compare_runs after pipeline.")

    args = parser.parse_args()

    try:
        classifier_pipeline(args)
    except PipelinePrecheckError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    except PipelineCommandError as exc:
        print(str(exc), file=sys.stderr)
        if "--fail-on-unavailable" in str(exc):
            print(
                "Hint: remove --fail-on-unavailable to continue with available theorems, "
                "or prepare missing LeanDojo trace artifacts first.",
                file=sys.stderr,
            )
        raise SystemExit(4) from exc


if __name__ == "__main__":
    main()
