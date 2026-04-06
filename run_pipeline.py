"""One-command research pipelines for this repo.

Examples:
  python run_pipeline.py --pipeline classifier --dry-run
  python run_pipeline.py --pipeline classifier --theorem-set toy_search
  python run_pipeline.py --pipeline charlm
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], dry_run: bool) -> None:
    print("$", shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def classifier_pipeline(args: argparse.Namespace) -> None:
    """Search -> SFT dataset -> classifier train -> model rollout."""
    _run(
        [
            sys.executable,
            "search_generate_traces.py",
            "--theorem-set",
            args.theorem_set,
            "--beam-width",
            str(args.beam_width),
            "--max-depth",
            str(args.max_depth),
            "--out",
            args.search_out,
            "--out-dir",
            args.out_dir,
        ],
        args.dry_run,
    )

    sft_cmd = [
        sys.executable,
        "build_sft_dataset.py",
        "--in",
        args.search_out,
        "--out",
        args.sft_out,
    ]
    if args.include_metadata:
        sft_cmd.append("--include-metadata")
    _run(sft_cmd, args.dry_run)

    _run([sys.executable, "train_action_classifier.py"], args.dry_run)

    _run(
        [
            sys.executable,
            "model_rollout.py",
            "--theorem-set",
            args.rollout_theorem_set,
            "--theorem-index",
            str(args.rollout_theorem_index),
            "--max-steps",
            str(args.rollout_max_steps),
            "--out-dir",
            args.out_dir,
        ],
        args.dry_run,
    )


def charlm_pipeline(args: argparse.Namespace) -> None:
    """Scripted traces -> char-LM train -> optional generation."""
    _run(
        [
            sys.executable,
            "collect_traces.py",
            "--out",
            args.charlm_trace_out,
            "--domain",
            args.domain,
            "--out-dir",
            args.out_dir,
        ],
        args.dry_run,
    )
    _run([sys.executable, "train_sft_char_lm.py", "--mode", "train"], args.dry_run)

    if args.charlm_state_file:
        if not Path(args.charlm_state_file).exists():
            raise FileNotFoundError(f"--charlm-state-file not found: {args.charlm_state_file}")
        _run(
            [
                sys.executable,
                "train_sft_char_lm.py",
                "--mode",
                "gen",
                "--state-file",
                args.charlm_state_file,
            ],
            args.dry_run,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one full pipeline for this repo.")
    parser.add_argument("--pipeline", choices=["classifier", "charlm"], default="classifier")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only; do not execute.")

    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--domain", default="mathlib4_nat_easy")

    parser.add_argument("--theorem-set", default="toy_search")
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--search-out", default="traces_from_search.jsonl")
    parser.add_argument("--sft-out", default="sft_dataset.jsonl")
    parser.add_argument("--include-metadata", action="store_true")

    parser.add_argument("--rollout-theorem-set", default="nat_single")
    parser.add_argument("--rollout-theorem-index", type=int, default=0)
    parser.add_argument("--rollout-max-steps", type=int, default=5)

    parser.add_argument("--charlm-trace-out", default="traces.jsonl")
    parser.add_argument("--charlm-state-file", default="")

    args = parser.parse_args()

    if args.pipeline == "classifier":
        classifier_pipeline(args)
    else:
        charlm_pipeline(args)


if __name__ == "__main__":
    main()
