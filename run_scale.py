"""One-command scaling: extract 200+ theorems → search → train → eval → self-play → report.

This is the "big red button" script that takes the system from its current
state to maximum scale.  It orchestrates everything through the incremental
pipeline so all progress is saved and nothing is repeated.

Workflow:
  1. Extract theorems from Lean files (TracedRepo)
  2. Register into incremental project state
  3. Search all unsearched theorems (beam search with action space)
  4. Train generative model on accumulated traces
  5. Evaluate on all theorems
  6. Self-play round: discover novel tactics
  7. Retrain on enriched corpus
  8. Final evaluation + report

Usage:
  python run_scale.py                           # full pipeline, core files
  python run_scale.py --preset extended         # scan more files
  python run_scale.py --skip-extraction         # reuse discovered theorems
  python run_scale.py --rounds 3                # 3 self-play refinement rounds
  python run_scale.py --dry-run                 # print commands without executing
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], dry_run: bool, label: str = "", critical: bool = True) -> bool:
    if label:
        print(f"\n{'='*64}")
        print(f"  {label}")
        print(f"{'='*64}")
    print("$", shlex.join(cmd))
    if dry_run:
        return True
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        msg = f"Stage failed (exit={exc.returncode}): {shlex.join(cmd)}"
        if critical:
            raise RuntimeError(msg) from exc
        print(f"  [WARN] {msg}  -- continuing")
        return False


def main():
    py = sys.executable

    parser = argparse.ArgumentParser(
        description="One-command scaling: extract → search → train → eval → selfplay."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project", default="project")

    # Extraction
    parser.add_argument("--preset", default="core",
                        choices=["core", "extended", "all_known"])
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--check-availability", action="store_true",
                        help="Verify each extracted theorem in Dojo (slow but accurate).")
    parser.add_argument("--max-per-file", type=int, default=0,
                        help="Cap theorems per file (0 = unlimited).")

    # Search
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--action-space", default="search_v4")
    parser.add_argument("--search-limit", type=int, default=0,
                        help="Max theorems to search per phase (0 = all).")

    # Training
    parser.add_argument("--model-type", default="gen",
                        choices=["gen", "decoder", "classifier"])
    parser.add_argument("--base-model", default="t5-small")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)

    # Self-play
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of self-play refinement rounds after initial training.")
    parser.add_argument("--selfplay-temperature", type=float, default=1.2)
    parser.add_argument("--selfplay-top-k", type=int, default=12)

    # Eval
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)

    args = parser.parse_args()

    discovered_path = str(Path(args.project) / "discovered_theorems.json")

    print("=" * 64)
    print("  LEAN THEOREM PROVER — FULL SCALE PIPELINE")
    print("=" * 64)
    print(f"  Project      : {args.project}")
    print(f"  File preset  : {args.preset}")
    print(f"  Model type   : {args.model_type} ({args.base_model})")
    print(f"  Self-play    : {args.rounds} round(s)")
    print(f"  Search       : beam={args.beam_width}, depth={args.max_depth}")
    print("=" * 64)

    # ──────────────────────────────────────────────────────────────
    # PHASE 1: Extract theorems from Lean files
    # ──────────────────────────────────────────────────────────────
    if not args.skip_extraction:
        extract_cmd = [
            py, "extract_theorems.py",
            "--preset", args.preset,
            "--out", discovered_path,
        ]
        if args.check_availability:
            extract_cmd.append("--check-availability")
        if args.max_per_file > 0:
            extract_cmd.extend(["--max-per-file", str(args.max_per_file)])

        _run(extract_cmd, args.dry_run,
             label="PHASE 1: Extract theorems from Lean files",
             critical=False)
    else:
        print("\n  [SKIP] Phase 1 — reusing existing discovered theorems")

    # ──────────────────────────────────────────────────────────────
    # PHASE 2: Register extracted theorems
    # ──────────────────────────────────────────────────────────────
    if Path(discovered_path).exists() or args.dry_run:
        _run([
            py, "run_incremental.py", "--project", args.project,
            "register", "--from-json", discovered_path,
        ], args.dry_run, label="PHASE 2: Register extracted theorems")

    # Also register the hand-curated curriculum sets
    for tier in ["curriculum_tier1", "curriculum_tier2", "curriculum_tier3"]:
        _run([
            py, "run_incremental.py", "--project", args.project,
            "register", "--theorem-set", tier,
        ], args.dry_run, label=f"PHASE 2b: Register {tier}",
            critical=False)

    # ──────────────────────────────────────────────────────────────
    # PHASE 3: Search all unsearched theorems
    # ──────────────────────────────────────────────────────────────
    search_cmd = [
        py, "run_incremental.py", "--project", args.project,
        "search",
        "--beam-width", str(args.beam_width),
        "--max-depth", str(args.max_depth),
        "--action-space", args.action_space,
    ]
    if args.search_limit > 0:
        search_cmd.extend(["--limit", str(args.search_limit)])

    _run(search_cmd, args.dry_run,
         label="PHASE 3: Search unsearched theorems (beam search)",
         critical=False)

    # ──────────────────────────────────────────────────────────────
    # PHASE 4: Train generative model
    # ──────────────────────────────────────────────────────────────
    _run([
        py, "run_incremental.py", "--project", args.project,
        "train",
        "--type", args.model_type,
        "--base-model", args.base_model,
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
    ], args.dry_run, label="PHASE 4: Train generative tactic model")

    # ──────────────────────────────────────────────────────────────
    # PHASE 5: Initial evaluation
    # ──────────────────────────────────────────────────────────────
    _run([
        py, "run_incremental.py", "--project", args.project,
        "eval", "--all",
        "--max-steps", str(args.max_steps),
        "--top-k", str(args.top_k),
    ], args.dry_run, label="PHASE 5: Initial evaluation on all theorems",
        critical=False)

    # ──────────────────────────────────────────────────────────────
    # PHASE 6: Self-play refinement rounds
    # ──────────────────────────────────────────────────────────────
    for rnd in range(1, args.rounds + 1):
        _run([
            py, "run_incremental.py", "--project", args.project,
            "selfplay",
            "--top-k", str(args.selfplay_top_k),
            "--temperature", str(args.selfplay_temperature),
            "--max-steps", str(args.max_steps),
        ], args.dry_run,
            label=f"PHASE 6.{rnd}: Self-play refinement round {rnd}/{args.rounds}",
            critical=False)

        # Retrain on enriched corpus
        _run([
            py, "run_incremental.py", "--project", args.project,
            "train",
            "--type", args.model_type,
            "--base-model", args.base_model,
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
        ], args.dry_run,
            label=f"PHASE 6.{rnd}b: Retrain after self-play round {rnd}",
            critical=False)

    # ──────────────────────────────────────────────────────────────
    # PHASE 7: Final evaluation
    # ──────────────────────────────────────────────────────────────
    _run([
        py, "run_incremental.py", "--project", args.project,
        "eval", "--all",
        "--max-steps", str(args.max_steps),
        "--top-k", str(args.top_k),
    ], args.dry_run, label="PHASE 7: Final evaluation",
        critical=False)

    # ──────────────────────────────────────────────────────────────
    # PHASE 8: Status report
    # ──────────────────────────────────────────────────────────────
    _run([
        py, "run_incremental.py", "--project", args.project,
        "status",
    ], args.dry_run, label="PHASE 8: Final status report")

    if not args.dry_run:
        # Save a scaling report
        report_path = Path(args.project) / "scale_report.json"
        try:
            from project_state import ProjectState
            state = ProjectState(args.project)
            proved = state.get_proved()
            total = sum(1 for t in state.theorems.values() if t.get("available") is not False)
            report = {
                "total_registered": len(state.theorems),
                "total_available": total,
                "total_proved": len(proved),
                "success_rate": len(proved) / total if total else 0,
                "total_traces": state.total_traces,
                "models_trained": len(state.models),
                "preset": args.preset,
                "model_type": args.model_type,
                "base_model": args.base_model,
                "selfplay_rounds": args.rounds,
            }
            report_path.write_text(json.dumps(report, indent=2))
            print(f"\n  Scale report saved: {report_path}")
        except Exception as exc:
            print(f"\n  [WARN] Could not generate report: {exc}")

    print("\n" + "=" * 64)
    print("  SCALING COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()
