"""Curriculum learning pipeline for Lean theorem proving.

Demonstrates bootstrapping: train on easy theorems, then progressively
attempt harder ones — showing that knowledge transfers upward.

Stages:
  1. Search + train on tier1 (easy theorems)
  2. Zero-shot evaluate tier1 model on tier2 (transfer test)
  3. Search on tier2 with expanded action space
  4. Merge tier1+tier2 traces, retrain on combined data
  5. Evaluate retrained model on tier2 (improvement) + tier3 (stretch)
  6. Print comparative report

Usage:
  python run_curriculum.py --dry-run
  python run_curriculum.py
  python run_curriculum.py --skip-stage1   # reuse existing tier1 artifacts
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


class StageError(RuntimeError):
    pass


def _run(cmd: list[str], dry_run: bool, label: str = "", critical: bool = True) -> bool:
    """Run a command. Returns True on success, False on failure.

    If ``critical=True`` (default for training stages), failures raise.
    If ``critical=False`` (eval/search stages), failures are logged and skipped.
    """
    if label:
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")
    print("$", shlex.join(cmd))
    if dry_run:
        return True
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        msg = f"Stage failed (exit={exc.returncode}): {shlex.join(cmd)}"
        if critical:
            raise StageError(msg) from exc
        print(f"  [WARN] {msg}  — continuing")
        return False


def _merge_jsonl(paths: list[str], out_path: str, dry_run: bool) -> None:
    """Concatenate multiple JSONL files into one."""
    print(f"\n  Merging {len(paths)} trace files → {out_path}")
    if dry_run:
        return
    total = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for p in paths:
            if not Path(p).exists():
                print(f"  [WARN] Trace file not found, skipping: {p}")
                continue
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
                        total += 1
    print(f"  Merged {total} total transitions")


def _load_metrics(path: str) -> dict:
    """Load metrics JSON, return empty dict on failure."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    py = sys.executable

    parser = argparse.ArgumentParser(
        description="Curriculum learning pipeline: easy → medium → hard."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true",
                        help="Reuse existing tier1 artifacts (skip search+train).")
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--action-space", default="search_v4",
                        help="Action space for search stages.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="curriculum_runs")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # File paths for each stage
    tier1_traces = str(out / "tier1_traces.jsonl")
    tier1_sft    = str(out / "tier1_sft.jsonl")
    tier1_ckpt   = str(out / "tier1_ckpt")

    tier2_traces = str(out / "tier2_traces.jsonl")
    merged_traces = str(out / "merged_traces.jsonl")
    merged_sft   = str(out / "merged_sft.jsonl")
    merged_ckpt  = str(out / "merged_ckpt")

    # Metrics files for the report
    eval_tier1_on_tier2 = str(out / "eval_tier1model_on_tier2.json")
    eval_merged_on_tier2 = str(out / "eval_mergedmodel_on_tier2.json")
    eval_merged_on_tier3 = str(out / "eval_mergedmodel_on_tier3.json")
    eval_merged_on_all  = str(out / "eval_mergedmodel_on_all.json")

    print("=" * 64)
    print("  CURRICULUM LEARNING PIPELINE")
    print("=" * 64)
    print(f"  Action space : {args.action_space}")
    print(f"  Beam width   : {args.beam_width}")
    print(f"  Max depth    : {args.max_depth}")
    print(f"  Top-k        : {args.top_k}")
    print(f"  Max steps    : {args.max_steps}")
    print(f"  Output dir   : {args.out_dir}")
    print("=" * 64)

    # ==================================================================
    # STAGE 1: Search + train on tier1 (easy theorems)
    # ==================================================================
    if not args.skip_stage1:
        _run([
            py, "search_generate_traces.py",
            "--theorem-set", "curriculum_tier1",
            "--beam-width", str(args.beam_width),
            "--max-depth", str(args.max_depth),
            "--action-space", args.action_space,
            "--out", tier1_traces,
            "--out-dir", str(out / "stage1_runs"),
        ], args.dry_run, label="STAGE 1a: Search on tier1 (easy theorems)")

        _run([
            py, "build_sft_dataset.py",
            "--in", tier1_traces,
            "--out", tier1_sft,
            "--action-space", args.action_space,
            "--min-goal-drop", "1",
            "--max-per-label", "64",
        ], args.dry_run, label="STAGE 1b: Build SFT from tier1 traces")

        _run([
            py, "train_action_classifier.py",
            "--sft-path", tier1_sft,
            "--action-space", args.action_space,
            "--output-dir", tier1_ckpt,
            "--seed", str(args.seed),
            "--val-split", "0.1",
        ], args.dry_run, label="STAGE 1c: Train classifier on tier1")
    else:
        print("\n  [SKIP] Stage 1 — reusing existing tier1 artifacts")

    # ==================================================================
    # STAGE 2: Zero-shot evaluate tier1 model on tier2
    # ==================================================================
    _run([
        py, "eval_rollout_all.py",
        "--theorem-set", "curriculum_tier2",
        "--ckpt-dir", tier1_ckpt,
        "--max-steps", str(args.max_steps),
        "--top-k", str(args.top_k),
        "--out-dir", str(out / "stage2_zeroshot"),
    ], args.dry_run, label="STAGE 2: Zero-shot eval of tier1 model on tier2 (TRANSFER TEST)",
        critical=False)

    # ==================================================================
    # STAGE 3: Search on tier2 with expanded action space
    # ==================================================================
    _run([
        py, "search_generate_traces.py",
        "--theorem-set", "curriculum_tier2",
        "--beam-width", str(args.beam_width),
        "--max-depth", str(args.max_depth),
        "--action-space", args.action_space,
        "--out", tier2_traces,
        "--out-dir", str(out / "stage3_runs"),
    ], args.dry_run, label="STAGE 3: Search on tier2 (medium theorems)",
        critical=False)

    # ==================================================================
    # STAGE 4: Merge traces, retrain
    # ==================================================================
    _merge_jsonl([tier1_traces, tier2_traces], merged_traces, args.dry_run)

    _run([
        py, "build_sft_dataset.py",
        "--in", merged_traces,
        "--out", merged_sft,
        "--action-space", args.action_space,
        "--min-goal-drop", "1",
        "--max-per-label", "64",
    ], args.dry_run, label="STAGE 4a: Build SFT from merged tier1+tier2 traces")

    _run([
        py, "train_action_classifier.py",
        "--sft-path", merged_sft,
        "--action-space", args.action_space,
        "--output-dir", merged_ckpt,
        "--seed", str(args.seed),
        "--val-split", "0.1",
    ], args.dry_run, label="STAGE 4b: Retrain classifier on merged data")

    # ==================================================================
    # STAGE 5: Evaluate retrained model
    # ==================================================================
    _run([
        py, "eval_rollout_all.py",
        "--theorem-set", "curriculum_tier2",
        "--ckpt-dir", merged_ckpt,
        "--max-steps", str(args.max_steps),
        "--top-k", str(args.top_k),
        "--out-dir", str(out / "stage5_tier2_retrained"),
    ], args.dry_run, label="STAGE 5a: Eval retrained model on tier2 (expect improvement)",
        critical=False)

    _run([
        py, "eval_rollout_all.py",
        "--theorem-set", "curriculum_tier3",
        "--ckpt-dir", merged_ckpt,
        "--max-steps", str(args.max_steps),
        "--top-k", str(args.top_k),
        "--out-dir", str(out / "stage5_tier3"),
    ], args.dry_run, label="STAGE 5b: Eval retrained model on tier3 (stretch goals)",
        critical=False)

    _run([
        py, "eval_rollout_all.py",
        "--theorem-set", "curriculum_all",
        "--ckpt-dir", merged_ckpt,
        "--max-steps", str(args.max_steps),
        "--top-k", str(args.top_k),
        "--out-dir", str(out / "stage5_all"),
    ], args.dry_run, label="STAGE 5c: Eval retrained model on ALL tiers combined",
        critical=False)

    # ==================================================================
    # STAGE 6: Comparative report
    # ==================================================================
    if args.dry_run:
        print("\n[DRY RUN] Skipping report generation.")
        return

    print("\n")
    print("=" * 64)
    print("  CURRICULUM LEARNING RESULTS")
    print("=" * 64)

    # Collect results from eval runs
    def _find_latest_metrics(run_dir: str) -> dict:
        """Find the most recent metrics.json under a run directory."""
        rd = Path(run_dir)
        if not rd.exists():
            return {}
        metrics_files = sorted(rd.glob("*/metrics.json"))
        if not metrics_files:
            return {}
        return _load_metrics(str(metrics_files[-1]))

    zs_metrics = _find_latest_metrics(str(out / "stage2_zeroshot"))
    rt_tier2   = _find_latest_metrics(str(out / "stage5_tier2_retrained"))
    rt_tier3   = _find_latest_metrics(str(out / "stage5_tier3"))
    rt_all     = _find_latest_metrics(str(out / "stage5_all"))

    # --- Section 1: Transfer test ---
    zs_proved = zs_metrics.get("proved", "?")
    zs_avail  = zs_metrics.get("available", "?")
    zs_rate   = zs_metrics.get("success_rate", 0)

    print(f"\n  STAGE 2 — Zero-shot transfer (tier1 model → tier2 theorems):")
    print(f"    Proved: {zs_proved}/{zs_avail}  ({zs_rate:.0%})")

    # --- Section 2: After retraining ---
    rt2_proved = rt_tier2.get("proved", "?")
    rt2_avail  = rt_tier2.get("available", "?")
    rt2_rate   = rt_tier2.get("success_rate", 0)

    print(f"\n  STAGE 5a — Retrained model on tier2:")
    print(f"    Proved: {rt2_proved}/{rt2_avail}  ({rt2_rate:.0%})")

    if isinstance(zs_rate, (int, float)) and isinstance(rt2_rate, (int, float)):
        delta = rt2_rate - zs_rate
        direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"    Improvement over zero-shot: {direction} {abs(delta):.0%}")

    # --- Section 3: Tier3 stretch ---
    rt3_proved = rt_tier3.get("proved", "?")
    rt3_avail  = rt_tier3.get("available", "?")
    rt3_rate   = rt_tier3.get("success_rate", 0)

    print(f"\n  STAGE 5b — Retrained model on tier3 (hard):")
    print(f"    Proved: {rt3_proved}/{rt3_avail}  ({rt3_rate:.0%})")

    # --- Section 4: Overall ---
    all_proved = rt_all.get("proved", "?")
    all_avail  = rt_all.get("available", "?")
    all_rate   = rt_all.get("success_rate", 0)

    print(f"\n  STAGE 5c — Retrained model on ALL tiers:")
    print(f"    Proved: {all_proved}/{all_avail}  ({all_rate:.0%})")

    # --- Per-theorem breakdown from curriculum_all ---
    per_thm = rt_all.get("per_theorem", [])
    if per_thm:
        # Classify each by tier
        tier1_names = {t.full_name for t in __import__("tasks").get_theorems("curriculum_tier1")}
        tier2_names = {t.full_name for t in __import__("tasks").get_theorems("curriculum_tier2")}
        tier3_names = {t.full_name for t in __import__("tasks").get_theorems("curriculum_tier3")}

        tier_results = {"tier1": [], "tier2": [], "tier3": [], "unknown": []}
        for r in per_thm:
            name = r.get("full_name", "")
            if name in tier1_names:
                tier_results["tier1"].append(r)
            elif name in tier2_names:
                tier_results["tier2"].append(r)
            elif name in tier3_names:
                tier_results["tier3"].append(r)
            else:
                tier_results["unknown"].append(r)

        print(f"\n  {'─'*60}")
        print(f"  PER-TIER BREAKDOWN (from curriculum_all eval):")
        for tier_label in ["tier1", "tier2", "tier3"]:
            results = tier_results[tier_label]
            if not results:
                continue
            avail = [r for r in results if r.get("available", False)]
            proved = [r for r in avail if r.get("finished", False)]
            rate = len(proved) / len(avail) if avail else 0
            print(f"    {tier_label}: {len(proved)}/{len(avail)} proved ({rate:.0%})")
            for r in results:
                name = r.get("full_name", "?")[:38]
                if not r.get("available"):
                    status = "SKIP"
                elif r.get("finished"):
                    status = "PROVED"
                elif r.get("has_error"):
                    status = "ERROR"
                else:
                    status = "EXHAUST"
                print(f"      {name:<38s}  {status}")

    print(f"\n  {'─'*60}")
    print(f"  CURRICULUM STORY:")
    print(f"    1. Trained on {len(__import__('tasks').get_theorems('curriculum_tier1'))} easy theorems (tier1)")
    print(f"    2. Zero-shot transfer to tier2: {zs_proved}/{zs_avail} proved")
    print(f"    3. After retraining on combined data:")
    print(f"       - Tier2: {rt2_proved}/{rt2_avail} proved")
    print(f"       - Tier3 (never-seen hard): {rt3_proved}/{rt3_avail} proved")
    print(f"       - Overall: {all_proved}/{all_avail} proved")
    print("=" * 64)

    # Save combined report
    report = {
        "zero_shot_tier2": {
            "proved": zs_proved, "available": zs_avail, "rate": zs_rate,
        },
        "retrained_tier2": {
            "proved": rt2_proved, "available": rt2_avail, "rate": rt2_rate,
        },
        "retrained_tier3": {
            "proved": rt3_proved, "available": rt3_avail, "rate": rt3_rate,
        },
        "retrained_all": {
            "proved": all_proved, "available": all_avail, "rate": all_rate,
        },
    }
    report_path = out / "curriculum_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Full report saved: {report_path}")


if __name__ == "__main__":
    main()
