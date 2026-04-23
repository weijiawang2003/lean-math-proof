"""Full generative curriculum pipeline: extract theorems → search → train seq2seq → rollout.

This is the next-generation pipeline that combines:
  - Automated theorem extraction from Lean files (no hand-curation)
  - Generative tactic model (no fixed action space ceiling)
  - Curriculum learning (easy → hard bootstrapping)

The pipeline can also compare classifier vs generative results side-by-side.

Usage:
  python run_generative_curriculum.py --dry-run
  python run_generative_curriculum.py
  python run_generative_curriculum.py --skip-extraction    # reuse discovered_theorems.json
  python run_generative_curriculum.py --compare-classifier # also train+eval a classifier for comparison
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
    print(f"\n  Merging {len(paths)} trace files → {out_path}")
    if dry_run:
        return
    total = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for p in paths:
            if not Path(p).exists():
                print(f"  [WARN] Not found, skipping: {p}")
                continue
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
                        total += 1
    print(f"  Merged {total} total transitions")


def _load_json(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _find_latest_metrics(run_dir: str) -> dict:
    rd = Path(run_dir)
    if not rd.exists():
        return {}
    files = sorted(rd.glob("*/metrics.json"))
    return _load_json(str(files[-1])) if files else {}


def main():
    py = sys.executable

    parser = argparse.ArgumentParser(
        description="Generative curriculum pipeline: extract → search → train seq2seq → rollout."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Reuse existing discovered_theorems.json.")
    parser.add_argument("--compare-classifier", action="store_true",
                        help="Also train and eval a classifier for side-by-side comparison.")

    # Search params
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--action-space", default="search_v4")

    # Model params
    parser.add_argument("--gen-model", default="t5-small",
                        help="Base model for generative policy.")
    parser.add_argument("--gen-epochs", type=int, default=15)

    # Eval params
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="gen_curriculum_runs")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Paths
    discovered_path   = str(out / "discovered_theorems.json")
    tier1_traces      = str(out / "tier1_traces.jsonl")
    tier2_traces      = str(out / "tier2_traces.jsonl")
    merged_traces     = str(out / "merged_traces.jsonl")
    seq2seq_data      = str(out / "seq2seq_data.jsonl")
    gen_ckpt          = str(out / "gen_ckpt")
    clf_sft           = str(out / "clf_sft.jsonl")
    clf_ckpt          = str(out / "clf_ckpt")

    print("=" * 64)
    print("  GENERATIVE CURRICULUM PIPELINE")
    print("=" * 64)
    print(f"  Base model   : {args.gen_model}")
    print(f"  Action space : {args.action_space} (for search only)")
    print(f"  Beam width   : {args.beam_width}")
    print(f"  Max depth    : {args.max_depth}")
    print(f"  Top-k        : {args.top_k}")
    print(f"  Output dir   : {args.out_dir}")
    print("=" * 64)

    # ==================================================================
    # STAGE 1: Extract theorems from Lean files
    # ==================================================================
    if not args.skip_extraction:
        _run([
            py, "extract_theorems.py",
            "--out", discovered_path,
            "--check-availability",
        ], args.dry_run, label="STAGE 1: Extract theorems from mathlib4 files")
    else:
        print("\n  [SKIP] Stage 1 — reusing existing discovered theorems")

    # ==================================================================
    # STAGE 2: Search on tier1 (easy) theorems
    # ==================================================================
    _run([
        py, "search_generate_traces.py",
        "--theorem-set", "curriculum_tier1",
        "--beam-width", str(args.beam_width),
        "--max-depth", str(args.max_depth),
        "--action-space", args.action_space,
        "--out", tier1_traces,
        "--out-dir", str(out / "stage2_search_tier1"),
    ], args.dry_run, label="STAGE 2: Search on tier1 (easy theorems)")

    # ==================================================================
    # STAGE 3: Search on tier2 (medium) theorems
    # ==================================================================
    _run([
        py, "search_generate_traces.py",
        "--theorem-set", "curriculum_tier2",
        "--beam-width", str(args.beam_width),
        "--max-depth", str(args.max_depth),
        "--action-space", args.action_space,
        "--out", tier2_traces,
        "--out-dir", str(out / "stage3_search_tier2"),
    ], args.dry_run, label="STAGE 3: Search on tier2 (medium theorems)",
        critical=False)

    # ==================================================================
    # STAGE 4: Merge traces → build seq2seq dataset → train generator
    # ==================================================================
    _merge_jsonl([tier1_traces, tier2_traces], merged_traces, args.dry_run)

    _run([
        py, "build_seq2seq_dataset.py",
        "--in", merged_traces,
        "--out", seq2seq_data,
        "--min-goal-drop", "1",
    ], args.dry_run, label="STAGE 4a: Build seq2seq dataset from merged traces")

    _run([
        py, "train_tactic_generator.py",
        "--data", seq2seq_data,
        "--model", args.gen_model,
        "--output-dir", gen_ckpt,
        "--epochs", str(args.gen_epochs),
        "--seed", str(args.seed),
        "--val-split", "0.1",
    ], args.dry_run, label="STAGE 4b: Train generative tactic model (CodeT5)")

    # ==================================================================
    # STAGE 5: Evaluate generative model on all tiers
    # ==================================================================
    for tier, tier_name in [("curriculum_tier1", "tier1"), ("curriculum_tier2", "tier2"),
                            ("curriculum_tier3", "tier3"), ("curriculum_all", "all")]:
        _run([
            py, "eval_rollout_all.py",
            "--theorem-set", tier,
            "--ckpt-dir", gen_ckpt,
            "--policy-type", "generative",
            "--max-steps", str(args.max_steps),
            "--top-k", str(args.top_k),
            "--out-dir", str(out / f"stage5_gen_{tier_name}"),
        ], args.dry_run,
            label=f"STAGE 5: Eval generative model on {tier_name}",
            critical=False)

    # ==================================================================
    # STAGE 6 (optional): Train + eval classifier for comparison
    # ==================================================================
    if args.compare_classifier:
        _run([
            py, "build_sft_dataset.py",
            "--in", merged_traces,
            "--out", clf_sft,
            "--action-space", args.action_space,
            "--min-goal-drop", "1",
            "--max-per-label", "64",
        ], args.dry_run, label="STAGE 6a: Build classifier SFT dataset")

        _run([
            py, "train_action_classifier.py",
            "--sft-path", clf_sft,
            "--action-space", args.action_space,
            "--output-dir", clf_ckpt,
            "--seed", str(args.seed),
            "--val-split", "0.1",
        ], args.dry_run, label="STAGE 6b: Train classifier for comparison")

        for tier, tier_name in [("curriculum_all", "all")]:
            _run([
                py, "eval_rollout_all.py",
                "--theorem-set", tier,
                "--ckpt-dir", clf_ckpt,
                "--policy-type", "classifier",
                "--max-steps", str(args.max_steps),
                "--top-k", str(args.top_k),
                "--out-dir", str(out / f"stage6_clf_{tier_name}"),
            ], args.dry_run,
                label=f"STAGE 6c: Eval classifier on {tier_name}",
                critical=False)

    # ==================================================================
    # STAGE 7: Comparative report
    # ==================================================================
    if args.dry_run:
        print("\n[DRY RUN] Skipping report generation.")
        return

    print("\n")
    print("=" * 64)
    print("  GENERATIVE CURRICULUM RESULTS")
    print("=" * 64)

    gen_results = {}
    for tier_name in ["tier1", "tier2", "tier3", "all"]:
        m = _find_latest_metrics(str(out / f"stage5_gen_{tier_name}"))
        gen_results[tier_name] = m
        proved = m.get("proved", "?")
        avail = m.get("available", "?")
        rate = m.get("success_rate", 0)
        print(f"  Generative on {tier_name:5s}: {proved}/{avail} proved ({rate:.0%})")

    if args.compare_classifier:
        clf_m = _find_latest_metrics(str(out / "stage6_clf_all"))
        clf_proved = clf_m.get("proved", "?")
        clf_avail = clf_m.get("available", "?")
        clf_rate = clf_m.get("success_rate", 0)
        gen_all = gen_results.get("all", {})
        gen_rate = gen_all.get("success_rate", 0)

        print(f"\n  {'─'*50}")
        print(f"  CLASSIFIER vs GENERATIVE (on curriculum_all):")
        print(f"    Classifier:  {clf_proved}/{clf_avail} ({clf_rate:.0%})")
        print(f"    Generative:  {gen_all.get('proved', '?')}/{gen_all.get('available', '?')} ({gen_rate:.0%})")
        if isinstance(gen_rate, (int, float)) and isinstance(clf_rate, (int, float)):
            delta = gen_rate - clf_rate
            direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            print(f"    Delta:       {direction} {abs(delta):.0%}")

    print("=" * 64)

    # Save report
    report = {
        "generative": {k: {"proved": v.get("proved"), "available": v.get("available"),
                           "rate": v.get("success_rate")} for k, v in gen_results.items()},
    }
    if args.compare_classifier:
        report["classifier_all"] = {
            "proved": clf_m.get("proved"), "available": clf_m.get("available"),
            "rate": clf_m.get("success_rate"),
        }

    report_path = out / "generative_curriculum_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
