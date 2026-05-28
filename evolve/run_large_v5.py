"""Run a single eval of the best v5 candidate on nat_defs_large_v5.

Usage:
    python -m evolve.run_large_v5 --best-genome /path/to/best/genome.json
                                  --out-dir project/evolve/autonomous_runs/large_v5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from evolve.autonomous_research_loop import (
    REPO_ROOT, summarize_metrics, write_strategy_config,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--best-genome", required=True, type=Path,
                   help="Path to the genome.json (or strategy_config.json) to evaluate")
    p.add_argument("--theorem-set", default="nat_defs_large_v5")
    p.add_argument("--ckpt-dir", default="project/models/gen_v5")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--timeout-seconds", type=int, default=3600)
    args = p.parse_args()

    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    eval_dir = out_root / f"eval-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    raw_genome = json.loads(args.best_genome.read_text())
    # Accept either genome.json (full) or strategy_config.json (subset).
    # If the file has "fallback_tactics" at top-level, treat as genome;
    # otherwise wrap it.
    if "fallback_tactics" not in raw_genome:
        raise ValueError(f"Expected genome.json with fallback_tactics; got {list(raw_genome.keys())[:5]}")
    # Fill in any v5 fields the genome lacks (loaded from baseline)
    from evolve.autonomous_research_loop import baseline_genome
    g = baseline_genome()
    g.update(raw_genome)
    strategy_path = eval_dir / "strategy_config.json"
    write_strategy_config(g, strategy_path)

    cmd = [
        sys.executable, "-u", "eval_rollout_all.py",
        "--theorem-set", args.theorem_set,
        "--policy-type", "hybrid_evolved",
        "--ckpt-dir", args.ckpt_dir,
        "--top-k", str(g["top_k"]),
        "--max-steps", str(g["max_steps"]),
        "--out-dir", str(eval_dir),
        "--strategy-config", str(strategy_path),
    ]
    print(f"  [run_large_v5] running eval ({args.timeout_seconds}s timeout)...", flush=True)
    log_path = eval_dir / "subprocess.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        try:
            r = subprocess.run(
                cmd, cwd=str(REPO_ROOT),
                stdout=logf, stderr=subprocess.STDOUT,
                timeout=args.timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT")
            return
    elapsed = time.time() - started
    print(f"  finished in {elapsed:.0f}s (rc={r.returncode})")
    if r.returncode != 0:
        print(f"  see {log_path}")
        return
    metrics_paths = sorted(eval_dir.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime)
    if not metrics_paths:
        print("  no metrics.json")
        return
    proved, progress, errored, pbo, tb_a, tb_adv, tb_p, *_ = summarize_metrics(metrics_paths[-1], None)
    m = json.loads(metrics_paths[-1].read_text())
    print(f"  proved: {proved}/{m['available']}  progress: {progress}  errored: {errored}")
    print(f"  origins: {pbo}")
    if tb_a:
        print(f"  term_builder: {tb_a}/{tb_adv}/{tb_p}")


if __name__ == "__main__":
    main()
