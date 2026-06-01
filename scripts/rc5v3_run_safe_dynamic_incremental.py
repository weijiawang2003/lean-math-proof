#!/usr/bin/env python3
"""RC5V3 Part 9 — run the safe dynamic stage INCREMENTALLY across budgets B1 / B3 / B5.

Uses the RC5S timeout-safe runner (process-group kill, hard wall cap, per-theorem checkpoint).
  - B1: rank 1 only, on every plan theorem.
  - B3: ranks 2-3, only on theorems still unsolved after B1.
  - B5: ranks 4-5, only on theorems still unsolved after B3.
Stops after the first dynamic success per theorem (escalation skips solved cases). Each stage gets
its own checkpoint for deterministic resume. Records per-budget successes / cumulative / probes /
timeouts / killed / max wall / unknown-name / first-success rank.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc5s_timeout_safe_runner as RR

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _stage_stats(results, cap):
    succ = [r for r in results if r.get("success")]
    killed = [r for r in results if r.get("killed_by_timeout")]
    attempts = sum(r.get("programs_attempted", 0) for r in results)
    unknown = sum(1 for r in results for f in r.get("failures", []) if f.get("outcome") == "unknown_name")
    max_wall = max((r.get("wall_seconds", 0) for r in results), default=0)
    unbounded = [r["full_name"] for r in results
                 if r.get("wall_seconds", 0) > cap + 15 and not r.get("killed_by_timeout")]
    return {"num_attempted_theorems": len(results), "successes": len(succ),
            "success_targets": sorted(r["full_name"] for r in succ),
            "programs_attempted": attempts, "killed_by_timeout": len(killed),
            "unknown_name": unknown, "max_wall_seconds": round(max_wall, 1),
            "no_global_stalls": len(unbounded) == 0, "global_stalls_unbounded": unbounded}


def _write_stage(out_path, label, results, cap, extra):
    stats = _stage_stats(results, cap)
    out = {"generated_by": "scripts/rc5v3_run_safe_dynamic_incremental.py", "budget": label,
           "wall_cap_seconds": cap, **stats, **extra, "results": results}
    json.dump(out, open(_p(out_path), "w"), ensure_ascii=False, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--budget-slices", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out-b1", required=True)
    ap.add_argument("--out-b3", required=True)
    ap.add_argument("--out-b5", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--ckpt-dir", default="project/evolve/experiments/rc5_v3/out")
    args = ap.parse_args()

    policy = json.load(open(_p(args.policy)))
    tp = policy["timeout_policy"]
    cap = tp["per_theorem_wall_cap_seconds"]
    per_tactic = tp["per_tactic_seconds"]
    plan = json.load(open(_p(args.plan)))
    theorems = plan["theorems"]
    all_names = {t["full_name"] for t in theorems}

    # --- B1: rank 1 only, all theorems ---
    b1 = RR.run_plan(theorems, budget=1, rank_lo=1, run_controls=True, only_unsolved=None,
                     ckpt_path=_p(args.ckpt_dir, "b1_dynamic_checkpoint.json"),
                     wall_cap=cap, per_tactic=per_tactic, open_timeout=90, label="b1")
    b1_solved = {r["full_name"] for r in b1 if r.get("success")}
    b1_out = _write_stage(args.out_b1, "B1", b1, cap,
                          {"cumulative_successes": len(b1_solved),
                           "first_success_rank_histogram": dict(Counter(r.get("first_success_rank")
                                                               for r in b1 if r.get("success")))})

    # --- B3: ranks 2-3, only unsolved after B1 ---
    unsolved_after_b1 = all_names - b1_solved
    b3 = RR.run_plan(theorems, budget=3, rank_lo=2, run_controls=True, only_unsolved=unsolved_after_b1,
                     ckpt_path=_p(args.ckpt_dir, "b3_dynamic_checkpoint.json"),
                     wall_cap=cap, per_tactic=per_tactic, open_timeout=90, label="b3")
    b3_solved = {r["full_name"] for r in b3 if r.get("success")}
    cum_b3 = b1_solved | b3_solved
    b3_out = _write_stage(args.out_b3, "B3", b3, cap,
                          {"cumulative_successes": len(cum_b3),
                           "marginal_successes": len(b3_solved),
                           "first_success_rank_histogram": dict(Counter(r.get("first_success_rank")
                                                               for r in b3 if r.get("success")))})

    # --- B5: ranks 4-5, only unsolved after B3 ---
    unsolved_after_b3 = all_names - cum_b3
    b5 = RR.run_plan(theorems, budget=5, rank_lo=4, run_controls=True, only_unsolved=unsolved_after_b3,
                     ckpt_path=_p(args.ckpt_dir, "b5_dynamic_checkpoint.json"),
                     wall_cap=cap, per_tactic=per_tactic, open_timeout=90, label="b5")
    b5_solved = {r["full_name"] for r in b5 if r.get("success")}
    cum_b5 = cum_b3 | b5_solved
    b5_out = _write_stage(args.out_b5, "B5", b5, cap,
                          {"cumulative_successes": len(cum_b5),
                           "marginal_successes": len(b5_solved),
                           "first_success_rank_histogram": dict(Counter(r.get("first_success_rank")
                                                               for r in b5 if r.get("success")))})

    all_results = b1 + b3 + b5
    global_max_wall = max((r.get("wall_seconds", 0) for r in all_results), default=0)
    no_stalls = all(o["no_global_stalls"] for o in (b1_out, b3_out, b5_out))
    md = ["# RC5V3 incremental dynamic run (B1/B3/B5)", "",
          f"- plan theorems: {len(theorems)}",
          f"- **B1** rank1: attempted {b1_out['num_attempted_theorems']}, "
          f"successes {b1_out['successes']}, probes {b1_out['programs_attempted']}, "
          f"max wall {b1_out['max_wall_seconds']}s, killed {b1_out['killed_by_timeout']}",
          f"- **B3** ranks2-3: attempted {b3_out['num_attempted_theorems']}, "
          f"marginal {b3_out['marginal_successes']} (cum {b3_out['cumulative_successes']}), "
          f"probes {b3_out['programs_attempted']}, max wall {b3_out['max_wall_seconds']}s, "
          f"killed {b3_out['killed_by_timeout']}",
          f"- **B5** ranks4-5: attempted {b5_out['num_attempted_theorems']}, "
          f"marginal {b5_out['marginal_successes']} (cum {b5_out['cumulative_successes']}), "
          f"probes {b5_out['programs_attempted']}, max wall {b5_out['max_wall_seconds']}s, "
          f"killed {b5_out['killed_by_timeout']}",
          f"- **cumulative dynamic successes (B5): {len(cum_b5)}**",
          f"- global max wall: {round(global_max_wall,1)}s (cap {cap}s) | no global stalls: {no_stalls}",
          f"- total dynamic probes: {b1_out['programs_attempted']+b3_out['programs_attempted']+b5_out['programs_attempted']}"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v3-incr] B1={b1_out['successes']} B3+={b3_out['marginal_successes']} "
          f"B5+={b5_out['marginal_successes']} cum={len(cum_b5)} "
          f"probes={b1_out['programs_attempted']+b3_out['programs_attempted']+b5_out['programs_attempted']} "
          f"no_stalls={no_stalls} max_wall={round(global_max_wall,1)}s")
    print(f"[rc5v3-incr] cumulative wins={sorted(cum_b5)}")


if __name__ == "__main__":
    main()
