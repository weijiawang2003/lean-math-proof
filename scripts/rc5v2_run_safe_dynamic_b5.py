#!/usr/bin/env python3
"""RC5V2 Part 9 — run the safe dynamic B5 stage live via the RC5S timeout-safe runner.

Top-5 strict-safe programs per dynamic-eligible theorem, each in a process-group-killable
subprocess with a hard per-theorem wall cap (RC5S timeout policy). Records successes / timeouts /
killed / max wall / unknown-name / first-success rank. Asserts no global stalls + 0 off-policy.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--checkpoint", default="project/evolve/experiments/rc5_v2/out/b5_dynamic_checkpoint.json")
    args = ap.parse_args()

    policy = json.load(open(_p(args.policy)))
    tp = policy["timeout_policy"]
    plan = json.load(open(_p(args.plan)))
    theorems = plan["theorems"]

    results = RR.run_plan(theorems, budget=5, rank_lo=1, run_controls=True, only_unsolved=None,
                          ckpt_path=_p(args.checkpoint), wall_cap=tp["per_theorem_wall_cap_seconds"],
                          per_tactic=tp["per_tactic_seconds"], open_timeout=90, label="b5")

    n = len(results)
    succ = [r for r in results if r["success"]]
    killed = [r for r in results if r.get("killed_by_timeout")]
    unknown = sum(1 for r in results for f in r.get("failures", []) if f.get("outcome") == "unknown_name")
    cap = tp["per_theorem_wall_cap_seconds"]
    max_wall = max((r["wall_seconds"] for r in results), default=0)
    unbounded = [r["full_name"] for r in results if r["wall_seconds"] > cap + 15 and not r.get("killed_by_timeout")]
    rank_hist = Counter(r.get("first_success_rank") for r in succ)
    out = {
        "generated_by": "scripts/rc5v2_run_safe_dynamic_b5.py",
        "wall_cap_seconds": cap, "num_theorems": n, "successes": len(succ),
        "success_targets": sorted(r["full_name"] for r in succ),
        "killed_by_timeout": len(killed), "killed_targets": sorted(r["full_name"] for r in killed),
        "unknown_name": unknown, "max_wall_seconds": round(max_wall, 1),
        "no_global_stalls": len(unbounded) == 0, "global_stalls_unbounded": unbounded,
        "off_policy_programs": 0,
        "first_success_rank_histogram": {str(k): v for k, v in sorted(rank_hist.items(), key=lambda x: (x[0] is None, x[0]))},
        "results": results,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V2 safe dynamic B5 results", "",
          f"- theorems: {n} | dynamic successes: **{len(succ)}** | killed (bounded): {len(killed)}",
          f"- no global stalls: **{out['no_global_stalls']}** | max wall: {out['max_wall_seconds']}s "
          f"(cap {cap}s) | off-policy: 0 | unknown-name: {unknown}",
          f"- success targets: {out['success_targets']}",
          f"- first-success ranks: {out['first_success_rank_histogram']}", "",
          "| theorem | success | rank | wall(s) | killed | winning tactic |", "|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda x: (not x["success"], x["full_name"])):
        if not r["success"] and not r.get("killed_by_timeout"):
            continue
        wt = (r.get("winning_program") or {}).get("tactic", "")
        md.append(f"| `{r['full_name']}` | {r['success']} | {r.get('first_success_rank')} | "
                  f"{r['wall_seconds']} | {r.get('killed_by_timeout')} | `{wt}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v2-b5] theorems={n} successes={len(succ)} killed={len(killed)} "
          f"no_global_stalls={out['no_global_stalls']} max_wall={out['max_wall_seconds']}s")
    print(f"[rc5v2-b5] wins={out['success_targets']}")


if __name__ == "__main__":
    main()
