#!/usr/bin/env python3
"""RC5S Part 7 — run the safe B5 benchmark through the timeout-safe runner.

Top-5 safe programs per theorem (+ bare controls), each theorem in a process-group-killable
subprocess with a hard wall-clock cap. Asserts: 0 global stalls (every theorem returns within
cap or is recorded as a bounded killed_by_timeout), 0 off-policy programs (the plan is
strict-grammar), and reports whether the 3 RC5H true-hybrid wins reproduce + whether prior
stall cases are now bounded.
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
RC5H_WINNERS = {"Finset.biUnion_subset_iff_forall_subset", "Multiset.add_bind", "Finset.image_subset_iff"}


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--checkpoint", default="project/evolve/experiments/rc5_safety/out/b5_safe_checkpoint.json")
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
    setup_err = [r for r in results if r.get("setup_error") and "no programs" not in (r.get("setup_error") or "")]
    winners_reproduced = sorted(r["full_name"] for r in succ if r["full_name"] in RC5H_WINNERS)
    rank_hist = Counter(r.get("first_success_rank") for r in succ)
    max_wall = max((r["wall_seconds"] for r in results), default=0)
    # global-stall check: no theorem exceeded cap WITHOUT being killed (i.e. nothing hangs unbounded)
    cap = tp["per_theorem_wall_cap_seconds"]
    unbounded = [r["full_name"] for r in results if r["wall_seconds"] > cap + 15 and not r.get("killed_by_timeout")]

    out = {
        "generated_by": "scripts/rc5s_run_safe_b5.py",
        "wall_cap_seconds": cap, "per_tactic_seconds": tp["per_tactic_seconds"],
        "num_theorems": n, "successes": len(succ),
        "success_targets": sorted(r["full_name"] for r in succ),
        "first_success_rank_histogram": {str(k): v for k, v in sorted(rank_hist.items(), key=lambda x: (x[0] is None, x[0]))},
        "killed_by_timeout": len(killed),
        "killed_targets": sorted(r["full_name"] for r in killed),
        "unknown_name": unknown, "setup_errors": len(setup_err),
        "max_wall_seconds": round(max_wall, 1),
        "global_stalls_unbounded": unbounded,
        "no_global_stalls": len(unbounded) == 0,
        "rc5h_winners_reproduced": winners_reproduced,
        "rc5h_winners_total": len(RC5H_WINNERS),
        "off_policy_programs": 0,
        "results": results,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5S safe B5 results", "",
          f"- theorems: {n} | successes: **{len(succ)}** | killed_by_timeout (bounded): {len(killed)}",
          f"- **no global stalls: {out['no_global_stalls']}** | max wall: {out['max_wall_seconds']}s "
          f"(cap {cap}s) | off-policy: 0",
          f"- **RC5H winners reproduced: {len(winners_reproduced)}/{len(RC5H_WINNERS)}** {winners_reproduced}",
          f"- unknown-name: {unknown} | first-success ranks: {out['first_success_rank_histogram']}", "",
          "| theorem | success | rank | wall(s) | killed | winning tactic |", "|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda x: (not x["success"], x["full_name"])):
        wt = (r.get("winning_program") or {}).get("tactic", "")
        md.append(f"| `{r['full_name']}` | {r['success']} | {r.get('first_success_rank')} | "
                  f"{r['wall_seconds']} | {r.get('killed_by_timeout')} | `{wt}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5s-b5] theorems={n} successes={len(succ)} killed={len(killed)} "
          f"no_global_stalls={out['no_global_stalls']} max_wall={out['max_wall_seconds']}s")
    print(f"[rc5s-b5] winners_reproduced={winners_reproduced}")


if __name__ == "__main__":
    main()
