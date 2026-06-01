#!/usr/bin/env python3
"""TR5 Part 6 — optional B10 / B20 continuation.

Runs only the NEXT rank window (6–10 for B10, 11–20 for B20) on theorems still UNSOLVED
by the previous budget, reusing the Part-5 worker (controls skipped — already run at B5).
Records incremental yield: new wins and marginal successes per additional probe.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tr5_run_ranked_live_search as R

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked-plan", required=True)
    ap.add_argument("--previous-results", required=True)
    ap.add_argument("--budget", type=int, required=True)  # 10 or 20
    ap.add_argument("--target-pool",
                    default="project/evolve/experiments/tr5/cases/tr5_target_pool.jsonl")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--open-timeout", type=int, default=90)
    ap.add_argument("--timeout-per-tactic", type=int, default=12)
    ap.add_argument("--hard-timeout", type=int, default=1200)
    args = ap.parse_args()

    plan = json.load(open(_p(args.ranked_plan)))
    theorems = plan["theorems"]
    theorems.sort(key=lambda t: (0 if t.get("rc2_status") == "CONFIRMED_RC2_FAILURE" else 1,
                                 t["full_name"]))
    for t in theorems:
        if t.get("file_path"):
            R._FILE_CACHE[t["full_name"]] = t["file_path"]
    if os.path.exists(_p(args.target_pool)):
        for l in open(_p(args.target_pool)):
            if l.strip():
                r = json.loads(l)
                R._FILE_CACHE.setdefault(r["full_name"], r.get("file_path"))
        for t in theorems:
            t.setdefault("file_path", R._FILE_CACHE.get(t["full_name"]))

    prev = json.load(open(_p(args.previous_results)))
    prev_by = {r["full_name"]: r for r in prev["results"]}
    prev_budget = prev["budget"]
    unsolved = {fn for fn, r in prev_by.items() if not r.get("success")}
    rank_lo = prev_budget + 1

    ckpt_path = _p(f"project/evolve/experiments/tr5/out/b{args.budget}_live_checkpoint.json")
    new_results = R.run_budget(theorems, args.budget, ckpt_path, args,
                               rank_lo=rank_lo, run_controls=False,
                               only_unsolved=unsolved)
    new_by = {r["full_name"]: r for r in new_results}

    # merged success picture: a theorem is solved if prev solved OR new solved
    merged = []
    new_wins = []
    for t in theorems:
        fn = t["full_name"]
        p = prev_by.get(fn, {})
        n = new_by.get(fn)
        if p.get("success"):
            merged.append({"full_name": fn, "namespace": t.get("namespace"),
                           "solved_at_budget": prev_budget,
                           "first_success_rank": p.get("first_success_rank"),
                           "winning_program": p.get("winning_program")})
        elif n and n.get("success"):
            merged.append({"full_name": fn, "namespace": t.get("namespace"),
                           "solved_at_budget": args.budget,
                           "first_success_rank": n.get("first_success_rank"),
                           "winning_program": n.get("winning_program")})
            new_wins.append(merged[-1])
        else:
            merged.append({"full_name": fn, "namespace": t.get("namespace"),
                           "solved_at_budget": None, "first_success_rank": None,
                           "winning_program": None})

    added_probes = sum(r["programs_attempted"] for r in new_results)
    out = {
        "generated_by": "scripts/tr5_continue_ranked_live_search.py",
        "budget": args.budget, "previous_budget": prev_budget,
        "rank_window": [rank_lo, args.budget],
        "num_unsolved_entering": len(unsolved),
        "num_new_wins": len(new_wins), "new_wins": new_wins,
        "added_probes": added_probes,
        "marginal_success_per_probe": round(len(new_wins) / max(1, added_probes), 4),
        "total_solved_after": sum(1 for m in merged if m["solved_at_budget"] is not None),
        "new_results": new_results, "merged": merged,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = [f"# TR5 B{args.budget} continuation", "",
          f"- ran ranks {rank_lo}-{args.budget} on {len(unsolved)} theorems unsolved by B{prev_budget}",
          f"- **new wins: {len(new_wins)}** | added probes: {added_probes} | "
          f"marginal success/probe: {out['marginal_success_per_probe']}",
          f"- total solved after B{args.budget}: {out['total_solved_after']}", ""]
    if new_wins:
        md += ["## New wins", "", "| theorem | rank | winning tactic |", "|---|---|---|"]
        for w in new_wins:
            wp = w["winning_program"]
            md.append(f"| `{w['full_name']}` | {w['first_success_rank']} | "
                      f"`{wp['tactic'][:50] if wp else ''}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr5-continue] B{args.budget}: new_wins={len(new_wins)} added_probes={added_probes} "
          f"total_solved={out['total_solved_after']}")


if __name__ == "__main__":
    main()
