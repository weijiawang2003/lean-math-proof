#!/usr/bin/env python3
"""RC5S Part 8 — optional safe B10 continuation.

Runs ranks 6–10 ONLY on B5-unsolved theorems and ONLY the B10-reserve programs (safe NON-aesop
families: exact/simpa/simp/rw — per the strict policy's B10 rule), each in a process-group-
killable subprocess. Reports marginal yield + timeout cost; recommends B5-only if B10 adds no
yield at material cost.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc5s_timeout_safe_runner as RR

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--b5", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--checkpoint", default="project/evolve/experiments/rc5_safety/out/b10_safe_checkpoint.json")
    args = ap.parse_args()

    policy = json.load(open(_p(args.policy)))
    tp = policy["timeout_policy"]
    plan = json.load(open(_p(args.plan)))
    b5 = json.load(open(_p(args.b5)))
    b5_by = {r["full_name"]: r for r in b5["results"]}
    unsolved = {fn for fn, r in b5_by.items() if not r.get("success")}

    # only theorems that have a B10-reserve program (safe non-aesop ranks 6-10)
    theorems = [t for t in plan["theorems"]
                if t["full_name"] in unsolved
                and any(p.get("budget_stage") == "B10" for p in t.get("programs_ranked", []))]

    results = RR.run_plan(theorems, budget=10, rank_lo=6, run_controls=False,
                          only_unsolved=unsolved, ckpt_path=_p(args.checkpoint),
                          wall_cap=tp["per_theorem_wall_cap_seconds"], per_tactic=tp["per_tactic_seconds"],
                          open_timeout=90, label="b10")

    new_wins = [r for r in results if r.get("success")]
    killed = [r for r in results if r.get("killed_by_timeout")]
    total_wall = round(sum(r["wall_seconds"] for r in results), 1)
    recommend = "B5_ONLY" if (len(new_wins) == 0 and total_wall > 60) else (
        "B10_ADDS_YIELD" if new_wins else "B5_ONLY")
    out = {
        "generated_by": "scripts/rc5s_run_safe_b10.py",
        "num_b10_eligible_theorems": len(theorems),
        "b5_unsolved": len(unsolved),
        "new_wins_this_stage": len(new_wins),
        "new_win_targets": sorted(r["full_name"] for r in new_wins),
        "killed_by_timeout": len(killed),
        "total_wall_seconds": total_wall,
        "no_global_stalls": all(not (r["wall_seconds"] > tp["per_theorem_wall_cap_seconds"] + 15
                                      and not r.get("killed_by_timeout")) for r in results),
        "marginal_yield_per_wall_minute": round(len(new_wins) / (total_wall / 60), 3) if total_wall else 0.0,
        "recommendation": recommend,
        "results": results,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5S safe B10 results (optional)", "",
          f"- B10-eligible theorems (safe reserve, B5-unsolved): {len(theorems)}",
          f"- new wins this stage: **{len(new_wins)}** {out['new_win_targets']}",
          f"- killed_by_timeout (bounded): {len(killed)} | total wall: {total_wall}s",
          f"- no global stalls: {out['no_global_stalls']}",
          f"- **recommendation: {recommend}**"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5s-b10] eligible={len(theorems)} new_wins={len(new_wins)} killed={len(killed)} "
          f"wall={total_wall}s recommend={recommend}")


if __name__ == "__main__":
    main()
