#!/usr/bin/env python3
"""RC5H Part 8 — run the dynamic stage live (B5/B10/B20).

Reuses the validated TR5/TR6 `run_budget` driver (one-Dojo-per-theorem workers, per-theorem
checkpoint, deterministic ordering, stop-after-first-success). B5 = ranks 1..5 + bare controls;
B10 = ranks 6..10 on B5-unsolved (controls skipped); B20 = ranks 11..20 on B10-unsolved. The
dynamic stage only ever sees the static failures in the program plan (built from static
failures), so it is additive over RC4 by construction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tr5_run_ranked_live_search as R  # reuse worker + run_budget

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = "project/evolve/experiments/rc5_hybrid/out"


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--programs-json")
    ap.add_argument("--run-controls", default="true")
    ap.add_argument("--program-plan")
    ap.add_argument("--previous-results")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--open-timeout", type=int, default=90)
    ap.add_argument("--timeout-per-tactic", type=int, default=12)
    ap.add_argument("--hard-timeout", type=int, default=1200)
    args = ap.parse_args()
    if args.worker:
        sys.exit(R.worker(args))

    plan = json.load(open(_p(args.program_plan)))
    theorems = [t for t in plan["theorems"] if t.get("rc5h_dynamic_eligible", True)]
    for t in theorems:
        for p in t.get("programs_ranked", []):
            p.setdefault("used_lemmas", p.get("lemmas", []))
        if t.get("file_path"):
            R._FILE_CACHE[t["full_name"]] = t["file_path"]
    theorems.sort(key=lambda t: t["full_name"])  # deterministic

    # route the worker subprocess to THIS script
    R.__dict__  # noqa
    _saved = R.__file__
    R.__file__ = os.path.abspath(__file__)

    ckpt = _p(CKPT_DIR, f"b{args.budget}_dynamic_checkpoint.json")
    if not args.previous_results:
        rank_lo, run_controls, only, prev_budget = 1, args.run_controls == "true", None, 0
    else:
        prev = json.load(open(_p(args.previous_results)))
        prev_by = {r["full_name"]: r for r in prev["results"]}
        only = {fn for fn, r in prev_by.items() if not r.get("success")}
        prev_budget = prev.get("budget", 5)
        rank_lo, run_controls = prev_budget + 1, False

    results = R.run_budget(theorems, args.budget, ckpt, args,
                           rank_lo=rank_lo, run_controls=run_controls, only_unsolved=only)
    R.__file__ = _saved

    if args.previous_results:
        prev = json.load(open(_p(args.previous_results)))
        prev_by = {r["full_name"]: r for r in prev["results"]}
        merged = []
        for t in theorems:
            fn = t["full_name"]
            pr = prev_by.get(fn, {})
            cur = next((r for r in results if r["full_name"] == fn), None)
            merged.append(pr if pr.get("success") else (cur if cur is not None else pr))
        results_out = [m for m in merged if m]
        new_wins = [r for r in results if r.get("success")]
    else:
        results_out = results
        new_wins = [r for r in results if r.get("success")]

    n_live = sum(1 for r in results_out if r.get("live"))
    n_win = sum(1 for r in results_out if r.get("success"))
    rank_hist = Counter(r.get("first_success_rank") for r in results_out if r.get("success"))
    out = {"generated_by": "scripts/rc5h_run_dynamic_stage.py", "budget": args.budget,
           "previous_budget": prev_budget, "rank_window_this_stage": [rank_lo, args.budget],
           "program_plan": args.program_plan, "num_theorems": len(results_out), "num_live": n_live,
           "num_success": n_win, "num_new_wins_this_stage": len(new_wins),
           "new_wins_this_stage": [r["full_name"] for r in new_wins],
           "first_success_rank_histogram": {str(k): v for k, v in sorted(
               rank_hist.items(), key=lambda x: (x[0] is None, x[0]))},
           "results": results_out}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = [f"# RC5H B{args.budget} dynamic results", "",
          f"- theorems: {len(results_out)} | live: {n_live} | total successes: **{n_win}** | "
          f"new this stage: {len(new_wins)}",
          f"- first-success-rank histogram: {out['first_success_rank_histogram']}", "",
          "| theorem | ns | success | rank | winning tactic |", "|---|---|---|---|---|"]
    for r in sorted(results_out, key=lambda x: (not x.get("success"), x["full_name"])):
        wt = (r.get("winning_program") or {}).get("tactic", "")
        md.append(f"| `{r['full_name']}` | {r.get('namespace')} | {r.get('success')} | "
                  f"{r.get('first_success_rank')} | `{wt}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5h-dynamic] B{args.budget}: theorems={len(results_out)} live={n_live} "
          f"successes={n_win} new_this_stage={len(new_wins)}")
    print(f"[rc5h-dynamic] new_wins={out['new_wins_this_stage']}")


if __name__ == "__main__":
    main()
