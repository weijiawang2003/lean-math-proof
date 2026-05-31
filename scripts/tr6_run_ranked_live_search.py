#!/usr/bin/env python3
"""TR6 Part 8 — live B5/B10/B20 ranker-guided search on fresh RC2 failures.

One script, two modes:
  * initial (no --previous-results): run ranks 1..budget per theorem WITH the 4 bare
    controls, stop after first success (this is B5).
  * continuation (--previous-results given): run ranks (prev_budget+1)..budget only on
    theorems still UNSOLVED, controls skipped (B10 over B5, B20 over B10).

Serialized one-Dojo-per-theorem workers (reused from tr5_run_ranked_live_search) under an
OS hard timeout; per-theorem checkpoint + resume; deterministic ordering. No win is final
here — TR6 attribution (Part 9) re-judges against literal RC2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tr5_run_ranked_live_search as R   # reuse worker + run_budget

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    # worker passthrough (delegates to tr5 worker)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--programs-json")
    ap.add_argument("--run-controls", default="true")
    ap.add_argument("--ranked-plan")
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

    plan = json.load(open(_p(args.ranked_plan)))
    theorems = plan["theorems"]
    # normalize: tr5 worker reads used_lemmas; tr6 programs carry `lemmas`
    for t in theorems:
        t.setdefault("file_path", t.get("file_path"))
        for p in t.get("programs_ranked", []):
            p.setdefault("used_lemmas", p.get("lemmas", []))
    theorems.sort(key=lambda t: (0 if t.get("rc2_status") == "CONFIRMED_RC2_FAILURE" else 1,
                                 t["full_name"]))
    for t in theorems:
        if t.get("file_path"):
            R._FILE_CACHE[t["full_name"]] = t["file_path"]

    # make tr6 worker the subprocess target (so --worker routes here)
    R.__dict__["_WORKER_SCRIPT"] = os.path.abspath(__file__)
    # monkeypatch the worker script path used by run_budget's subprocess cmd
    import tr5_run_ranked_live_search as _M
    _orig_abspath = os.path.abspath
    # run_budget builds cmd with os.path.abspath(_M.__file__); override via env trick:
    # simplest — temporarily set __file__ of the tr5 module to this script.
    _saved_file = _M.__file__
    _M.__file__ = os.path.abspath(__file__)

    if not args.previous_results:
        rank_lo, run_controls, only = 1, True, None
        ckpt = _p(f"project/evolve/experiments/tr6/out/b{args.budget}_live_checkpoint.json")
        prev_budget = 0
    else:
        prev = json.load(open(_p(args.previous_results)))
        prev_by = {r["full_name"]: r for r in prev["results"]}
        only = {fn for fn, r in prev_by.items() if not r.get("success")}
        prev_budget = prev.get("budget", 5)
        rank_lo, run_controls = prev_budget + 1, False
        ckpt = _p(f"project/evolve/experiments/tr6/out/b{args.budget}_live_checkpoint.json")

    results = R.run_budget(theorems, args.budget, ckpt, args,
                           rank_lo=rank_lo, run_controls=run_controls, only_unsolved=only)
    _M.__file__ = _saved_file

    # For continuation, carry forward prev successes into the merged success picture
    if args.previous_results:
        prev = json.load(open(_p(args.previous_results)))
        prev_by = {r["full_name"]: r for r in prev["results"]}
        merged = []
        for t in theorems:
            fn = t["full_name"]
            pr = prev_by.get(fn, {})
            cur = next((r for r in results if r["full_name"] == fn), None)
            if pr.get("success"):
                merged.append(pr)
            elif cur is not None:
                merged.append(cur)
            elif pr:
                merged.append(pr)
        results_out = merged
        new_wins = [r for r in results if r.get("success")]
    else:
        results_out = results
        new_wins = [r for r in results if r.get("success")]

    n_live = sum(1 for r in results_out if r.get("live"))
    n_win = sum(1 for r in results_out if r.get("success"))
    rank_hist = Counter(r.get("first_success_rank") for r in results_out if r.get("success"))
    out = {
        "generated_by": "scripts/tr6_run_ranked_live_search.py",
        "budget": args.budget, "previous_budget": prev_budget,
        "rank_window_this_stage": [rank_lo, args.budget],
        "ranked_plan": args.ranked_plan,
        "num_theorems": len(results_out), "num_live": n_live, "num_success": n_win,
        "num_new_wins_this_stage": len(new_wins),
        "new_wins_this_stage": [r["full_name"] for r in new_wins],
        "first_success_rank_histogram": {str(k): v for k, v in sorted(
            rank_hist.items(), key=lambda x: (x[0] is None, x[0]))},
        "results": results_out,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = [f"# TR6 B{args.budget} live results", "",
          f"- theorems: {len(results_out)} | live: {n_live} | total successes: **{n_win}** | "
          f"new this stage: {len(new_wins)}",
          f"- first-success rank histogram: {dict(rank_hist)}", "",
          "| theorem | ns | success | first_rank | winning tactic |", "|---|---|---|---|---|"]
    for r in sorted(results_out, key=lambda x: (not x.get("success"), x["full_name"])):
        wp = r.get("winning_program")
        wt = wp["tactic"][:48] if wp else ""
        md.append(f"| `{r['full_name']}` | {r.get('namespace')} | {r.get('success')} | "
                  f"{r.get('first_success_rank')} | `{wt}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr6-live] B{args.budget}: live={n_live} total_success={n_win} "
          f"new_this_stage={len(new_wins)} rank_hist={dict(rank_hist)}")


if __name__ == "__main__":
    main()
