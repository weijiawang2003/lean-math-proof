#!/usr/bin/env python3
"""SF4 Part 3 — live RC2 failure confirmation.

Runs LITERAL RC2 (production harness) on a selected subset of the failure pool and
classifies each theorem:

  CONFIRMED_RC2_FAILURE   available, finished==false
  NOW_SOLVED_BY_RC2       finished==true (drop from the failure-first pipeline)
  OPEN_FLAKE              not available due to a Dojo open/skip issue
  PATH_ERROR              file_path could not be resolved / theorem not found
  TRACE_INSUFFICIENT      ran but produced no usable record

Selection: confirmed-failure rows first (re-confirm the authoritative 13), then
frontier-unconfirmed rows, capped by --max-cases. Uses the SF1 in-process
registration trick; no file on disk is modified; protected configs untouched.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def _load_pool(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _select(rows, max_cases):
    confirmed = [r for r in rows if "rc2_confirmed_failure" in r.get("tags", [])]
    other = [r for r in rows if "rc2_confirmed_failure" not in r.get("tags", [])]
    sel = (confirmed + other)[:max_cases]
    return sel


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--rc2-wrapper", required=True)
    p.add_argument("--route-config", required=True)
    p.add_argument("--policy-type", default="hybrid_evolved")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-cases", type=int, default=50)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-dir", default="project/evolve/experiments/sf4/out/rc2_confirm_runs")
    args = p.parse_args(argv)

    rows = _load_pool(args.pool)
    sel = _select(rows, args.max_cases)

    from core_types import TheoremConfig
    fields = {f.name for f in dataclasses.fields(TheoremConfig)}
    tcs, name_to_row = [], {}
    for r in sel:
        if not r.get("file_path"):
            continue
        kw = {k: v for k, v in r.items() if k in fields and v is not None}
        kw.setdefault("file_path", r["file_path"])
        kw.setdefault("full_name", r["full_name"])
        tcs.append(TheoremConfig(**{k: v for k, v in kw.items() if k in fields}))
        name_to_row[r["full_name"]] = r

    import eval_rollout_all as E
    os.makedirs(args.out_dir, exist_ok=True)
    name = "sf4_confirm"
    _get, _list = E.get_theorems, E.list_theorem_sets
    E.list_theorem_sets = lambda: (list(_list()) + [name]) if name not in _list() else list(_list())
    E.get_theorems = lambda n: tcs if n == name else _get(n)
    sys.argv = ["eval_rollout_all.py", "--theorem-set", name,
                "--policy-type", args.policy_type, "--route-config", args.route_config,
                "--strategy-config", args.rc2_wrapper, "--top-k", str(args.top_k),
                "--max-steps", str(args.max_steps), "--out-dir", args.out_dir]
    print(f"[sf4-confirm] running literal RC2 on {len(tcs)} theorems ...", flush=True)
    try:
        E.main()
    except SystemExit:
        pass

    mfiles = sorted(glob.glob(os.path.join(args.out_dir, "eval-*", "metrics.json")),
                    key=os.path.getmtime)
    if not mfiles:
        print("[sf4-confirm] ERROR: no metrics", file=sys.stderr)
        return 4
    m = json.load(open(mfiles[-1]))
    run_dir = os.path.dirname(mfiles[-1])
    trace_path = os.path.join(run_dir, "traces.jsonl")
    seen = {r["full_name"]: r for r in m.get("per_theorem", [])}

    results = []
    for fn, row in name_to_row.items():
        r = seen.get(fn)
        if r is None:
            cls = "PATH_ERROR"
            rec = {"full_name": fn, "classification": cls, "available": False,
                   "rc2_finished": None, "note": "theorem not present in run output"}
        else:
            avail = r.get("available")
            fin = bool(r.get("finished"))
            if not avail:
                cls = "PATH_ERROR" if (r.get("skip_reason") and "resolve" in str(r.get("skip_reason")).lower()) else "OPEN_FLAKE"
            elif fin:
                cls = "NOW_SOLVED_BY_RC2"
            else:
                cls = "CONFIRMED_RC2_FAILURE"
            rec = {"full_name": fn, "file_path": r.get("file_path"), "namespace": row.get("namespace"),
                   "classification": cls, "available": avail, "rc2_finished": fin,
                   "num_steps": r.get("num_steps"), "winning_tactic": r.get("winning_tactic"),
                   "tactics_used": r.get("tactics_used"), "error_message": r.get("error_message"),
                   "skip_reason": r.get("skip_reason"), "tags": row.get("tags"),
                   "trace_path": trace_path}
        results.append(rec)

    hist = {}
    for r in results:
        hist[r["classification"]] = hist.get(r["classification"], 0) + 1
    confirmed = [r["full_name"] for r in results if r["classification"] == "CONFIRMED_RC2_FAILURE"]

    out = {
        "rc2_wrapper": args.rc2_wrapper, "route_config": args.route_config,
        "policy_type": args.policy_type, "top_k": args.top_k, "max_steps": args.max_steps,
        "num_run": len(results), "classification_histogram": hist,
        "confirmed_rc2_failures": sorted(confirmed), "num_confirmed": len(confirmed),
        "metrics_path": mfiles[-1], "trace_path": trace_path,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# SF4 RC2 failure confirmation", "",
         f"- theorems run: **{len(results)}**",
         f"- **CONFIRMED_RC2_FAILURE: {len(confirmed)}**", "",
         "## Histogram", ""]
    for k, v in sorted(hist.items()):
        L.append(f"- `{k}`: {v}")
    L += ["", "## Per theorem", "", "| theorem | class | finished | steps |", "|---|---|---|---|"]
    for r in results:
        L.append(f"| `{r['full_name']}` | {r['classification']} | {r.get('rc2_finished')} | {r.get('num_steps')} |")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[sf4-confirm] confirmed={len(confirmed)} hist={hist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
