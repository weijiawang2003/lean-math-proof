#!/usr/bin/env python3
"""RC4C Part 5 — literal RC2 baseline over the validation sets.

Reuse-first: TR6 / TR3 / TR5 / SF4 / TR2 ran literal RC2 at the IDENTICAL config
(rc2_release wrapper, ns24, hybrid_evolved, top-k 8, max-steps 8), so those
classifications are reused verbatim; the rest run live via eval_rollout_all in chunked
worker subprocesses under an OS hard timeout. Records solved/failed/flake + trace paths
per theorem and emits the exact commands used.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
REUSE_SOURCES = [
    "project/evolve/experiments/tr6/out/tr6_rc2_confirmation.json",
    "project/evolve/experiments/tr3/out/tr3_rc2_confirmation.json",
    "project/evolve/experiments/tr5/out/tr5_rc2_confirmation.json",
    "project/evolve/experiments/sf4/out/rc2_failure_confirmation.json",
    "project/evolve/experiments/tr2/out/tr2_rc2_confirmation.json",
]
_MAP = {"CONFIRMED_RC2_FAILURE": "failed", "NOW_SOLVED_BY_RC2": "solved",
        "RC2_SOLVED": "solved", "OPEN_FLAKE": "open_flake",
        "PATH_ERROR": "path_error", "TRACE_INSUFFICIENT": "trace_insufficient"}


def _p(*a):
    return os.path.join(_REPO, *a)


def worker(args):
    cases = json.loads(args.cases_json)
    out = {}
    try:
        sys.path.insert(0, _REPO)
        from core_types import TheoremConfig
        import eval_rollout_all as E
        fields = {f.name for f in dataclasses.fields(TheoremConfig)}
        tcs, names = [], {}
        for c in cases:
            if not c.get("file_path"):
                out[c["full_name"]] = {"status": "path_error"}
                continue
            tcs.append(TheoremConfig(**{k: v for k, v in
                                        {"file_path": c["file_path"], "full_name": c["full_name"]}.items()
                                        if k in fields}))
            names[c["full_name"]] = c
        if tcs:
            nm = "rc4c_literal_rc2"
            _get, _list = E.get_theorems, E.list_theorem_sets
            E.list_theorem_sets = lambda: (list(_list()) + [nm]) if nm not in _list() else list(_list())
            E.get_theorems = lambda n: tcs if n == nm else _get(n)
            os.makedirs(args.out_dir, exist_ok=True)
            sys.argv = ["eval_rollout_all.py", "--theorem-set", nm,
                        "--policy-type", args.policy_type, "--route-config", args.route_config,
                        "--strategy-config", args.rc2_wrapper, "--top-k", str(args.top_k),
                        "--max-steps", str(args.max_steps), "--out-dir", args.out_dir]
            try:
                E.main()
            except SystemExit:
                pass
            mfiles = sorted(glob.glob(os.path.join(args.out_dir, "eval-*", "metrics.json")),
                            key=os.path.getmtime)
            if mfiles:
                m = json.load(open(mfiles[-1]))
                trace = os.path.join(os.path.dirname(mfiles[-1]), "traces.jsonl")
                seen = {r["full_name"]: r for r in m.get("per_theorem", [])}
                for fn in names:
                    r = seen.get(fn)
                    if r is None:
                        out[fn] = {"status": "path_error"}
                    elif not r.get("available"):
                        out[fn] = {"status": "open_flake"}
                    else:
                        out[fn] = {"status": "solved" if r.get("finished") else "failed",
                                   "num_steps": r.get("num_steps"),
                                   "winning_tactic": r.get("winning_tactic"),
                                   "trace_path": trace}
            else:
                for fn in names:
                    out[fn] = {"status": "trace_insufficient"}
    except Exception as e:
        for c in cases:
            out.setdefault(c["full_name"], {"status": "trace_insufficient", "error": str(e)[:120]})
    json.dump(out, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def _reuse_map():
    m = {}
    for path in REUSE_SOURCES:
        if not os.path.exists(_p(path)):
            continue
        for r in json.load(open(_p(path))).get("results", []):
            fn = r["full_name"]
            if fn in m:
                continue
            if r.get("rc2_finished") is True:
                st = "solved"
            else:
                st = _MAP.get(r.get("classification"))
            if st:
                m[fn] = {"status": st, "num_steps": r.get("num_steps"),
                         "winning_tactic": r.get("winning_tactic"),
                         "trace_path": r.get("trace_path"),
                         "provenance": "reused:" + os.path.basename(path).split(".")[0]}
    return m


def driver(args):
    manifest = json.load(open(_p(args.manifest)))
    theorems, seen, set_membership = [], set(), {}
    for setname, rel in manifest["set_files"].items():
        for e in json.load(open(_p(rel))):
            fn = e["full_name"]
            set_membership.setdefault(fn, []).append(setname)
            if fn not in seen:
                seen.add(fn)
                theorems.append({"full_name": fn, "file_path": e.get("file_path"),
                                 "namespace": e.get("namespace")})
    reuse = _reuse_map()
    ckpt_path = _p(args.checkpoint)
    ckpt = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}

    results, live = {}, []
    for t in theorems:
        fn = t["full_name"]
        if fn in ckpt:
            results[fn] = ckpt[fn]
        elif fn in reuse:
            results[fn] = reuse[fn]
        else:
            live.append(t)
    n_ckpt = sum(1 for t in theorems if t["full_name"] in ckpt)
    print(f"[rc4c-rc2] theorems={len(theorems)} reused={len(theorems)-len(live)-n_ckpt} "
          f"ckpt={n_ckpt} live={len(live)}", flush=True)

    os.makedirs(_p(args.out_dir), exist_ok=True)
    chunks = [live[i:i + args.chunk_size] for i in range(0, len(live), args.chunk_size)]
    for ci, chunk in enumerate(chunks):
        wout = _p(args.out_dir, f"chunk_{ci}.json")
        cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
               sys.executable, os.path.abspath(__file__), "--worker", "--worker-out", wout,
               "--cases-json", json.dumps(chunk), "--out-dir", _p(args.out_dir, "rc2_runs"),
               "--rc2-wrapper", args.rc2_wrapper, "--route-config", args.route_config,
               "--policy-type", args.policy_type, "--top-k", str(args.top_k),
               "--max-steps", str(args.max_steps)]
        print(f"[rc4c-rc2] chunk {ci+1}/{len(chunks)} ({len(chunk)}) ...", flush=True)
        subprocess.run(cmd, capture_output=True, text=True)
        try:
            wres = json.load(open(wout))
        except Exception:
            wres = {c["full_name"]: {"status": "trace_insufficient"} for c in chunk}
        for fn, rec in wres.items():
            rec["provenance"] = "rc4c_live"
            results[fn] = rec
            ckpt[fn] = rec
        json.dump(ckpt, open(ckpt_path, "w"), ensure_ascii=False, indent=2)

    from collections import Counter
    recs = []
    for t in theorems:
        fn = t["full_name"]
        r = results.get(fn, {"status": "trace_insufficient"})
        recs.append({"full_name": fn, "file_path": t["file_path"], "namespace": t["namespace"],
                     "sets": set_membership[fn], "rc2_status": r["status"],
                     "rc2_finished": r["status"] == "solved",
                     "num_steps": r.get("num_steps"), "winning_tactic": r.get("winning_tactic"),
                     "trace_path": r.get("trace_path"), "provenance": r.get("provenance")})
    hist = Counter(r["rc2_status"] for r in recs)
    floors = {}
    for setname in manifest["set_files"]:
        sub = [r for r in recs if setname in r["sets"]]
        floors[setname] = {"n": len(sub),
                           "solved": sum(1 for r in sub if r["rc2_status"] == "solved"),
                           "failed": sum(1 for r in sub if r["rc2_status"] == "failed")}

    out = {"generated_by": "scripts/rc4c_run_literal_rc2.py",
           "rc2_wrapper": args.rc2_wrapper, "route_config": args.route_config,
           "policy_type": args.policy_type, "top_k": args.top_k, "max_steps": args.max_steps,
           "num_theorems": len(recs), "status_histogram": dict(hist),
           "per_set": floors, "results": recs}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC4C literal RC2 baseline", "",
          f"- theorems: {len(recs)} | status: {dict(hist)}", "", "## Per-set", "",
          "| set | n | solved | failed |", "|---|---|---|---|"]
    for s, f in floors.items():
        md.append(f"| {s} | {f['n']} | {f['solved']} | {f['failed']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")

    sh = ["#!/bin/sh",
          "# Literal RC2 baseline command (per chunk; identical config to TR3/TR5/TR6):",
          f"# strategy-config={args.rc2_wrapper} route-config={args.route_config}",
          f"# policy-type={args.policy_type} top-k={args.top_k} max-steps={args.max_steps}",
          "# python3 eval_rollout_all.py --theorem-set <registered> --policy-type "
          f"{args.policy_type} \\",
          f"#   --route-config {args.route_config} --strategy-config {args.rc2_wrapper} \\",
          f"#   --top-k {args.top_k} --max-steps {args.max_steps} --out-dir <out>"]
    open(_p(os.path.dirname(args.out_json), "literal_rc2_commands.sh"), "w").write("\n".join(sh) + "\n")

    print(f"[rc4c-rc2] done: {dict(hist)}; per-set={floors}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--cases-json")
    ap.add_argument("--manifest")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--out-dir", default="project/evolve/experiments/rc4_candidates/d2_simp_aesop/out/rc2_baseline")
    ap.add_argument("--checkpoint", default="project/evolve/experiments/rc4_candidates/d2_simp_aesop/out/rc2_baseline_checkpoint.json")
    ap.add_argument("--rc2-wrapper", default="project/evolve/experiments/rc2_release/rc2_production_wrapper.json")
    ap.add_argument("--route-config", default="project/evolve/routing/ns24_router.json")
    ap.add_argument("--policy-type", default="hybrid_evolved")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=12)
    ap.add_argument("--hard-timeout", type=int, default=1800)
    args = ap.parse_args()
    if args.worker:
        sys.exit(worker(args))
    driver(args)


if __name__ == "__main__":
    main()
