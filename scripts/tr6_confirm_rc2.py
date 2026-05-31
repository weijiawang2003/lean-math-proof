#!/usr/bin/env python3
"""TR6 Part 5 — live literal-RC2 confirmation on the fresh batch (no reuse).

All TR6 cases are fresh (zero exclusion overlap), so every theorem is run live via
eval_rollout_all under the exact RC2 config (rc2_release wrapper, ns24 router,
hybrid_evolved, top-k 8, max-steps 8), in chunked worker subprocesses under an OS hard
timeout with a per-chunk checkpoint. This run is ALSO the availability filter: a theorem
that does not resolve becomes PATH_ERROR / TRACE_INSUFFICIENT and drops out.

Classifications: CONFIRMED_RC2_FAILURE / RC2_SOLVED / OPEN_FLAKE / PATH_ERROR /
TRACE_INSUFFICIENT. Only CONFIRMED_RC2_FAILURE is TRUE_DELTA-eligible.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import subprocess
import sys
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")


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
                out[c["full_name"]] = {"classification": "PATH_ERROR", "rc2_finished": None}
                continue
            kw = {"file_path": c["file_path"], "full_name": c["full_name"]}
            tcs.append(TheoremConfig(**{k: v for k, v in kw.items() if k in fields}))
            names[c["full_name"]] = c
        if tcs:
            name = "tr6_confirm_chunk"
            _get, _list = E.get_theorems, E.list_theorem_sets
            E.list_theorem_sets = lambda: (list(_list()) + [name]) if name not in _list() else list(_list())
            E.get_theorems = lambda n: tcs if n == name else _get(n)
            os.makedirs(args.out_dir, exist_ok=True)
            sys.argv = ["eval_rollout_all.py", "--theorem-set", name,
                        "--policy-type", args.policy_type, "--route-config", args.route_config,
                        "--strategy-config", args.rc2_wrapper, "--top-k", str(args.top_k),
                        "--max-steps", str(args.max_steps), "--out-dir", args.out_dir]
            try:
                E.main()
            except SystemExit:
                pass
            mfiles = sorted(glob.glob(os.path.join(args.out_dir, "eval-*", "metrics.json")),
                            key=os.path.getmtime)
            if not mfiles:
                for fn in names:
                    out[fn] = {"classification": "TRACE_INSUFFICIENT", "rc2_finished": None}
            else:
                m = json.load(open(mfiles[-1]))
                trace = os.path.join(os.path.dirname(mfiles[-1]), "traces.jsonl")
                seen = {r["full_name"]: r for r in m.get("per_theorem", [])}
                for fn in names:
                    r = seen.get(fn)
                    if r is None:
                        out[fn] = {"classification": "PATH_ERROR", "rc2_finished": None}
                    elif not r.get("available"):
                        out[fn] = {"classification": "OPEN_FLAKE", "rc2_finished": None}
                    elif r.get("finished"):
                        out[fn] = {"classification": "RC2_SOLVED", "rc2_finished": True,
                                   "num_steps": r.get("num_steps"),
                                   "winning_tactic": r.get("winning_tactic"), "trace_path": trace}
                    else:
                        out[fn] = {"classification": "CONFIRMED_RC2_FAILURE", "rc2_finished": False,
                                   "num_steps": r.get("num_steps"),
                                   "tactics_used": r.get("tactics_used"), "trace_path": trace}
    except Exception as e:
        import traceback
        for c in cases:
            out.setdefault(c["full_name"], {"classification": "TRACE_INSUFFICIENT",
                                            "error": f"{type(e).__name__}: {e}",
                                            "trace": traceback.format_exc()[-300:]})
    json.dump(out, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def driver(args):
    batch = json.load(open(_p(args.batch)))
    pool = batch["theorems"]
    if args.max_cases:
        pool = pool[: args.max_cases]
    ckpt_path = _p(args.checkpoint)
    ckpt = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}

    results = {fn: ckpt[fn] for fn in (c["full_name"] for c in pool) if fn in ckpt}
    live_cases = [c for c in pool if c["full_name"] not in ckpt]
    print(f"[tr6-confirm] batch={len(pool)} ckpt={len(results)} live_to_run={len(live_cases)}", flush=True)

    os.makedirs(_p(args.out_dir), exist_ok=True)
    chunks = [live_cases[i:i + args.chunk_size] for i in range(0, len(live_cases), args.chunk_size)]
    for ci, chunk in enumerate(chunks):
        wout = _p(args.out_dir, f"chunk_{ci}.json")
        cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
               sys.executable, os.path.abspath(__file__), "--worker",
               "--worker-out", wout, "--cases-json", json.dumps(chunk),
               "--out-dir", _p(args.out_dir, "rc2_runs"),
               "--rc2-wrapper", args.rc2_wrapper, "--route-config", args.route_config,
               "--policy-type", args.policy_type, "--top-k", str(args.top_k),
               "--max-steps", str(args.max_steps)]
        print(f"[tr6-confirm] chunk {ci+1}/{len(chunks)} ({len(chunk)} cases) ...", flush=True)
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        try:
            wres = json.load(open(wout))
        except Exception:
            wres = {c["full_name"]: {"classification": "TRACE_INSUFFICIENT",
                                     "error": f"worker no output rc={rc}"} for c in chunk}
        cmap = {c["full_name"]: c for c in chunk}
        for fn, rec in wres.items():
            rec["provenance"] = "tr6_live"
            rec["case"] = cmap.get(fn)
            results[fn] = rec
            ckpt[fn] = rec
        json.dump(ckpt, open(ckpt_path, "w"), ensure_ascii=False, indent=2)

    recs = []
    for c in pool:
        fn = c["full_name"]
        r = results.get(fn, {"classification": "TRACE_INSUFFICIENT"})
        recs.append({
            "full_name": fn, "file_path": c.get("file_path"), "namespace": c.get("namespace"),
            "statement_text": c.get("statement_text"),
            "candidate_family_tags": c.get("candidate_family_tags", []),
            "features": c.get("features", {}),
            "classification": r.get("classification"), "rc2_finished": r.get("rc2_finished"),
            "num_steps": r.get("num_steps"), "winning_tactic": r.get("winning_tactic"),
            "tactics_used": r.get("tactics_used"), "trace_path": r.get("trace_path"),
            "provenance": r.get("provenance", "tr6_live"),
        })
    hist = Counter(r["classification"] for r in recs)
    confirmed = [r["full_name"] for r in recs if r["classification"] == "CONFIRMED_RC2_FAILURE"]
    ns_fail = Counter(r["namespace"] for r in recs if r["classification"] == "CONFIRMED_RC2_FAILURE")

    out = {
        "generated_by": "scripts/tr6_confirm_rc2.py",
        "rc2_wrapper": args.rc2_wrapper, "route_config": args.route_config,
        "policy_type": args.policy_type, "top_k": args.top_k, "max_steps": args.max_steps,
        "num_cases": len(recs), "classification_histogram": dict(hist),
        "confirmed_rc2_failures": sorted(confirmed), "num_confirmed": len(confirmed),
        "confirmed_by_namespace": dict(ns_fail),
        "target_50_met": len(confirmed) >= 50,
        "results": recs,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR6 RC2 confirmation (live, fresh)", "",
          f"- cases: {len(recs)} | **confirmed RC2 failures: {len(confirmed)}** "
          f"(≥50 target: {out['target_50_met']})",
          f"- classifications: {dict(hist)}",
          f"- confirmed failures by namespace: {dict(ns_fail)}", "",
          "| theorem | ns | class | finished |", "|---|---|---|---|"]
    for r in recs:
        md.append(f"| `{r['full_name']}` | {r['namespace']} | {r['classification']} | "
                  f"{r['rc2_finished']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr6-confirm] done: {dict(hist)} | confirmed={len(confirmed)} | by_ns={dict(ns_fail)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--cases-json")
    ap.add_argument("--batch")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--out-dir", default="project/evolve/experiments/tr6/out/rc2_confirm")
    ap.add_argument("--checkpoint",
                    default="project/evolve/experiments/tr6/out/rc2_confirm_checkpoint.json")
    ap.add_argument("--rc2-wrapper",
                    default="project/evolve/experiments/rc2_release/rc2_production_wrapper.json")
    ap.add_argument("--route-config", default="project/evolve/routing/ns24_router.json")
    ap.add_argument("--policy-type", default="hybrid_evolved")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--hard-timeout", type=int, default=2400)
    ap.add_argument("--max-cases", type=int, default=0)
    args = ap.parse_args()
    if args.worker:
        sys.exit(worker(args))
    driver(args)


if __name__ == "__main__":
    main()
