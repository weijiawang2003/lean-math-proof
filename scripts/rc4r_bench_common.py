#!/usr/bin/env python3
"""RC4R shared benchmark runner core (imported by the RC2 / RC4 benchmark runners).

Runs a given wrapper over the benchmark manifest through the real eval_rollout_all search at
the RC2-release config (hybrid_evolved, top-k 8, max-steps 8), reuse-first from a supplied list
of prior exact-config result files, the rest live in chunked worker subprocesses under an OS
hard timeout. Records per-theorem solved/failed/flake/path_error + winning_tactic + trace path,
and rolls up by theorem set and by namespace.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import os
import subprocess
import sys
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
_MAP = {"CONFIRMED_RC2_FAILURE": "failed", "NOW_SOLVED_BY_RC2": "solved",
        "RC2_SOLVED": "solved", "OPEN_FLAKE": "open_flake",
        "PATH_ERROR": "path_error", "TRACE_INSUFFICIENT": "trace_insufficient"}


def _p(*a):
    return os.path.join(_REPO, *a)


def run_worker(worker_out, cases_json, out_dir, wrapper, route_config, policy_type,
               top_k, max_steps, set_label):
    """eval_rollout_all over `cases` with `wrapper`; writes {fn: {status,...}} to worker_out."""
    cases = json.loads(cases_json)
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
            nm = "rc4r_bench_" + set_label
            _get, _list = E.get_theorems, E.list_theorem_sets
            E.list_theorem_sets = lambda: (list(_list()) + [nm]) if nm not in _list() else list(_list())
            E.get_theorems = lambda n: tcs if n == nm else _get(n)
            os.makedirs(out_dir, exist_ok=True)
            sys.argv = ["eval_rollout_all.py", "--theorem-set", nm, "--policy-type", policy_type,
                        "--route-config", route_config, "--strategy-config", wrapper,
                        "--top-k", str(top_k), "--max-steps", str(max_steps), "--out-dir", out_dir]
            try:
                E.main()
            except SystemExit:
                pass
            mfiles = sorted(glob.glob(os.path.join(out_dir, "eval-*", "metrics.json")),
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
                                   "winning_tactic": r.get("winning_tactic"), "trace_path": trace}
            else:
                for fn in names:
                    out[fn] = {"status": "trace_insufficient"}
    except Exception as e:
        for c in cases:
            out.setdefault(c["full_name"], {"status": "trace_insufficient", "error": str(e)[:120]})
    json.dump(out, open(worker_out, "w"), ensure_ascii=False, indent=2)


def build_reuse_map(result_files, confirmation_files):
    """{full_name: {status, ...}} from prior exact-config benchmark/literal results +
    floor-benchmark solved-name lists + TR confirmations."""
    m = {}
    for path in result_files:
        ap = _p(path)
        if not os.path.exists(ap):
            continue
        data = json.load(open(ap))
        # literal/benchmark results form
        for r in data.get("results", []):
            fn = r["full_name"]
            if fn in m:
                continue
            st = "solved" if r.get("rc2_finished") else r.get("rc2_status")
            if st:
                m[fn] = {"status": st, "winning_tactic": r.get("winning_tactic"),
                         "trace_path": r.get("trace_path"),
                         "provenance": "reused:" + os.path.basename(os.path.dirname(os.path.dirname(path)))}
        # floor-benchmark form: floors[].{rc2_solved/rc4d_solved, solved names live in run files}
    for path in confirmation_files:
        ap = _p(path)
        if not os.path.exists(ap):
            continue
        for r in json.load(open(ap)).get("results", []):
            fn = r["full_name"]
            if fn in m:
                continue
            st = "solved" if r.get("rc2_finished") is True else _MAP.get(r.get("classification"))
            if st:
                m[fn] = {"status": st, "winning_tactic": r.get("winning_tactic"),
                         "trace_path": r.get("trace_path"),
                         "provenance": "reused:" + os.path.basename(path).split(".")[0]}
    return m


def add_floor_reuse(m, floor_bench_json, side):
    """Reuse a full_floor_benchmark.json: per floor, the solved_names came from the run dirs;
    we only have counts there, so this seeds nothing per-theorem. Floors are run live here to
    get per-theorem status (cheap demo, reuse large via run dirs if present)."""
    return m  # floors are materialized fresh; per-theorem reuse handled by caller via run dirs


def run_benchmark(manifest, wrapper, route_config, out_dir, checkpoint, reuse_map,
                  worker_entry, top_k=8, max_steps=8, chunk_size=10, hard_timeout=1800,
                  only_sets=None, skip_predicate=None, label="rc2"):
    """Drive the benchmark. worker_entry = absolute path to the runner script (for --worker).
    skip_predicate(entry)->status_dict reuses/forces a result without a live run (e.g. RC4≡RC2
    on non-gate-firing theorems)."""
    theorems, seen, membership, entry_by_fn = [], set(), {}, {}
    for setname, rel in manifest["set_files"].items():
        if only_sets and setname not in only_sets:
            continue
        for e in json.load(open(_p(rel))):
            fn = e["full_name"]
            membership.setdefault(fn, []).append(setname)
            entry_by_fn.setdefault(fn, e)
            if fn not in seen:
                seen.add(fn)
                theorems.append({"full_name": fn, "file_path": e.get("file_path"),
                                 "namespace": e.get("namespace") or fn.split(".")[0]})
    ckpt = json.load(open(_p(checkpoint))) if os.path.exists(_p(checkpoint)) else {}
    results, live = {}, []
    forced = 0
    for t in theorems:
        fn = t["full_name"]
        if fn in ckpt:
            results[fn] = ckpt[fn]
        elif skip_predicate and skip_predicate(entry_by_fn[fn]) is not None:
            results[fn] = skip_predicate(entry_by_fn[fn]); forced += 1
        elif fn in reuse_map:
            results[fn] = reuse_map[fn]
        else:
            live.append(t)
    print(f"[rc4r-{label}] theorems={len(theorems)} reused={len(theorems)-len(live)-forced-sum(1 for t in theorems if t['full_name'] in ckpt)} "
          f"forced={forced} ckpt={sum(1 for t in theorems if t['full_name'] in ckpt)} live={len(live)}", flush=True)

    os.makedirs(_p(out_dir), exist_ok=True)
    chunks = [live[i:i + chunk_size] for i in range(0, len(live), chunk_size)]
    for ci, chunk in enumerate(chunks):
        wout = _p(out_dir, f"chunk_{ci}.json")
        cmd = [sys.executable, _TIMEOUT_HELPER, str(hard_timeout),
               sys.executable, worker_entry, "--worker", "--worker-out", wout,
               "--cases-json", json.dumps(chunk), "--out-dir", _p(out_dir, "runs"),
               "--wrapper", wrapper, "--route-config", route_config,
               "--top-k", str(top_k), "--max-steps", str(max_steps), "--set-label", label]
        print(f"[rc4r-{label}] chunk {ci+1}/{len(chunks)} ({len(chunk)}) ...", flush=True)
        subprocess.run(cmd, capture_output=True, text=True)
        try:
            wres = json.load(open(wout))
        except Exception:
            wres = {c["full_name"]: {"status": "trace_insufficient"} for c in chunk}
        for fn, rec in wres.items():
            rec.setdefault("provenance", label + "_live")
            results[fn] = rec
            ckpt[fn] = rec
        json.dump(ckpt, open(_p(checkpoint), "w"), ensure_ascii=False, indent=2)

    recs = []
    for t in theorems:
        fn = t["full_name"]
        r = results.get(fn, {"status": "trace_insufficient"})
        recs.append({"full_name": fn, "file_path": t["file_path"], "namespace": t["namespace"],
                     "sets": membership[fn], "status": r["status"],
                     "finished": r["status"] == "solved",
                     "winning_tactic": r.get("winning_tactic"), "trace_path": r.get("trace_path"),
                     "provenance": r.get("provenance")})
    return recs, membership


def rollup(recs, manifest):
    hist = Counter(r["status"] for r in recs)
    by_set = {}
    for setname in manifest["set_files"]:
        sub = [r for r in recs if setname in r["sets"]]
        by_set[setname] = {"n": len(sub),
                           "solved": sum(1 for r in sub if r["status"] == "solved"),
                           "failed": sum(1 for r in sub if r["status"] == "failed"),
                           "flake": sum(1 for r in sub if r["status"] == "open_flake"),
                           "path_error": sum(1 for r in sub if r["status"] == "path_error")}
    by_ns = {}
    for ns in sorted({r["namespace"] for r in recs}):
        sub = [r for r in recs if r["namespace"] == ns]
        by_ns[ns] = {"n": len(sub), "solved": sum(1 for r in sub if r["status"] == "solved")}
    return {"status_histogram": dict(hist), "by_set": by_set, "by_namespace": by_ns}
