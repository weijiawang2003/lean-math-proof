#!/usr/bin/env python3
"""TR2 Part 4 — confirm literal RC2 status for the selected batch cases.

Reuse-first, live-fallback. Every selected case already has an identical-config
literal-RC2 result in the SF4 confirmation artifact (same wrapper / route / policy /
top-k / max-steps), so by default this REUSES that verified oracle and runs live
LeanDojo ONLY for cases with no SF4 (or TR1) record. Provenance is recorded per case:

  sf4_reused  — classification taken from SF4 identical-config literal-RC2 run
  tr1_reused  — classification taken from the TR1 example rc2_status (no SF4 row)
  tr2_live    — freshly confirmed live in this run

Classifications:
  CONFIRMED_RC2_FAILURE   available, finished == false  (eligible for true delta)
  RC2_SOLVED              finished == true              (negative / baseline-duplicate)
  OPEN_FLAKE              not available (Dojo open/skip)
  PATH_ERROR              file_path unresolved / theorem absent
  TRACE_INSUFFICIENT      ran but no usable record

Protected configs are read-only; nothing on disk is modified.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SF4_CONF_DEFAULT = "project/evolve/experiments/sf4/out/rc2_failure_confirmation.json"
POOL_DEFAULT = "project/evolve/experiments/tr2/cases/tr2_candidate_pool.jsonl"

_MAP = {"CONFIRMED_RC2_FAILURE": "CONFIRMED_RC2_FAILURE", "NOW_SOLVED_BY_RC2": "RC2_SOLVED",
        "OPEN_FLAKE": "OPEN_FLAKE", "PATH_ERROR": "PATH_ERROR", "TRACE_INSUFFICIENT": "TRACE_INSUFFICIENT"}


def _all_members(manifest):
    seen, out = set(), []
    for s in ("model", "rule", "random"):
        for fn in manifest["batches"][s]["members"]:
            if fn not in seen:
                seen.add(fn); out.append(fn)
    return out


def _live_confirm(cases, args):
    """Run literal RC2 live on the given [{full_name,file_path}] via the SF1
    in-process registration trick. Returns {full_name: rec}. Empty if no cases."""
    if not cases:
        return {}, None
    sys.path.insert(0, _REPO)
    from core_types import TheoremConfig
    fields = {f.name for f in dataclasses.fields(TheoremConfig)}
    tcs, names = [], {}
    for c in cases:
        if not c.get("file_path"):
            continue
        kw = {"file_path": c["file_path"], "full_name": c["full_name"]}
        tcs.append(TheoremConfig(**{k: v for k, v in kw.items() if k in fields}))
        names[c["full_name"]] = c
    if not tcs:
        return {}, None
    import eval_rollout_all as E
    os.makedirs(args.out_dir, exist_ok=True)
    name = "tr2_confirm"
    _get, _list = E.get_theorems, E.list_theorem_sets
    E.list_theorem_sets = lambda: (list(_list()) + [name]) if name not in _list() else list(_list())
    E.get_theorems = lambda n: tcs if n == name else _get(n)
    sys.argv = ["eval_rollout_all.py", "--theorem-set", name,
                "--policy-type", args.policy_type, "--route-config", args.route_config,
                "--strategy-config", args.rc2_wrapper, "--top-k", str(args.top_k),
                "--max-steps", str(args.max_steps), "--out-dir", args.out_dir]
    print(f"[tr2-confirm] LIVE literal RC2 on {len(tcs)} unknown case(s) ...", flush=True)
    try:
        E.main()
    except SystemExit:
        pass
    mfiles = sorted(glob.glob(os.path.join(args.out_dir, "eval-*", "metrics.json")), key=os.path.getmtime)
    if not mfiles:
        return {fn: {"classification": "TRACE_INSUFFICIENT"} for fn in names}, None
    m = json.load(open(mfiles[-1]))
    trace = os.path.join(os.path.dirname(mfiles[-1]), "traces.jsonl")
    seen = {r["full_name"]: r for r in m.get("per_theorem", [])}
    out = {}
    for fn in names:
        r = seen.get(fn)
        if r is None:
            out[fn] = {"classification": "PATH_ERROR", "rc2_finished": None}
        elif not r.get("available"):
            out[fn] = {"classification": "OPEN_FLAKE", "rc2_finished": None}
        elif r.get("finished"):
            out[fn] = {"classification": "RC2_SOLVED", "rc2_finished": True,
                       "num_steps": r.get("num_steps"), "winning_tactic": r.get("winning_tactic")}
        else:
            out[fn] = {"classification": "CONFIRMED_RC2_FAILURE", "rc2_finished": False,
                       "num_steps": r.get("num_steps"), "tactics_used": r.get("tactics_used")}
        out[fn]["trace_path"] = trace
    return out, mfiles[-1]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--batch-manifest", required=True)
    p.add_argument("--rc2-wrapper", required=True)
    p.add_argument("--route-config", required=True)
    p.add_argument("--policy-type", default="hybrid_evolved")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--sf4-confirmation", default=SF4_CONF_DEFAULT)
    p.add_argument("--pool", default=POOL_DEFAULT)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-dir", default="project/evolve/experiments/tr2/out/tr2_confirm_runs")
    p.add_argument("--no-live", action="store_true", help="never spawn live runs; mark unknowns TRACE_INSUFFICIENT")
    args = p.parse_args(argv)

    manifest = json.load(open(args.batch_manifest))
    members = _all_members(manifest)
    in_batch = {fn: [s for s in ("model", "rule", "random") if fn in manifest["batches"][s]["members"]]
                for fn in members}

    sf4 = json.load(open(args.sf4_confirmation)) if os.path.exists(args.sf4_confirmation) else {"results": []}
    sf4_by = {r["full_name"]: r for r in sf4.get("results", [])}
    pool = {r["full_name"]: r for r in (json.loads(l) for l in open(args.pool) if l.strip())}

    results, unknown = [], []
    for fn in members:
        prow = pool.get(fn, {})
        if fn in sf4_by:
            sr = sf4_by[fn]
            cls = _MAP.get(sr.get("classification"), "TRACE_INSUFFICIENT")
            results.append({"full_name": fn, "file_path": sr.get("file_path") or prow.get("file_path"),
                            "namespace": sr.get("namespace") or prow.get("namespace"),
                            "classification": cls, "rc2_finished": sr.get("rc2_finished"),
                            "num_steps": sr.get("num_steps"), "winning_tactic": sr.get("winning_tactic"),
                            "tactics_used": sr.get("tactics_used"), "trace_path": sr.get("trace_path"),
                            "provenance": "sf4_reused", "in_batches": in_batch[fn]})
        elif prow.get("known_rc2_status") in ("solved", "failed"):
            cls = "RC2_SOLVED" if prow["known_rc2_status"] == "solved" else "CONFIRMED_RC2_FAILURE"
            results.append({"full_name": fn, "file_path": prow.get("file_path"),
                            "namespace": prow.get("namespace"), "classification": cls,
                            "rc2_finished": prow["known_rc2_status"] == "solved",
                            "provenance": "tr1_reused", "in_batches": in_batch[fn]})
        else:
            unknown.append({"full_name": fn, "file_path": prow.get("file_path"),
                            "namespace": prow.get("namespace")})

    live_map, live_metrics = ({}, None)
    if unknown and not args.no_live:
        live_map, live_metrics = _live_confirm(unknown, args)
    for u in unknown:
        fn = u["full_name"]
        lv = live_map.get(fn, {"classification": "TRACE_INSUFFICIENT"})
        results.append({"full_name": fn, "file_path": u["file_path"], "namespace": u["namespace"],
                        "classification": lv["classification"], "rc2_finished": lv.get("rc2_finished"),
                        "num_steps": lv.get("num_steps"), "winning_tactic": lv.get("winning_tactic"),
                        "tactics_used": lv.get("tactics_used"), "trace_path": lv.get("trace_path"),
                        "provenance": "tr2_live" if not args.no_live else "unconfirmed", "in_batches": in_batch[fn]})

    import collections
    hist = collections.Counter(r["classification"] for r in results)
    prov = collections.Counter(r["provenance"] for r in results)
    confirmed = sorted(r["full_name"] for r in results if r["classification"] == "CONFIRMED_RC2_FAILURE")
    out = {"batch_manifest": args.batch_manifest, "rc2_wrapper": args.rc2_wrapper,
           "route_config": args.route_config, "policy_type": args.policy_type,
           "top_k": args.top_k, "max_steps": args.max_steps,
           "num_cases": len(results), "num_live_runs": len(unknown) if not args.no_live else 0,
           "live_metrics_path": live_metrics,
           "classification_histogram": dict(hist), "provenance_histogram": dict(prov),
           "confirmed_rc2_failures": confirmed, "num_confirmed": len(confirmed),
           "eligible_for_true_delta": confirmed,
           "reuse_note": ("Identical-config literal-RC2 confirmation already exists for every selected "
                          "case (SF4); reused as oracle. Live runs occur only for cases lacking a record."),
           "results": results}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# TR2 RC2 confirmation", "",
         f"- cases: **{len(results)}**  ·  confirmed RC2 failures: **{len(confirmed)}**  ·  live runs: {out['num_live_runs']}",
         f"- classification: {dict(hist)}",
         f"- provenance: {dict(prov)}", "",
         f"> {out['reuse_note']}", "",
         "| theorem | class | finished | provenance | batches |", "|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (r["classification"], r["full_name"])):
        L.append(f"| `{r['full_name']}` | {r['classification']} | {r.get('rc2_finished')} | "
                 f"{r['provenance']} | {','.join(r['in_batches'])} |")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[tr2-confirm] cases={len(results)} confirmed={len(confirmed)} live={out['num_live_runs']} "
          f"hist={dict(hist)} prov={dict(prov)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
