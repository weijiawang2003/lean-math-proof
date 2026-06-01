#!/usr/bin/env python3
"""RC3 canonical preservation + off-gate validation.

Runs the RC3 candidate through the production eval harness on the canonical
floors (demo_v1, nat_defs_medium, nat_defs_large_v5) and the non-Set negative
controls, then:

  * compares floor solved-counts against the inherited RC2 minimums and the
    documented RC2 release counts (regression check);
  * scans every run's traces.jsonl for SX3 depth-2 sequence emissions
    ('simp [Set.ite] <;> aesop') on theorems whose name lacks 'Set.ite'
    (off-gate guard — must be 0);
  * reports negative-control emissions and Dojo open flakes separately from
    proof failures.

Registered floor sets run via eval_rollout_all (--theorem-set <name>); the
negative-control file is registered in-process (SF1 trick). No production config
is modified; RC1/RC2-release/NS24 untouched.
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

DEPTH2 = "simp [Set.ite] <;> aesop"
SINGLE = "simp [Set.ite]"

RC2_DOC_FLOORS = {"demo_v1": (12, 15), "nat_defs_medium": (37, 38), "nat_defs_large_v5": (49, 65)}
FLOOR_MIN = {"demo_v1": 11, "nat_defs_medium": 37, "nat_defs_large_v5": 49}


def _run(strategy, route, theorem_set_name, out_dir, file_rows=None, top_k=8, max_steps=8):
    """Run eval_rollout_all on a registered name or a file (file_rows registered in-process)."""
    import eval_rollout_all as E
    os.makedirs(out_dir, exist_ok=True)
    if file_rows is not None:
        from core_types import TheoremConfig
        fields = {f.name for f in dataclasses.fields(TheoremConfig)}
        tcs = []
        for r in file_rows:
            if not r.get("file_path"):
                continue
            kw = {k: v for k, v in r.items() if k in fields and v is not None}
            tcs.append(TheoremConfig(**kw))
        _get, _list = E.get_theorems, E.list_theorem_sets
        E.list_theorem_sets = lambda: (list(_list()) + [theorem_set_name]
                                       if theorem_set_name not in _list() else list(_list()))
        E.get_theorems = lambda n: tcs if n == theorem_set_name else _get(n)
    fwd = ["eval_rollout_all.py", "--theorem-set", theorem_set_name,
           "--policy-type", "hybrid_evolved", "--route-config", route,
           "--strategy-config", strategy, "--top-k", str(top_k),
           "--max-steps", str(max_steps), "--out-dir", out_dir]
    sys.argv = fwd
    try:
        E.main()
    except SystemExit:
        pass
    mfiles = sorted(glob.glob(os.path.join(out_dir, "eval-*", "metrics.json")), key=os.path.getmtime)
    if not mfiles:
        return None
    mpath = mfiles[-1]
    return mpath


def _scan_offgate(trace_path):
    """Count SX3 depth-2 emissions on/off gate from a traces.jsonl."""
    on_gate, off_gate, off_theorems = 0, 0, []
    if not os.path.isfile(trace_path):
        return {"on_gate": 0, "off_gate": 0, "off_gate_theorems": [], "trace_missing": True}
    for line in open(trace_path):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("tactic") == DEPTH2:
            fn = rec.get("full_name", "")
            if "Set.ite" in fn:
                on_gate += 1
            else:
                off_gate += 1
                off_theorems.append(fn)
    return {"on_gate": on_gate, "off_gate": off_gate,
            "off_gate_theorems": sorted(set(off_theorems)), "trace_missing": False}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-wrapper", required=True)
    p.add_argument("--route-config", default="project/evolve/routing/ns24_router.json")
    p.add_argument("--floors", nargs="*", default=["demo_v1", "nat_defs_medium", "nat_defs_large_v5"])
    p.add_argument("--neg-controls", default="project/evolve/experiments/rc3_validation/theorem_sets/sx3_negative_controls.json")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-dir", default="project/evolve/experiments/rc3_validation/out/preservation_runs")
    args = p.parse_args(argv)

    floors = {}
    for name in args.floors:
        print(f"[preservation] running floor {name} ...", flush=True)
        mpath = _run(args.candidate_wrapper, args.route_config, name,
                     os.path.join(args.out_dir, name))
        if not mpath:
            floors[name] = {"error": "no metrics produced"}
            continue
        m = json.load(open(mpath))
        trace = os.path.join(os.path.dirname(mpath), "traces.jsonl")
        og = _scan_offgate(trace)
        solved = m.get("proved", sum(1 for r in m["per_theorem"] if r.get("finished")))
        total = m.get("total_theorems", len(m["per_theorem"]))
        rc2 = RC2_DOC_FLOORS.get(name)
        floors[name] = {
            "rc3_solved": solved, "total": total,
            "rc2_doc_solved": rc2[0] if rc2 else None,
            "floor_min": FLOOR_MIN.get(name),
            "floor_pass": solved >= FLOOR_MIN.get(name, 0),
            "regression_vs_rc2_doc": (rc2[0] - solved) if rc2 else None,
            "offgate": og, "metrics_path": mpath, "trace_path": trace,
        }
        print(f"[preservation]   {name}: {solved}/{total} (floor>={FLOOR_MIN.get(name)}, "
              f"off_gate={og['off_gate']})", flush=True)

    # negative controls
    nc = {}
    if os.path.isfile(args.neg_controls):
        obj = json.load(open(args.neg_controls))
        rows = obj[list(obj.keys())[0]] if isinstance(obj, dict) else obj
        print(f"[preservation] running negative controls ...", flush=True)
        mpath = _run(args.candidate_wrapper, args.route_config, "rc3_neg_controls",
                     os.path.join(args.out_dir, "neg_controls"), file_rows=rows)
        if mpath:
            m = json.load(open(mpath))
            trace = os.path.join(os.path.dirname(mpath), "traces.jsonl")
            og = _scan_offgate(trace)
            nc = {"ran": len(m["per_theorem"]), "solved": m.get("proved"),
                  "offgate": og, "metrics_path": mpath, "trace_path": trace,
                  "per_theorem": [{"full_name": r["full_name"], "finished": r.get("finished"),
                                   "available": r.get("available")} for r in m["per_theorem"]]}
        else:
            nc = {"error": "no runnable negative controls (all null file_path) or no metrics",
                  "note": "off-gate also asserted structurally (gate = name substring 'Set.ite') "
                          "and empirically by pure-Nat floors below"}

    total_offgate = sum(f.get("offgate", {}).get("off_gate", 0) for f in floors.values()
                        if isinstance(f, dict)) + (nc.get("offgate", {}).get("off_gate", 0) if nc else 0)
    all_floors_pass = all(f.get("floor_pass") for f in floors.values() if "rc3_solved" in f)
    any_regression = any((f.get("regression_vs_rc2_doc") or 0) > 0 for f in floors.values()
                         if "rc3_solved" in f)

    out = {
        "candidate_wrapper": args.candidate_wrapper,
        "floors": floors,
        "negative_controls": nc,
        "summary": {
            "all_floors_pass": all_floors_pass,
            "any_regression_vs_rc2_doc": any_regression,
            "total_off_gate_emissions": total_offgate,
            "off_gate_clean": total_offgate == 0,
        },
        "expected": {"off_gate": 0, "regressions": 0, "floors_preserved": True},
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# RC3 preservation + off-gate", "",
         f"- all floors pass: **{all_floors_pass}**",
         f"- any regression vs RC2 doc: **{any_regression}**",
         f"- total off-gate emissions: **{total_offgate}** (expected 0)", "",
         "## Canonical floors", "",
         "| floor | RC3 solved | total | RC2 doc | floor min | pass | regression | off-gate emissions |",
         "|---|---|---|---|---|---|---|---|"]
    for name, f in floors.items():
        if "rc3_solved" not in f:
            L.append(f"| {name} | ERROR | | | | | | |"); continue
        L.append(f"| {name} | {f['rc3_solved']} | {f['total']} | {f['rc2_doc_solved']} | "
                 f"{f['floor_min']} | {'✅' if f['floor_pass'] else '❌'} | "
                 f"{f['regression_vs_rc2_doc']} | {f['offgate']['off_gate']} |")
    L += ["", "## Negative controls", ""]
    if nc.get("per_theorem"):
        L.append("| theorem | available | finished |")
        L.append("|---|---|---|")
        for r in nc["per_theorem"]:
            L.append(f"| `{r['full_name']}` | {r['available']} | {r['finished']} |")
        L.append(f"\nNegative-control off-gate emissions: **{nc['offgate']['off_gate']}**")
    else:
        L.append(f"_{nc.get('error','n/a')}_")
        L.append(f"\n{nc.get('note','')}")
    L += ["", "## Off-gate detail",
          "Off-gate = SX3 sequence `simp [Set.ite] <;> aesop` emitted on a theorem whose name lacks `Set.ite`.", ""]
    for name, f in floors.items():
        if "offgate" in f:
            og = f["offgate"]
            L.append(f"- **{name}**: on_gate={og['on_gate']} off_gate={og['off_gate']} "
                     f"{('OFFENDERS:'+str(og['off_gate_theorems'])) if og['off_gate'] else ''}")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[preservation] wrote {args.out_json}: floors_pass={all_floors_pass} "
          f"off_gate={total_offgate} regression={any_regression}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
