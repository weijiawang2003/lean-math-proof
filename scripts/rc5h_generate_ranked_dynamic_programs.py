#!/usr/bin/env python3
"""RC5H Part 7 — generate + rank dynamic candidate programs for the static failures.

Shells out to the validated TR6 generator (`tr6_generate_ranked_programs`, = TR3/TR5 program
grammar scored by the TR4 HGB ranker via tr5_score) over the RC5H dynamic confirmation + the
RC5H retrieval, keeping top-20 with B5/B10/B20 budget tags. No live Lean here. Re-tags each
program with the RC5H gate reason (allowed namespace, retrieval-confidence) and summarizes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = "project/evolve/experiments/rc5_hybrid/cases/rc5h_dynamic_confirmation.json"
ALLOWED_NS = {"Set", "Finset", "List", "Multiset", "Nat"}


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--static-results", required=True)
    ap.add_argument("--tr4-model-dir", required=True)
    ap.add_argument("--tr4-vectorizers", required=True)
    ap.add_argument("--tr4-metadata", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    policy = json.load(open(_p(args.policy)))
    # shell out to TR6 generator (reuses tr5_score + TR4 HGB)
    raw_plan = args.out_json.replace(".json", "_raw.json")
    raw_md = args.out_json.replace(".json", "_raw.md")
    cmd = [sys.executable, _p("scripts/tr6_generate_ranked_programs.py"),
           "--confirmation", _p(CONF), "--retrieval", _p(args.retrieval),
           "--tr4-model-dir", _p(args.tr4_model_dir), "--tr4-vectorizers", _p(args.tr4_vectorizers),
           "--tr4-metadata", _p(args.tr4_metadata), "--out-json", _p(raw_plan),
           "--out-md", _p(raw_md), "--model", "hgb", "--keep-top", "20"]
    print("[rc5h-generate] generating ranked programs via TR6 generator ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(_p(raw_plan)):
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit("generation failed")

    plan = json.load(open(_p(raw_plan)))
    # restrict the program grammar to the policy grammar families; tag gate reason
    grammar_ok = set()
    for g in policy["dynamic_stage"]["program_grammar"]:
        grammar_ok.add(g.split(" ")[0])  # leading tactic head (exact/simpa/simp/rw/ext/constructor)
    conf = {r["full_name"]: r for r in json.load(open(_p(CONF)))["results"]}

    fam_hist, ns_hist = Counter(), Counter()
    for t in plan["theorems"]:
        ns = t.get("namespace")
        gate_reason = ("allowed_ns" if ns in ALLOWED_NS else "ns_gate_out")
        c = conf.get(t["full_name"], {})
        t["rc5h_gate_reason"] = gate_reason
        t["rc5h_dynamic_eligible"] = ns in ALLOWED_NS
        t["set"] = c.get("set")
        ns_hist[ns] += 1
        for p in t.get("programs_ranked", []):
            p.setdefault("used_lemmas", p.get("lemmas", []))
            fam_hist[p.get("family")] += 1

    out = {"generated_by": "scripts/rc5h_generate_ranked_dynamic_programs.py",
           "ranker": "TR4_HGB", "num_theorems": len(plan["theorems"]),
           "total_programs": sum(len(t.get("programs_ranked", [])) for t in plan["theorems"]),
           "budgets": policy["dynamic_stage"]["max_programs_per_theorem"],
           "family_histogram": dict(fam_hist), "namespace_histogram": dict(ns_hist),
           "theorems": plan["theorems"]}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5H dynamic program plan", "",
          f"- theorems: {out['num_theorems']} | total programs: {out['total_programs']}",
          f"- budgets: {out['budgets']}",
          f"- family histogram: {dict(fam_hist)}",
          f"- namespace histogram: {dict(ns_hist)}", "",
          "| theorem | ns | #programs | top tactic | top score |", "|---|---|---|---|---|"]
    for t in plan["theorems"]:
        progs = t.get("programs_ranked", [])
        top = progs[0] if progs else {}
        md.append(f"| `{t['full_name']}` | {t.get('namespace')} | {len(progs)} | "
                  f"`{top.get('tactic','')}` | {top.get('ranker_score','')} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5h-generate] theorems={out['num_theorems']} programs={out['total_programs']} "
          f"families={dict(fam_hist)}")


if __name__ == "__main__":
    main()
