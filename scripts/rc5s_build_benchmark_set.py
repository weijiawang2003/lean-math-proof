#!/usr/bin/env python3
"""RC5S Part 4 — build the hardened safety benchmark set.

A safety/hardening benchmark (not a coverage benchmark): the 3 RC5H true-hybrid winners + the
prior B10 stall cases (theorems whose RC5H top-5 carried simp_all/depth-3 programs) + off-policy
emission cases + TR7 dynamic-tail + Nat/Order hard negatives + eligible-but-no-win fresh
failures + a small canonical-floor smoke. Target 80–150 theorems. Each entry is tagged with its
safety category and whether its RC5H plan carried stall/off-policy programs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc5s_grammar as G

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RC5H = "project/evolve/experiments/rc5_hybrid"
TR7 = "project/evolve/experiments/tr7"


def _p(*a):
    return os.path.join(_REPO, *a)


def _j(*a):
    return json.load(open(_p(*a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    ap.add_argument("--max-floor-smoke", type=int, default=10)
    ap.add_argument("--max-hard-neg", type=int, default=12)
    args = ap.parse_args()

    plan = {t["full_name"]: t for t in _j(RC5H, "out/rc5h_dynamic_program_plan.json")["theorems"]}
    attr = _j(RC5H, "out/rc5h_hybrid_attribution.json")
    winners = set(attr["true_hybrid_delta_targets"])
    b5 = {r["full_name"]: r for r in _j(RC5H, "out/rc5h_b5_dynamic_results.json")["results"]}
    man = _j(RC5H, "cases/rc5h_benchmark_manifest.json")
    set_of = {}
    meta = {}
    for setname, rel in man["set_files"].items():
        for e in _j(rel):
            meta.setdefault(e["full_name"], e)
            set_of.setdefault(e["full_name"], setname)

    out_dir = os.path.dirname(_p(args.out_manifest))
    os.makedirs(out_dir, exist_ok=True)

    def base_entry(fn, category):
        t = plan.get(fn, {})
        e = meta.get(fn, {})
        ns = t.get("namespace") or e.get("namespace") or fn.split(".")[0]
        # does the RC5H plan for this theorem carry stall/off-policy programs?
        stall = offpol = 0
        for pgm in t.get("programs_ranked", [])[:10]:
            k, _ = G.classify_program(pgm.get("tactic"), ns)
            if k == "REMOVED_STALL_RISK":
                stall += 1
            elif k == "REMOVED_OFF_POLICY":
                offpol += 1
        b5r = b5.get(fn, {})
        return {"full_name": fn, "file_path": t.get("file_path") or e.get("file_path"),
                "namespace": ns, "goal_text": t.get("goal_text") or e.get("goal_text") or e.get("statement_text"),
                "statement_text": e.get("statement_text") or t.get("goal_text"),
                "category": category, "rc5h_set": set_of.get(fn),
                "rc5h_top10_stall_programs": stall, "rc5h_top10_offpolicy_programs": offpol,
                "rc5h_b5_success": bool(b5r.get("success")),
                "is_rc5h_winner": fn in winners,
                "dynamic_eligible": fn in plan}

    sets, seen = {}, set()

    def add(name, fns):
        entries = []
        for fn in fns:
            if fn in seen:
                continue
            e = base_entry(fn, name)
            if not e["file_path"]:
                continue
            seen.add(fn)
            entries.append(e)
        sets[name] = entries

    # 1) true winners
    add("true_winners", sorted(winners))
    # 2) prior stall cases: dynamic-eligible theorems whose RC5H top-10 carried stall programs
    stall_cases = sorted(fn for fn, t in plan.items()
                         if any(G.classify_program(p.get("tactic"), t.get("namespace"))[0] == "REMOVED_STALL_RISK"
                                for p in t.get("programs_ranked", [])[:10]))
    add("prior_stall_cases", stall_cases)
    # 3) off-policy emission cases
    offpol_cases = sorted(fn for fn, t in plan.items()
                          if any(G.classify_program(p.get("tactic"), t.get("namespace"))[0] == "REMOVED_OFF_POLICY"
                                 for p in t.get("programs_ranked", [])[:10]))
    add("off_policy_cases", offpol_cases)
    # 4) TR7 dynamic-tail
    tr7 = [json.loads(l) for l in open(_p(TR7, "data/tr7_fresh_delta_gap_examples.jsonl"))]
    tail = [r["full_name"] for r in tr7
            if r.get("dynamic_vs_static_class") in ("DYNAMIC_RETRIEVAL_PREFERRED",
                                                    "STATIC_WRAPPER_COMPATIBLE_WITH_ALLOWLIST_EXPANSION")]
    add("tr7_dynamic_tail", tail)
    # 5) Nat/Order hard negatives (from RC5H benchmark hard-neg set, namespace Nat/Int/Order)
    hardneg = [e["full_name"] for e in _j(RC5H, "cases/Multi_namespace_hard_negatives.json")][:args.max_hard_neg]
    add("hard_negatives", hardneg)
    # 6) eligible-but-no-win fresh failures (dynamic-eligible, not a winner, RC5H B5 no-win)
    nowin = sorted(fn for fn, t in plan.items()
                   if fn not in winners and not b5.get(fn, {}).get("success"))[:25]
    add("eligible_no_win", nowin)
    # 7) floor smoke
    floors = [e["full_name"] for e in _j(RC5H, "cases/canonical_floors.json")][:args.max_floor_smoke]
    add("floor_smoke", floors)

    set_files, sizes, ns_dist, cat_counts = {}, {}, {}, {}
    for name, entries in sets.items():
        path = os.path.join(out_dir, name + ".json")
        json.dump(entries, open(path, "w"), ensure_ascii=False, indent=2)
        set_files[name] = os.path.relpath(path, _REPO)
        sizes[name] = len(entries)
        ns_dist[name] = dict(Counter(e["namespace"] for e in entries))
    total = sum(sizes.values())
    manifest = {"generated_by": "scripts/rc5s_build_benchmark_set.py", "set_files": set_files,
                "sizes": sizes, "total": total, "unique_total": len(seen),
                "purpose": "safety/hardening benchmark (not coverage)"}
    json.dump(manifest, open(_p(args.out_manifest), "w"), ensure_ascii=False, indent=2)
    summary = {"generated_by": "scripts/rc5s_build_benchmark_set.py", "sizes": sizes,
               "total": total, "unique_total": len(seen), "namespace_distribution": ns_dist,
               "winners_included": sorted(winners),
               "dynamic_eligible_total": sum(1 for nm in sets for e in sets[nm] if e["dynamic_eligible"])}
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5S hardened benchmark set", "",
          f"- total: {total} | unique: {len(seen)} (safety/hardening, not coverage)", "",
          "| category | size |", "|---|---|"]
    for name in sizes:
        md.append(f"| {name} | {sizes[name]} |")
    md += ["", f"- winners included: {sorted(winners)}"]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5s-bench] sizes={sizes} total={total} unique={len(seen)}")


if __name__ == "__main__":
    main()
