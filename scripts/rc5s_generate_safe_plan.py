#!/usr/bin/env python3
"""RC5S Part 5 — build the safe dynamic program plan (strict-grammar only).

For each benchmark theorem, take the strict-filtered RC5H programs (already grammar-checked +
TR4-ranker-scored), reject anything off-policy BEFORE it can be scheduled, keep the top-5 (B5)
and a B10 reserve of safe NON-aesop families only. Dynamic-ineligible theorems (namespace
disabled / RC4-solved floors / hard negatives) are gated out with 0 programs. Hard requirement:
0 off-policy programs in the final plan.
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
FILTERED = "project/evolve/experiments/rc5_safety/out/rc5s_filtered_existing_plan.json"
SAFE_B10_FAMILIES = {"exact_L", "simpa_using_L", "simpa_L", "simp_L", "rw_L"}


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    policy = json.load(open(_p(args.policy)))
    pol = {"allowed_namespaces": policy["allowed_namespaces"], "aesop_namespaces": policy["aesop_namespaces"]}
    manifest = json.load(open(_p(args.manifest)))
    filt = {t["full_name"]: t for t in json.load(open(_p(FILTERED)))["theorems"]} \
        if os.path.exists(_p(FILTERED)) else {}

    # benchmark theorems (unique)
    bench = {}
    for setname, rel in manifest["set_files"].items():
        for e in json.load(open(_p(rel))):
            bench.setdefault(e["full_name"], e)

    theorems, gated = [], []
    total_b5 = total_b10 = total_rejected = offpolicy_final = 0
    fam_hist = Counter()
    for fn, e in bench.items():
        ns = e.get("namespace")
        ft = filt.get(fn)
        if not ft or ns not in policy["allowed_namespaces"]:
            gated.append({"full_name": fn, "namespace": ns,
                          "reason": "namespace_disabled" if ns not in policy["allowed_namespaces"]
                          else "not_dynamic_eligible"})
            continue
        progs = ft.get("programs_ranked", [])
        # double-check every program is policy-allowed (defensive — 0 off-policy guarantee)
        safe = []
        for p in progs:
            k, ok = G.classify_program(p.get("tactic"), ns, pol)
            if not ok:
                total_rejected += 1
                continue
            safe.append(p)
        safe.sort(key=lambda p: p.get("rank", 99))
        b5 = safe[:5]
        # B10 reserve: ranks beyond 5 but only safe non-aesop families
        b10_reserve = [p for p in safe[5:10]
                       if G.pattern_of(p.get("tactic")) in SAFE_B10_FAMILIES]
        for i, p in enumerate(b5, 1):
            p["rank"] = i
            p["budget_stage"] = "B5"
            p["rc5s_pattern"] = G.pattern_of(p.get("tactic"))
            fam_hist[p["rc5s_pattern"]] += 1
        for i, p in enumerate(b10_reserve, 6):
            p["rank"] = i
            p["budget_stage"] = "B10"
            p["rc5s_pattern"] = G.pattern_of(p.get("tactic"))
        progs_out = b5 + b10_reserve
        # final off-policy check
        for p in progs_out:
            if G.classify_program(p.get("tactic"), ns, pol)[1] is not True:
                offpolicy_final += 1
        total_b5 += len(b5)
        total_b10 += len(b10_reserve)
        theorems.append({"full_name": fn, "namespace": ns, "file_path": e.get("file_path"),
                         "rc2_status": ft.get("rc2_status"), "category": e.get("category"),
                         "num_programs": len(progs_out), "programs_ranked": progs_out})

    out = {"generated_by": "scripts/rc5s_generate_safe_plan.py", "policy": args.policy,
           "num_theorems_with_programs": len(theorems), "num_gated_out": len(gated),
           "total_b5_programs": total_b5, "total_b10_reserve_programs": total_b10,
           "candidates_rejected_at_generation": total_rejected,
           "off_policy_in_final_plan": offpolicy_final,
           "pattern_histogram_b5": dict(fam_hist),
           "gated_out": gated, "theorems": theorems}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5S safe dynamic plan", "",
          f"- theorems with programs: {len(theorems)} | gated out: {len(gated)}",
          f"- B5 programs: {total_b5} | B10 reserve: {total_b10} | rejected at generation: {total_rejected}",
          f"- **off-policy in final plan: {offpolicy_final}** (must be 0)",
          f"- B5 pattern histogram: {dict(fam_hist)}", "",
          "| theorem | ns | category | #programs | top tactic |", "|---|---|---|---|---|"]
    for t in sorted(theorems, key=lambda x: x["full_name"]):
        top = (t["programs_ranked"] or [{}])[0]
        md.append(f"| `{t['full_name']}` | {t['namespace']} | {t.get('category')} | "
                  f"{t['num_programs']} | `{top.get('tactic','')}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5s-safeplan] theorems={len(theorems)} gated={len(gated)} B5={total_b5} "
          f"B10_reserve={total_b10} off_policy_final={offpolicy_final}")
    print(f"[rc5s-safeplan] B5 patterns={dict(fam_hist)}")


if __name__ == "__main__":
    main()
