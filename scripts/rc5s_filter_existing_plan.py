#!/usr/bin/env python3
"""RC5S Part 3 — filter the existing RC5H dynamic plan through the strict policy.

Classifies every RC5H-emitted program (POLICY_ALLOWED / REMOVED_STALL_RISK / REMOVED_OFF_POLICY /
REMOVED_NAMESPACE_DISABLED / REMOVED_LOW_CONFIDENCE / REMOVED_DUPLICATE), writes the filtered
plan (allowed programs only, re-ranked), and a report. Hard check: the 3 RC5H TRUE_HYBRID_DELTA
winning programs must survive.
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
RC5H_B5 = "project/evolve/experiments/rc5_hybrid/out/rc5h_b5_dynamic_results.json"
RC5H_B10 = "project/evolve/experiments/rc5_hybrid/out/rc5h_b10_dynamic_results.json"


def _p(*a):
    return os.path.join(_REPO, *a)


def _winning_programs():
    """{full_name: winning_tactic} for the RC5H true-hybrid wins."""
    out = {}
    for path in (RC5H_B5, RC5H_B10):
        if not os.path.exists(_p(path)):
            continue
        for r in json.load(open(_p(path)))["results"]:
            if r.get("success"):
                out.setdefault(r["full_name"], (r.get("winning_program") or {}).get("tactic"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--rc5h-plan", required=True)
    ap.add_argument("--out-plan", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--min-score", type=float, default=0.0)
    args = ap.parse_args()

    policy = json.load(open(_p(args.policy)))
    pol = {"allowed_namespaces": policy["allowed_namespaces"], "aesop_namespaces": policy["aesop_namespaces"]}
    plan = json.load(open(_p(args.rc5h_plan)))
    winners = _winning_programs()

    hist = Counter()
    orig_count = 0
    filtered_theorems = []
    winner_survival = {}
    for t in plan["theorems"]:
        ns = t.get("namespace")
        kept, seen_tac = [], set()
        for pgm in t.get("programs_ranked", []):
            orig_count += 1
            tac = pgm.get("tactic")
            klass, allowed = G.classify_program(tac, ns, pol)
            if allowed:
                if tac in seen_tac:
                    klass, allowed = "REMOVED_DUPLICATE", False
                elif (pgm.get("ranker_score") or 0) < args.min_score:
                    klass, allowed = "REMOVED_LOW_CONFIDENCE", False
            hist[klass] += 1
            if allowed:
                seen_tac.add(tac)
                kept.append({**pgm, "rc5s_class": klass})
            # track winner survival
            if t["full_name"] in winners and tac == winners[t["full_name"]]:
                winner_survival[t["full_name"]] = {"tactic": tac, "class": klass, "survives": allowed}
        # re-rank kept programs (preserve ranker order; reassign 1..n)
        kept.sort(key=lambda p: p.get("rank", 99))
        for i, p in enumerate(kept, 1):
            p["rank"] = i
        if kept:
            filtered_theorems.append({**{k: t[k] for k in ("full_name", "namespace", "file_path",
                                                            "rc2_status", "set") if k in t},
                                      "programs_ranked": kept, "num_programs": len(kept)})

    out_plan = {"generated_by": "scripts/rc5s_filter_existing_plan.py",
                "source_plan": args.rc5h_plan, "policy": args.policy,
                "num_theorems": len(filtered_theorems), "theorems": filtered_theorems}
    json.dump(out_plan, open(_p(args.out_plan), "w"), ensure_ascii=False, indent=2)

    filtered_count = sum(len(t["programs_ranked"]) for t in filtered_theorems)
    winners_survive = all(v["survives"] for v in winner_survival.values()) and len(winner_survival) == len(winners)
    report = {
        "generated_by": "scripts/rc5s_filter_existing_plan.py",
        "original_program_count": orig_count,
        "filtered_program_count": filtered_count,
        "removed_total": orig_count - filtered_count,
        "classification_histogram": dict(hist),
        "off_policy_removed": hist.get("REMOVED_OFF_POLICY", 0),
        "stall_risk_removed": hist.get("REMOVED_STALL_RISK", 0),
        "namespace_disabled_removed": hist.get("REMOVED_NAMESPACE_DISABLED", 0),
        "rc5h_true_hybrid_winners": list(winners),
        "winner_survival": winner_survival,
        "all_3_winners_survive": winners_survive,
        "lost_winners": [fn for fn, v in winner_survival.items() if not v["survives"]],
    }
    json.dump(report, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5S existing-plan filter", "",
          f"- original programs: {orig_count} → filtered (allowed): **{filtered_count}** "
          f"(removed {orig_count - filtered_count})",
          f"- classification: {dict(hist)}",
          f"- off-policy removed: {report['off_policy_removed']} | stall-risk removed: "
          f"{report['stall_risk_removed']} | namespace-disabled: {report['namespace_disabled_removed']}",
          f"- **all 3 RC5H true-hybrid winners survive: {winners_survive}**", "",
          "## Winner survival", "", "| theorem | winning tactic | class | survives |", "|---|---|---|---|"]
    for fn, v in winner_survival.items():
        md.append(f"| `{fn}` | `{v['tactic']}` | {v['class']} | {v['survives']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5s-filter] {orig_count}->{filtered_count} | {dict(hist)}")
    print(f"[rc5s-filter] all_3_winners_survive={winners_survive} survival={ {k: v['survives'] for k,v in winner_survival.items()} }")


if __name__ == "__main__":
    main()
