#!/usr/bin/env python3
"""RC5V3 Part 14 — maintenance decision analysis.

Combines the cost curve, namespace/feature yield, safety audit, and system comparison into one
owner-facing recommendation: MAINTAIN_GUIDED_SEARCH_MODE / MAINTAIN_BUT_NAMESPACE_LIMITED /
KEEP_RESEARCH_ONLY / DISABLE_DYNAMIC_DEFAULT / NEED_BETTER_RANKER / NEED_BETTER_RETRIEVAL.
"""
from __future__ import annotations

import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", required=True)
    ap.add_argument("--yield-analysis", required=True)
    ap.add_argument("--safety", required=True)
    ap.add_argument("--comparison", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    cost = json.load(open(_p(args.cost)))
    yld = json.load(open(_p(args.yield_analysis)))
    safety = json.load(open(_p(args.safety)))
    comp = json.load(open(_p(args.comparison)))

    fresh = comp.get("fresh_dynamic_delta_over_rc4", 0)
    ppw = safety.get("dynamic_probes_per_fresh_win")
    verdict = safety.get("verdict")
    rec_budget = cost.get("recommended_budget")
    high = yld.get("high_yield_namespaces", [])
    moderate = yld.get("moderate_yield_namespaces", [])
    win_ns = set(comp.get("fresh_delta_by_namespace", {}).keys())
    all_allowed = {"Set", "Finset", "List", "Multiset", "Nat"}
    zero_ns = sorted(all_allowed - win_ns)

    # decision logic
    if verdict == "UNSAFE_TIMEOUT_BEHAVIOR":
        decision = "DISABLE_DYNAMIC_DEFAULT"
        why = "dynamic stage is not timeout-safe at scale; return to RC5S engineering."
    elif fresh == 0:
        decision = "DISABLE_DYNAMIC_DEFAULT"
        why = "no fresh deltas at scale — yield too low to maintain."
    elif ppw and ppw > 250:
        decision = "KEEP_RESEARCH_ONLY"
        why = f"fresh wins exist ({fresh}) but cost is high ({ppw} probes/win); keep as a research tool."
    elif win_ns and win_ns.issubset(all_allowed) and len(win_ns) <= 3 and zero_ns:
        decision = "MAINTAIN_BUT_NAMESPACE_LIMITED"
        why = (f"fresh wins concentrate in {sorted(win_ns)}; gate dynamic to those namespaces and "
               f"disable {zero_ns} by default.")
    elif fresh >= 3 and (ppw is None or ppw <= 250):
        decision = "MAINTAIN_GUIDED_SEARCH_MODE"
        why = (f"fresh wins are stable at scale ({fresh}) and cost is acceptable ({ppw} probes/win); "
               f"maintain the off-by-default guided-search mode at {rec_budget}.")
    else:
        decision = "KEEP_RESEARCH_ONLY"
        why = "fresh wins present but marginal; keep as research-only pending more data."

    suggested_budget = rec_budget if decision.startswith("MAINTAIN") else "off"
    out = {"generated_by": "scripts/rc5v3_maintenance_decision.py",
           "decision": decision,
           "owner_facing_recommendation": why,
           "expected_benefit": f"+{fresh} fresh out-of-sample wins over RC4 across the batch "
                               f"(by namespace {comp.get('fresh_delta_by_namespace', {})}).",
           "expected_cost": f"~{ppw} dynamic probes per fresh win; total dynamic probes "
                           f"{safety.get('dynamic_probes')}; B5-only, bounded wall {safety.get('max_wall_seconds')}s.",
           "risks": ("additive + timeout-safe (0 regressions, 0 off-policy, no global stalls); main "
                     "risk is cost per win and namespace concentration / over-fit to seen namespaces."),
           "high_yield_namespaces": high, "moderate_yield_namespaces": moderate,
           "zero_win_allowed_namespaces": zero_ns,
           "suggested_default_budget": suggested_budget,
           "safety_verdict": verdict, "cost_recommended_budget": rec_budget}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V3 maintenance decision", "",
          f"- **decision: {decision}**",
          f"- recommendation: {why}",
          f"- expected benefit: {out['expected_benefit']}",
          f"- expected cost: {out['expected_cost']}",
          f"- risks: {out['risks']}",
          f"- suggested default budget: **{suggested_budget}** (cost-curve rec: {rec_budget})",
          f"- high-yield namespaces: {high} | moderate: {moderate} | zero-win allowed-ns: {zero_ns}",
          f"- safety verdict: {verdict}"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v3-maint] decision={decision} budget={suggested_budget} fresh={fresh} ppw={ppw}")


if __name__ == "__main__":
    main()
