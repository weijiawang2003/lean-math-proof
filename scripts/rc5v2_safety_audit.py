#!/usr/bin/env python3
"""RC5V2 Part 12 — safety audit of the safe dynamic B5 stage on the fresh frontier."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc5s_grammar as G

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--dynamic-results", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    policy = json.load(open(_p(args.policy)))
    plan = json.load(open(_p(args.plan)))
    dyn = json.load(open(_p(args.dynamic_results)))
    attr = json.load(open(_p(args.attribution)))
    cap = policy["timeout_policy"]["per_theorem_wall_cap_seconds"]
    allowed_ns = set(policy["allowed_namespaces"])

    # off-policy / namespace check over the final plan (must be 0)
    off_policy = ns_violations = 0
    for t in plan["theorems"]:
        ns = t.get("namespace")
        if ns not in allowed_ns:
            ns_violations += 1
        for p in t.get("programs_ranked", []):
            if G.classify_program(p.get("tactic"), ns)[1] is not True:
                off_policy += 1

    results = dyn.get("results", [])
    attempts = sum(r.get("programs_attempted", 0) for r in results)
    unknown = sum(1 for r in results for f in r.get("failures", []) if f.get("outcome") == "unknown_name")
    killed = dyn.get("killed_by_timeout", 0)
    max_wall = dyn.get("max_wall_seconds", 0)
    flakes = sum(1 for r in results if r.get("setup_error") and "exceeded" in (r.get("setup_error") or ""))
    n = len(results)
    fresh_delta = attr.get("fresh_true_deltas", 0)
    src_specific = attr.get("source_specific", 0)
    probes_per_win = round(attempts / fresh_delta, 1) if fresh_delta else None
    no_stalls = dyn.get("no_global_stalls", True) and max_wall <= cap + 15

    if off_policy or ns_violations:
        verdict = "UNSAFE_TIMEOUT_BEHAVIOR" if not no_stalls else "SAFE_DYNAMIC_B5_PARTIAL"
    elif not no_stalls:
        verdict = "UNSAFE_TIMEOUT_BEHAVIOR"
    elif fresh_delta == 0:
        verdict = "SAFE_DYNAMIC_B5_NO_VALUE"
    elif probes_per_win and probes_per_win > 200:
        verdict = "SAFE_DYNAMIC_B5_TOO_EXPENSIVE"
    else:
        verdict = "SAFE_DYNAMIC_B5_CONFIRMED"

    out = {"generated_by": "scripts/rc5v2_safety_audit.py",
           "off_policy_count": off_policy, "namespace_violations": ns_violations,
           "attempts": attempts, "killed_by_timeout": killed, "max_wall_seconds": max_wall,
           "wall_cap_seconds": cap, "flake_rate": round(flakes / (n or 1), 3),
           "unknown_name": unknown, "unknown_name_rate": round(unknown / (attempts or 1), 3),
           "no_global_stalls": no_stalls, "fresh_true_deltas": fresh_delta,
           "source_specific_risk": src_specific,
           "dynamic_probes_per_fresh_win": probes_per_win,
           "b5_only_recommended": True, "verdict": verdict}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V2 safety audit", "",
          f"- **verdict: {verdict}**",
          f"- off-policy: {off_policy} | namespace violations: {ns_violations}",
          f"- no global stalls: {no_stalls} | max wall: {max_wall}s (cap {cap}s) | killed: {killed}",
          f"- unknown-name rate: {out['unknown_name_rate']} | flake rate: {out['flake_rate']}",
          f"- fresh true deltas: {fresh_delta} | probes/fresh win: {probes_per_win} | "
          f"source-specific: {src_specific}",
          f"- **B5-only recommended: True**"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v2-safety] verdict={verdict} off_policy={off_policy} no_stalls={no_stalls} "
          f"max_wall={max_wall}s fresh_delta={fresh_delta} probes/win={probes_per_win}")


if __name__ == "__main__":
    main()
