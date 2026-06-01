#!/usr/bin/env python3
"""RC5V3 Part 13 — safety audit of the safe dynamic stage across B1/B3/B5 at scale.

Aggregates off-policy / timeouts / killed / max wall / flake / unknown-name / namespace violations /
probes-per-win across the three budget result files; classifies SAFE_DYNAMIC_SCALING_CONFIRMED /
PARTIAL / TOO_EXPENSIVE / NO_VALUE / UNSAFE_TIMEOUT_BEHAVIOR.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc5s_grammar as G

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--dynamic-b1", required=True)
    ap.add_argument("--dynamic-b3", required=True)
    ap.add_argument("--dynamic-b5", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    policy = json.load(open(_p(args.policy)))
    plan = json.load(open(_p(args.plan)))
    attr = json.load(open(_p(args.attribution)))
    cap = policy["timeout_policy"]["per_theorem_wall_cap_seconds"]
    allowed_ns = set(policy["allowed_namespaces"])
    pol = {"allowed_namespaces": policy["allowed_namespaces"], "aesop_namespaces": policy["aesop_namespaces"]}

    # off-policy / namespace check over final plan (must be 0)
    off_policy = ns_violations = 0
    for t in plan["theorems"]:
        ns = t.get("namespace")
        if ns not in allowed_ns:
            ns_violations += 1
        for p in t.get("programs_ranked", []):
            if G.classify_program(p.get("tactic"), ns, pol)[1] is not True:
                off_policy += 1

    attempts = killed = unknown = flakes = 0
    max_wall = 0.0
    unbounded = []
    for path in (args.dynamic_b1, args.dynamic_b3, args.dynamic_b5):
        d = json.load(open(_p(path)))
        for r in d.get("results", []):
            attempts += r.get("programs_attempted", 0)
            if r.get("killed_by_timeout"):
                killed += 1
            unknown += sum(1 for f in r.get("failures", []) if f.get("outcome") == "unknown_name")
            if r.get("setup_error") and "exceeded" in (r.get("setup_error") or ""):
                flakes += 1
            w = r.get("wall_seconds", 0)
            max_wall = max(max_wall, w)
            if w > cap + 15 and not r.get("killed_by_timeout"):
                unbounded.append(r["full_name"])

    fresh_delta = attr.get("fresh_true_deltas", 0)
    src_specific = attr.get("source_specific", 0)
    probes_per_win = round(attempts / fresh_delta, 1) if fresh_delta else None
    no_stalls = len(unbounded) == 0 and max_wall <= cap + 15

    if off_policy or ns_violations or not no_stalls:
        verdict = "UNSAFE_TIMEOUT_BEHAVIOR"
    elif fresh_delta == 0:
        verdict = "SAFE_DYNAMIC_NO_VALUE"
    elif probes_per_win and probes_per_win > 250:
        verdict = "SAFE_DYNAMIC_TOO_EXPENSIVE"
    elif fresh_delta < 3:
        verdict = "SAFE_DYNAMIC_SCALING_PARTIAL"
    else:
        verdict = "SAFE_DYNAMIC_SCALING_CONFIRMED"

    out = {"generated_by": "scripts/rc5v3_safety_audit.py",
           "off_policy_count": off_policy, "namespace_violations": ns_violations,
           "dynamic_probes": attempts, "killed_by_timeout": killed,
           "max_wall_seconds": round(max_wall, 1), "wall_cap_seconds": cap,
           "flake_count": flakes, "unknown_name": unknown,
           "unknown_name_rate": round(unknown / (attempts or 1), 3),
           "no_global_stalls": no_stalls, "global_stalls_unbounded": unbounded,
           "fresh_true_deltas": fresh_delta, "source_specific_risk": src_specific,
           "dynamic_probes_per_fresh_win": probes_per_win,
           "b5_remains_safe": no_stalls and off_policy == 0,
           "verdict": verdict}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V3 safety audit", "",
          f"- **verdict: {verdict}**",
          f"- off-policy: {off_policy} | namespace violations: {ns_violations}",
          f"- no global stalls: {no_stalls} | max wall: {round(max_wall,1)}s (cap {cap}s) | killed: {killed}",
          f"- dynamic probes: {attempts} | unknown-name rate: {out['unknown_name_rate']} | flakes: {flakes}",
          f"- fresh true deltas: {fresh_delta} | probes/fresh win: {probes_per_win} | "
          f"source-specific: {src_specific}",
          f"- **B5 remains safe: {out['b5_remains_safe']}**"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v3-safety] verdict={verdict} off_policy={off_policy} no_stalls={no_stalls} "
          f"max_wall={round(max_wall,1)}s killed={killed} fresh_delta={fresh_delta} probes/win={probes_per_win}")


if __name__ == "__main__":
    main()
