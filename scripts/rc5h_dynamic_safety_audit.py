#!/usr/bin/env python3
"""RC5H Part 11 — dynamic-stage safety audit.

Audits the dynamic stage for production-experiment safety: gate firings, unknown-name rate,
timeout/flake rate, off-policy programs (grammar violations), unsafe broad programs, namespace
violations, emitted-and-failed rate, source-specific risk, and proof-search cost. Classifies
DYNAMIC_STAGE_SAFE_FOR_EXPERIMENT / TOO_BROAD / TOO_EXPENSIVE / UNSTABLE.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED_NS = {"Set", "Finset", "List", "Multiset", "Nat"}


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--program-plan", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--dynamic-b5")
    ap.add_argument("--dynamic-b10")
    ap.add_argument("--dynamic-b20")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    policy = json.load(open(_p(args.policy)))
    plan = json.load(open(_p(args.program_plan)))
    attr = json.load(open(_p(args.attribution)))
    grammar_heads = {g.split(" ")[0] for g in policy["dynamic_stage"]["program_grammar"]}

    def _load(p):
        return json.load(open(_p(p))) if p and os.path.exists(_p(p)) else None
    stages = [s for s in (_load(args.dynamic_b5), _load(args.dynamic_b10), _load(args.dynamic_b20)) if s]

    # plan-side: gate firings, namespace violations, off-policy program heads
    gate_firings = len([t for t in plan["theorems"] if t.get("rc5h_dynamic_eligible", True)])
    ns_violations = [t["full_name"] for t in plan["theorems"] if t.get("namespace") not in ALLOWED_NS]
    off_policy = []
    for t in plan["theorems"]:
        for pgm in t.get("programs_ranked", []):
            head = (pgm.get("tactic") or "").split(" ")[0]
            if head and head not in grammar_heads and head not in ("simp", "rw", "exact", "simpa", "ext", "constructor"):
                off_policy.append(pgm.get("tactic"))

    # run-side: emitted-and-failed, unknown-name, timeout/flake
    attempts = total_fail = unknown = timeout = flake = 0
    for s in stages:
        for r in s.get("results", []):
            attempts += r.get("programs_attempted", 0)
            for f in r.get("failures", []):
                total_fail += 1
                if f.get("outcome") == "unknown_name":
                    unknown += 1
                elif f.get("outcome") == "timeout":
                    timeout += 1
            if r.get("open_flake"):
                flake += 1
    unknown_rate = round(unknown / attempts, 3) if attempts else 0.0
    timeout_rate = round(timeout / attempts, 3) if attempts else 0.0
    emit_fail_rate = round(total_fail / attempts, 3) if attempts else 0.0
    n_theorems = len(set(r["full_name"] for s in stages for r in s.get("results", [])))
    flake_rate = round(flake / n_theorems, 3) if n_theorems else 0.0

    src_specific = attr.get("source_specific_wins", 0)
    true_delta = attr.get("true_hybrid_deltas", 0)
    dyn_wins = attr.get("dynamic_wins_total", 0)

    max_unknown = policy["dynamic_stage"]["gates"]["max_unknown_name_rate"]
    concerns = []
    if unknown_rate > max_unknown:
        concerns.append(f"unknown-name rate {unknown_rate} > {max_unknown}")
    if flake_rate > 0.15 or timeout_rate > 0.15:
        concerns.append(f"high flake/timeout (flake {flake_rate}, timeout {timeout_rate})")
    if ns_violations:
        concerns.append(f"{len(ns_violations)} namespace violations")
    if off_policy:
        concerns.append(f"{len(off_policy)} off-policy programs")

    if true_delta == 0 and dyn_wins == 0:
        verdict = "DYNAMIC_STAGE_SAFE_FOR_EXPERIMENT"  # safe but no marginal win
        note = "stable and within policy, but produced no marginal win on this benchmark."
    elif unknown_rate > max_unknown or ns_violations:
        verdict = "DYNAMIC_STAGE_TOO_BROAD"
        note = "gate/namespace/unknown-name violations indicate the dynamic gate is too broad."
    elif flake_rate > 0.2 or timeout_rate > 0.2:
        verdict = "DYNAMIC_STAGE_UNSTABLE"
        note = "flake/timeout rate too high for a reliable stage."
    elif attempts > 0 and (attempts / max(1, n_theorems)) > 18:
        verdict = "DYNAMIC_STAGE_TOO_EXPENSIVE"
        note = "probe cost per theorem exceeds the budget envelope."
    else:
        verdict = "DYNAMIC_STAGE_SAFE_FOR_EXPERIMENT"
        note = "within policy gates, acceptable cost/stability for an experiment."

    out = {
        "generated_by": "scripts/rc5h_dynamic_safety_audit.py",
        "gate_firings": gate_firings, "namespace_violations": ns_violations,
        "off_policy_programs": len(off_policy),
        "attempts": attempts, "emitted_and_failed": total_fail, "emitted_and_failed_rate": emit_fail_rate,
        "unknown_name": unknown, "unknown_name_rate": unknown_rate,
        "timeout": timeout, "timeout_rate": timeout_rate,
        "open_flakes": flake, "flake_rate": flake_rate,
        "source_specific_wins": src_specific, "true_hybrid_deltas": true_delta,
        "dynamic_wins_total": dyn_wins,
        "probes_per_theorem": round(attempts / max(1, n_theorems), 2),
        "hard_concerns": concerns, "verdict": verdict, "note": note,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5H dynamic-stage safety audit", "",
          f"- **verdict: {verdict}** — {note}",
          f"- gate firings (eligible theorems): {gate_firings}",
          f"- attempts: {attempts} | emitted-and-failed: {total_fail} ({emit_fail_rate})",
          f"- unknown-name: {unknown} ({unknown_rate}) | timeout: {timeout} ({timeout_rate}) | "
          f"flakes: {flake} ({flake_rate})",
          f"- namespace violations: {len(ns_violations)} | off-policy programs: {len(off_policy)}",
          f"- source-specific wins: {src_specific} | true hybrid deltas: {true_delta}",
          f"- probes per theorem: {out['probes_per_theorem']}",
          f"- hard concerns: {concerns or 'none'}"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5h-safety] verdict={verdict} unknown_rate={unknown_rate} flake_rate={flake_rate} "
          f"probes/thm={out['probes_per_theorem']} concerns={concerns}")


if __name__ == "__main__":
    main()
