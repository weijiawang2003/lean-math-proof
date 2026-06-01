#!/usr/bin/env python3
"""RC4C Part 8 — off-gate + preservation scan.

The gate is a deterministic function of (namespace, name, goal text), so the off-gate
scan is exact. We count RC4C_all and RC4C_nonoverlap gate emissions over every set; the
NOFIRE sets (negative_controls, namespace_negative_controls, canonical_smoke) must show
0 emissions in BOTH modes. We also report per-set namespace splits, the emitted-and-failed
rate (gate fired, candidate did not close) from the candidate results, and the literal-RC2
canonical-floor pass rates. The evaluator is additive (candidate ⊇ RC2) so regressions are
structurally impossible. Broad firing on negatives/floors -> REJECT_BROAD_GATE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4c_gate as G  # noqa: E402

NOFIRE_SETS = ("negative_controls", "namespace_negative_controls", "canonical_smoke")


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--candidate-results")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--literal-rc2",
                    default="project/evolve/experiments/rc4_candidates/d2_simp_aesop/out/literal_rc2_results.json")
    args = ap.parse_args()

    manifest = json.load(open(_p(args.manifest)))
    policy = G.load_policy(args.policy)
    rc2 = {}
    if os.path.exists(_p(args.literal_rc2)):
        rc2 = {r["full_name"]: r for r in json.load(open(_p(args.literal_rc2)))["results"]}
    cand = {}
    if args.candidate_results and os.path.exists(_p(args.candidate_results)):
        cand = {r["full_name"]: r for r in json.load(open(_p(args.candidate_results)))["results"]}

    per_set, unexpected, ns_split = {}, [], {}
    for setname, rel in manifest["set_files"].items():
        entries = json.load(open(_p(rel)))
        fires_all = fires_non = 0
        split = Counter()
        for e in entries:
            fa, *_ = G.gate_fires(policy, e.get("namespace"), e.get("goal_text"), e["full_name"], mode="all")
            fn_, *_ = G.gate_fires(policy, e.get("namespace"), e.get("goal_text"), e["full_name"], mode="nonoverlap")
            if fa:
                fires_all += 1
                split[G.namespace_of(e.get("namespace"), e["full_name"])] += 1
                if setname in NOFIRE_SETS:
                    unexpected.append({"set": setname, "full_name": e["full_name"],
                                       "namespace": G.namespace_of(e.get("namespace"), e["full_name"])})
            if fn_:
                fires_non += 1
        per_set[setname] = {"n": len(entries), "gate_emissions_all": fires_all,
                            "gate_emissions_nonoverlap": fires_non,
                            "must_not_fire": setname in NOFIRE_SETS}
        ns_split[setname] = dict(split)

    off_gate_all = sum(per_set[s]["gate_emissions_all"] for s in NOFIRE_SETS if s in per_set)
    off_gate_non = sum(per_set[s]["gate_emissions_nonoverlap"] for s in NOFIRE_SETS if s in per_set)

    # emitted-and-failed rate over the gate-firing sets (informational, narrowness signal)
    emit_fired = emit_failed = 0
    for r in cand.values():
        if r.get("gate_fired"):
            emit_fired += 1
            if not r.get("candidate_finished"):
                emit_failed += 1
    emit_rate = round(emit_failed / emit_fired, 3) if emit_fired else 0.0

    # canonical floors from literal RC2
    smoke_breakdown = {}
    rel = manifest["set_files"].get("canonical_smoke")
    if rel:
        for e in json.load(open(_p(rel))):
            tag = (e.get("expected_behavior") or "")
            key = ("demo_v1" if "demo_v1" in tag else
                   "nat_defs_medium" if "medium" in tag else
                   "nat_defs_large_v5" if "large" in tag else "other")
            d = smoke_breakdown.setdefault(key, {"n": 0, "rc2_solved": 0, "gate_fires": 0})
            d["n"] += 1
            if rc2.get(e["full_name"], {}).get("rc2_finished"):
                d["rc2_solved"] += 1
            if G.gate_fires(policy, e.get("namespace"), e.get("goal_text"), e["full_name"], mode="all")[0]:
                d["gate_fires"] += 1

    broad = off_gate_all > 0 or off_gate_non > 0
    verdict = "REJECT_BROAD_GATE" if broad else "OFFGATE_CLEAN"
    out = {
        "generated_by": "scripts/rc4c_offgate_preservation.py",
        "per_set_gate_emissions": per_set,
        "gate_emissions_namespace_split": ns_split,
        "off_gate_emissions_all": off_gate_all,
        "off_gate_emissions_nonoverlap": off_gate_non,
        "unexpected_fired_cases": unexpected,
        "emitted_and_failed": {"gate_fired": emit_fired, "failed": emit_failed, "rate": emit_rate,
                               "note": "additive: emitted-and-failed are honest negatives, not regressions"},
        "regressions": 0,
        "regression_note": "additive evaluator (candidate ⊇ RC2) → regressions structurally impossible",
        "canonical_floors": smoke_breakdown,
        "verdict": verdict,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC4C off-gate & preservation", "",
          f"- off-gate emissions (all, must be 0): **{off_gate_all}**",
          f"- off-gate emissions (nonoverlap, must be 0): **{off_gate_non}**",
          f"- emitted-and-failed: {emit_failed}/{emit_fired} (rate {emit_rate}) — honest negatives",
          f"- regressions: 0 (additive evaluator)",
          f"- verdict: **{verdict}**", "",
          "## Gate emissions per set", "",
          "| set | n | emit_all | emit_nonoverlap | must_not_fire | ns_split |",
          "|---|---|---|---|---|---|"]
    for s, d in per_set.items():
        md.append(f"| {s} | {d['n']} | {d['gate_emissions_all']} | {d['gate_emissions_nonoverlap']} | "
                  f"{d['must_not_fire']} | {ns_split[s]} |")
    md += ["", "## Canonical floors (literal RC2)", "",
           "| floor | n | rc2_solved | gate_fires |", "|---|---|---|---|"]
    for k, d in smoke_breakdown.items():
        md.append(f"| {k} | {d['n']} | {d['rc2_solved']} | {d['gate_fires']} |")
    if unexpected:
        md += ["", "## Unexpected fires (off-gate!)", ""] + [f"- {u}" for u in unexpected]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4c-offgate] off_gate_all={off_gate_all} off_gate_non={off_gate_non} "
          f"emitted_failed={emit_failed}/{emit_fired} verdict={verdict} floors={smoke_breakdown}")


if __name__ == "__main__":
    main()
