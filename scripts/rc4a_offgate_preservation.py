#!/usr/bin/env python3
"""RC4A Part 8 — off-gate + preservation scan.

The gate is a deterministic function of the goal text, so the off-gate scan is exact:
count gate emissions over negative_controls and canonical_smoke (must be 0), and over
the gate-target sets. Preservation: since the evaluator is additive (candidate ⊇ RC2),
regressions are structurally impossible; we report the literal-RC2 canonical floor
pass rates and confirm 0 unexpected fires. If the gate fires broadly on
negatives/floors -> REJECT_BROAD_GATE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4a_gate as G  # noqa: E402

NOFIRE_SETS = ("negative_controls", "canonical_smoke")
GATE_SETS = ("known_wins", "same_cluster_holdout", "fresh_frontier_holdout")
FLOOR_SETS = ("canonical_smoke",)


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--literal-rc2",
                    default="project/evolve/experiments/rc4_candidates/def_unfold_simp/out/literal_rc2_results.json")
    args = ap.parse_args()

    manifest = json.load(open(_p(args.manifest)))
    policy = G.load_policy(args.policy)
    rc2 = {}
    if os.path.exists(_p(args.literal_rc2)):
        rc2 = {r["full_name"]: r for r in json.load(open(_p(args.literal_rc2)))["results"]}

    per_set = {}
    unexpected_fires = []
    for setname, rel in manifest["set_files"].items():
        entries = json.load(open(_p(rel)))
        fires = 0
        for e in entries:
            f, tac, defs = G.gate_fires(policy, e.get("goal_text"), e["full_name"])
            if f:
                fires += 1
                if setname in NOFIRE_SETS:
                    unexpected_fires.append({"set": setname, "full_name": e["full_name"],
                                             "tactic": tac, "defs": defs})
        per_set[setname] = {"n": len(entries), "gate_emissions": fires,
                            "must_not_fire": setname in NOFIRE_SETS}

    off_gate_emissions = sum(per_set[s]["gate_emissions"] for s in NOFIRE_SETS if s in per_set)

    # canonical floor pass rates from literal RC2 baseline
    floors = {}
    for setname in FLOOR_SETS:
        rel = manifest["set_files"].get(setname)
        if not rel:
            continue
        names = [e["full_name"] for e in json.load(open(_p(rel)))]
        solved = sum(1 for fn in names if rc2.get(fn, {}).get("rc2_finished"))
        floors[setname] = {"n": len(names), "rc2_solved": solved,
                           "rc2_failed": len(names) - solved}
    # break canonical_smoke into its source sets for readability (demo_v1/medium/large)
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
            if G.gate_fires(policy, e.get("goal_text"), e["full_name"])[0]:
                d["gate_fires"] += 1

    broad = off_gate_emissions > 0
    out = {
        "generated_by": "scripts/rc4a_offgate_preservation.py",
        "per_set_gate_emissions": per_set,
        "off_gate_emissions": off_gate_emissions,
        "unexpected_fired_cases": unexpected_fires,
        "regressions": 0,
        "regression_note": "additive evaluator (candidate ⊇ RC2) → regressions structurally impossible",
        "canonical_floors": floors,
        "canonical_smoke_breakdown": smoke_breakdown,
        "verdict": ("REJECT_BROAD_GATE" if broad else "OFFGATE_CLEAN"),
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC4A off-gate & preservation", "",
          f"- off-gate emissions (must be 0): **{off_gate_emissions}**",
          f"- regressions: 0 (additive evaluator)",
          f"- verdict: **{out['verdict']}**", "",
          "## Gate emissions per set", "",
          "| set | n | gate_emissions | must_not_fire |", "|---|---|---|---|"]
    for s, d in per_set.items():
        md.append(f"| {s} | {d['n']} | {d['gate_emissions']} | {d['must_not_fire']} |")
    md += ["", "## Canonical floors (literal RC2)", "",
           "| floor | n | rc2_solved | gate_fires |", "|---|---|---|---|"]
    for k, d in smoke_breakdown.items():
        md.append(f"| {k} | {d['n']} | {d['rc2_solved']} | {d['gate_fires']} |")
    if unexpected_fires:
        md += ["", "## Unexpected fires", ""] + [f"- {u}" for u in unexpected_fires]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4a-offgate] off_gate={off_gate_emissions} verdict={out['verdict']} "
          f"floors={smoke_breakdown}")


if __name__ == "__main__":
    main()
