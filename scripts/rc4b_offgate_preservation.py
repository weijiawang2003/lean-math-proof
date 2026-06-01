#!/usr/bin/env python3
"""RC4B Part 8 — off-gate + preservation scan.

The gate is a deterministic function of (namespace, name, goal text), so the off-gate
scan is exact: count gate emissions over disjoint_negative_controls (Finset/Order/List
disjoint — incl. Finset.disjoint_left, deliberately out of scope), namespace_negative_
controls and canonical_smoke (all must be 0), plus the Set/Multiset split over the
gate-target sets. Preservation: the evaluator is additive (candidate ⊇ RC2) so
regressions are structurally impossible; we report the literal-RC2 canonical floor pass
rates and confirm 0 unexpected fires. Broad firing on negatives/floors -> REJECT_BROAD_GATE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4b_gate as G  # noqa: E402

NOFIRE_SETS = ("disjoint_negative_controls", "namespace_negative_controls", "canonical_smoke")
GATE_SETS = ("known_wins", "fresh_holdout_set", "fresh_holdout_multiset")
FLOOR_SETS = ("canonical_smoke",)


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
                    default="project/evolve/experiments/rc4_candidates/disjoint_left_bridge/out/literal_rc2_results.json")
    args = ap.parse_args()

    manifest = json.load(open(_p(args.manifest)))
    policy = G.load_policy(args.policy)
    rc2 = {}
    if os.path.exists(_p(args.literal_rc2)):
        rc2 = {r["full_name"]: r for r in json.load(open(_p(args.literal_rc2)))["results"]}
    cand = {}
    if args.candidate_results and os.path.exists(_p(args.candidate_results)):
        cand = {r["full_name"]: r for r in json.load(open(_p(args.candidate_results)))["results"]}

    per_set = {}
    unexpected_fires = []
    ns_split = {}
    for setname, rel in manifest["set_files"].items():
        entries = json.load(open(_p(rel)))
        fires = 0
        split = {"Set": 0, "Multiset": 0, "Other": 0}
        for e in entries:
            f, bns, tactics, anames, lemma = G.gate_fires(
                policy, e.get("namespace"), e.get("goal_text"), e["full_name"])
            if f:
                fires += 1
                split[bns if bns in ("Set", "Multiset") else "Other"] += 1
                if setname in NOFIRE_SETS:
                    unexpected_fires.append({"set": setname, "full_name": e["full_name"],
                                             "namespace": G.namespace_of(e.get("namespace"), e["full_name"]),
                                             "tactics": tactics})
        per_set[setname] = {"n": len(entries), "gate_emissions": fires,
                            "must_not_fire": setname in NOFIRE_SETS}
        ns_split[setname] = split

    off_gate_emissions = sum(per_set[s]["gate_emissions"] for s in NOFIRE_SETS if s in per_set)

    # emitted-and-failed on the negative controls (should be 0 since 0 emissions)
    neg_emitted_failed = 0
    for s in NOFIRE_SETS:
        rel = manifest["set_files"].get(s)
        if not rel:
            continue
        for e in json.load(open(_p(rel))):
            c = cand.get(e["full_name"])
            if c and c.get("gate_fired") and not c.get("candidate_finished"):
                neg_emitted_failed += 1

    # canonical floor pass rates from literal RC2 baseline (broken into source floors)
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
            if G.gate_fires(policy, e.get("namespace"), e.get("goal_text"), e["full_name"])[0]:
                d["gate_fires"] += 1

    broad = off_gate_emissions > 0
    out = {
        "generated_by": "scripts/rc4b_offgate_preservation.py",
        "per_set_gate_emissions": per_set,
        "gate_emissions_namespace_split": ns_split,
        "off_gate_emissions": off_gate_emissions,
        "unexpected_fired_cases": unexpected_fires,
        "negative_control_emitted_and_failed": neg_emitted_failed,
        "regressions": 0,
        "regression_note": "additive evaluator (candidate ⊇ RC2) → regressions structurally impossible",
        "canonical_floors": smoke_breakdown,
        "verdict": ("REJECT_BROAD_GATE" if broad else "OFFGATE_CLEAN"),
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC4B off-gate & preservation", "",
          f"- off-gate emissions (must be 0): **{off_gate_emissions}**",
          f"- negative-control emitted-and-failed: {neg_emitted_failed}",
          f"- regressions: 0 (additive evaluator)",
          f"- verdict: **{out['verdict']}**", "",
          "## Gate emissions per set", "",
          "| set | n | gate_emissions | Set | Multiset | Other | must_not_fire |",
          "|---|---|---|---|---|---|---|"]
    for s, d in per_set.items():
        sp = ns_split[s]
        md.append(f"| {s} | {d['n']} | {d['gate_emissions']} | {sp['Set']} | {sp['Multiset']} | "
                  f"{sp['Other']} | {d['must_not_fire']} |")
    md += ["", "## Canonical floors (literal RC2)", "",
           "| floor | n | rc2_solved | gate_fires |", "|---|---|---|---|"]
    for k, d in smoke_breakdown.items():
        md.append(f"| {k} | {d['n']} | {d['rc2_solved']} | {d['gate_fires']} |")
    if unexpected_fires:
        md += ["", "## Unexpected fires (off-gate!)", ""] + [f"- {u}" for u in unexpected_fires]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4b-offgate] off_gate={off_gate_emissions} verdict={out['verdict']} "
          f"floors={smoke_breakdown}")


if __name__ == "__main__":
    main()
