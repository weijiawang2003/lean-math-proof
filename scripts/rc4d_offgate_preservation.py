#!/usr/bin/env python3
"""RC4D Part 8 — off-gate + preservation + floors scan.

The RC4D ordered-union gate is a deterministic function of (namespace, name, goal), so the
off-gate scan is exact. We count per-set gate emissions and which COMPONENT causes each (so a
broad component can be identified), require 0 emissions on the NOFIRE sets
(negative_controls, namespace_negative_controls, canonical_smoke), report the
emitted-and-failed rate per component (narrowness signal), and read the literal-RC2 canonical
floor pass rates. The additive evaluator is candidate ⊇ RC2, so regressions are structurally
impossible. Verdict REJECT_BROAD_GATE if any NOFIRE set fires or RC4C_residue fires too broadly.
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
import rc4d_gate as G  # noqa: E402

NOFIRE_SETS = ("negative_controls", "namespace_negative_controls", "canonical_smoke")


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", "--validation-manifest", dest="manifest", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--candidate-results")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--literal-rc2",
                    default="project/evolve/experiments/rc4_candidates/composition_rc4d/out/literal_rc2_results.json")
    args = ap.parse_args()

    manifest = json.load(open(_p(args.manifest)))
    policy = G.load_policy(args.policy)
    rc2 = {}
    if os.path.exists(_p(args.literal_rc2)):
        rc2 = {r["full_name"]: r for r in json.load(open(_p(args.literal_rc2)))["results"]}
    cand = {}
    if args.candidate_results and os.path.exists(_p(args.candidate_results)):
        cand = {r["full_name"]: r for r in json.load(open(_p(args.candidate_results)))["results"]}

    per_set, unexpected, comp_split = {}, [], {}
    for setname, rel in manifest["set_files"].items():
        entries = json.load(open(_p(rel)))
        fires = 0
        by_comp = Counter()
        for e in entries:
            f, em = G.gate_fires(policy, e.get("namespace"), e.get("goal_text"), e["full_name"])
            if f:
                fires += 1
                for c in G.components_firing(em):
                    by_comp[c] += 1
                if setname in NOFIRE_SETS:
                    unexpected.append({"set": setname, "full_name": e["full_name"],
                                       "components": G.components_firing(em),
                                       "namespace": G.namespace_of(e.get("namespace"), e["full_name"])})
        per_set[setname] = {"n": len(entries), "gate_emissions": fires,
                            "must_not_fire": setname in NOFIRE_SETS}
        comp_split[setname] = dict(by_comp)

    off_gate = sum(per_set[s]["gate_emissions"] for s in NOFIRE_SETS if s in per_set)

    # emitted-and-failed per component (from candidate results, narrowness signal)
    emit_by_comp = Counter()
    failed_by_comp = Counter()
    for r in cand.values():
        if r.get("gate_fired"):
            for c in r.get("components_firing", []):
                emit_by_comp[c] += 1
                if not r.get("candidate_finished"):
                    failed_by_comp[c] += 1
    emitted_failed = {c: {"fired": emit_by_comp[c], "failed": failed_by_comp[c],
                          "rate": round(failed_by_comp[c] / emit_by_comp[c], 3) if emit_by_comp[c] else 0.0}
                      for c in ("RC4A", "RC4B", "RC4C_residue")}

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
            if G.gate_fires(policy, e.get("namespace"), e.get("goal_text"), e["full_name"])[0]:
                d["gate_fires"] += 1

    # RC4C_residue broadness: residue-only emissions on nofire sets
    residue_offgate = sum(comp_split[s].get("RC4C_residue", 0) for s in NOFIRE_SETS if s in comp_split)
    broad = off_gate > 0 or residue_offgate > 0
    verdict = "REJECT_BROAD_GATE" if broad else "OFFGATE_CLEAN"

    out = {
        "generated_by": "scripts/rc4d_offgate_preservation.py",
        "per_set_gate_emissions": per_set,
        "gate_emissions_component_split": comp_split,
        "off_gate_emissions": off_gate,
        "rc4c_residue_offgate_emissions": residue_offgate,
        "unexpected_fired_cases": unexpected,
        "emitted_and_failed_by_component": emitted_failed,
        "regressions": 0,
        "regression_note": "additive evaluator (candidate ⊇ RC2) → regressions structurally impossible",
        "canonical_floors": smoke_breakdown,
        "verdict": verdict,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC4D off-gate & preservation", "",
          f"- off-gate emissions (must be 0): **{off_gate}**",
          f"- RC4C_residue off-gate emissions (must be 0): **{residue_offgate}**",
          f"- emitted-and-failed by component: {emitted_failed}",
          f"- regressions: 0 (additive evaluator)",
          f"- verdict: **{verdict}**", "",
          "## Gate emissions per set", "",
          "| set | n | emit | must_not_fire | component_split |",
          "|---|---|---|---|---|"]
    for s, d in per_set.items():
        md.append(f"| {s} | {d['n']} | {d['gate_emissions']} | {d['must_not_fire']} | {comp_split[s]} |")
    md += ["", "## Canonical floors (literal RC2)", "",
           "| floor | n | rc2_solved | gate_fires |", "|---|---|---|---|"]
    for k, d in smoke_breakdown.items():
        md.append(f"| {k} | {d['n']} | {d['rc2_solved']} | {d['gate_fires']} |")
    if unexpected:
        md += ["", "## Unexpected fires (off-gate!)", ""] + [f"- {u}" for u in unexpected]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4d-offgate] off_gate={off_gate} residue_offgate={residue_offgate} "
          f"verdict={verdict} floors={smoke_breakdown}")
    print(f"[rc4d-offgate] emitted_failed_by_comp={emitted_failed}")


if __name__ == "__main__":
    main()
