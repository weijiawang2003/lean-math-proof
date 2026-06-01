#!/usr/bin/env python3
"""RC4R Part 7 — off-gate audit of the RC4 benchmark.

The RC4 ordered-union gate is a pure function of (namespace, name, goal), so we recompute it
over every benchmark entry and audit: which RC4 actions are emitted, whether each emission
satisfies its intended gate, any emission on the NOFIRE sets (negative_controls,
offgate_controls), the component responsible, and the emitted-and-failed count per component.
Requirement: 0 off-gate emissions.
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

POLICY = "project/evolve/experiments/rc4_candidates/composition_rc4d/rc4d_composition_policy.json"
NOFIRE = ("negative_controls", "offgate_controls")


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rc4-results", required=True)
    ap.add_argument("--rc4-wrapper")
    ap.add_argument("--policy", default=POLICY)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    manifest = json.load(open(_p(args.manifest)))
    policy = G.load_policy(args.policy)
    rc4 = {r["full_name"]: r for r in json.load(open(_p(args.rc4_results)))["results"]}

    per_set, unexpected, comp_emit = {}, [], Counter()
    emit_fired = Counter()
    emit_failed = Counter()
    all_emitted_actions = Counter()
    for setname, rel in manifest["set_files"].items():
        entries = json.load(open(_p(rel)))
        fires = 0
        for e in entries:
            goal = e.get("goal_text") or e.get("statement_text")
            f, em = G.gate_fires(policy, e.get("namespace"), goal, e["full_name"])
            if f:
                fires += 1
                comps = G.components_firing(em)
                for c in comps:
                    comp_emit[c] += 1
                for x in em:
                    all_emitted_actions[x["action"]] += 1
                # emitted-and-failed (from RC4 results)
                r = rc4.get(e["full_name"], {})
                for c in comps:
                    emit_fired[c] += 1
                    if r.get("status") != "solved":
                        emit_failed[c] += 1
                if setname in NOFIRE:
                    unexpected.append({"set": setname, "full_name": e["full_name"],
                                       "components": comps,
                                       "namespace": G.namespace_of(e.get("namespace"), e["full_name"])})
        per_set[setname] = {"n": len(entries), "gate_emissions": fires,
                            "must_not_fire": setname in NOFIRE}

    off_gate = sum(per_set[s]["gate_emissions"] for s in NOFIRE if s in per_set)
    emitted_failed = {c: {"fired": emit_fired[c], "failed": emit_failed[c],
                          "rate": round(emit_failed[c] / emit_fired[c], 3) if emit_fired[c] else 0.0}
                      for c in ("RC4A", "RC4B", "RC4C_residue")}
    broad_warnings = []
    for c, d in emitted_failed.items():
        if d["fired"] and d["rate"] >= 0.85:
            broad_warnings.append(f"{c} emitted-and-failed rate {d['rate']} (fired {d['fired']})")
    verdict = "OFFGATE_CLEAN" if off_gate == 0 else "OFFGATE_VIOLATION"

    out = {"generated_by": "scripts/rc4r_offgate_audit.py",
           "per_set_gate_emissions": per_set,
           "emitted_actions_histogram": dict(all_emitted_actions),
           "component_emission_counts": dict(comp_emit),
           "off_gate_emissions": off_gate, "unexpected_fired_cases": unexpected,
           "emitted_and_failed_by_component": emitted_failed,
           "broad_gate_warnings": broad_warnings,
           "requirement_0_offgate_met": off_gate == 0,
           "verdict": verdict}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC4 off-gate audit", "",
          f"- off-gate emissions (must be 0): **{off_gate}** → **{verdict}**",
          f"- component emission counts: {dict(comp_emit)}",
          f"- emitted-and-failed by component: {emitted_failed}",
          f"- broad-gate warnings: {broad_warnings or 'none'}", "",
          "## Per-set gate emissions", "",
          "| set | n | emissions | must_not_fire |", "|---|---|---|---|"]
    for s, d in per_set.items():
        md.append(f"| {s} | {d['n']} | {d['gate_emissions']} | {d['must_not_fire']} |")
    md += ["", "## Emitted actions histogram", ""]
    for a, n in all_emitted_actions.most_common():
        md.append(f"- {a}: {n}")
    if unexpected:
        md += ["", "## OFF-GATE VIOLATIONS", ""] + [f"- {u}" for u in unexpected]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4r-offgate] off_gate={off_gate} verdict={verdict} comp_emit={dict(comp_emit)}")
    print(f"[rc4r-offgate] emitted_failed={emitted_failed} warnings={broad_warnings}")


if __name__ == "__main__":
    main()
