#!/usr/bin/env python3
"""TR7 Part 9 — RC5 design recommendations.

Synthesizes all TR7 analyses into a ranked set of RC5 recommendations
(RC5_STATIC_ALLOWLIST_EXPANSION / RC5_DYNAMIC_RETRIEVAL_WRAPPER / RC5_HYBRID_STATIC_PLUS_RANKER /
TR8_MORE_FRONTIER_DATA / ORDER_STRUCTURAL_BATTERY), each with rationale / expected benefit /
risk / required validation / next-task name. Recommendation only — implements nothing.
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
    ap.add_argument("--distribution", required=True)
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--replay")
    ap.add_argument("--allowlist", required=True)
    ap.add_argument("--gate-analysis", required=True)
    ap.add_argument("--dynamic-static", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    dist = json.load(open(_p(args.distribution)))
    cov = json.load(open(_p(args.coverage)))["summary"]
    allow = json.load(open(_p(args.allowlist)))
    gate = json.load(open(_p(args.gate_analysis)))
    dynstat = json.load(open(_p(args.dynamic_static)))

    pct_static = dynstat["pct_static_compatible"]
    pct_dyn = dynstat["pct_dynamic_only"]
    direction = dynstat["recommended_rc5_direction"]
    add_lemmas = allow["add_to_static_allowlist"]
    need_more = allow["need_more_evidence"]
    rc4a_tighten = gate["rc4a_broad_gate"]["needs_narrowing"]

    recs = []
    # primary
    if direction == "hybrid RC5":
        recs.append({
            "recommendation": "RC5_HYBRID_STATIC_PLUS_RANKER", "priority": "primary",
            "rationale": (f"{pct_static:.0%} of TR6 fresh wins are static-compatible (already RC4 "
                          f"or fixable by allowlist/gate/schema work) but {pct_dyn:.0%} require "
                          "theorem-specific retrieved lemmas (tauto/rw/exact families) that cannot "
                          "become a small static allowlist. RC4 static is safe and reproduces the "
                          "validated component wins; the fresh out-of-sample delta needs the TR6 "
                          "ranker-guided dynamic retrieval that found per-theorem lemmas. Keep RC4 "
                          "static as the deterministic core and add a gated ranker-guided dynamic "
                          "retrieval stage for the dynamic tail."),
            "expected_benefit": "recovers the fresh-frontier generalization RC4 static lacks "
                                "without losing RC4's safety/determinism on the static core.",
            "risk": "dynamic retrieval reintroduces nondeterminism + probe cost; must be gated and "
                    "owner-billed; ranker is namespace-specific (TR4/TR6 caveat).",
            "required_validation": "RC5 hybrid benchmark = RC4 static floors/known-wins preserved "
                                   "+ ranker-guided dynamic stage measured on a FRESH frontier with "
                                   "SX4-style attribution; determinism scoped to the static core.",
            "next_task": "RC5H — Hybrid static+ranker wrapper prototype & fresh-frontier benchmark",
        })
    elif pct_static >= 0.8:
        recs.append({"recommendation": "RC5_STATIC_ALLOWLIST_EXPANSION", "priority": "primary",
                     "rationale": f"{pct_static:.0%} static-compatible, dynamic tail small.",
                     "expected_benefit": "captures most TR6 wins statically.",
                     "risk": "allowlist creep.", "required_validation": "per-lemma literal validation.",
                     "next_task": "RC5S — static allowlist expansion validation"})
    else:
        recs.append({"recommendation": "RC5_DYNAMIC_RETRIEVAL_WRAPPER", "priority": "primary",
                     "rationale": f"{pct_dyn:.0%} dynamic-only dominates.",
                     "expected_benefit": "captures theorem-specific wins.",
                     "risk": "nondeterminism + cost.", "required_validation": "fresh benchmark + attribution.",
                     "next_task": "RC5D — dynamic retrieval wrapper"})

    # secondary: gate refinement
    if rc4a_tighten:
        recs.append({
            "recommendation": "RC4A_TIGHTEN_MONO_GATE (gate refinement, not a release)",
            "priority": "secondary",
            "rationale": (f"RC4A def-unfold gate fires {gate['rc4a_broad_gate']['fired']}× and "
                          f"closes only {gate['rc4a_broad_gate']['closed']} (precision "
                          f"{gate['rc4a_broad_gate']['precision']}); it fires on every "
                          "monotone/antitone theorem. Tighten to the iff-unfold shape (require "
                          "`_iff_` in the name) to cut wasted emissions. Additive/safe today, so "
                          "low urgency, but it is the loosest component."),
            "expected_benefit": "less wasted probe budget; cleaner precision before any expansion.",
            "risk": "tightening could drop a future iff-unfold win — validate additively.",
            "required_validation": "re-run RC4A external-additive eval with the tightened gate; "
                                   "confirm 0 lost wins, lower fire count.",
            "next_task": "folded into RC5H or a standalone RC4A-gate patch",
        })

    # secondary: more data for the need-more-evidence lemmas
    if need_more:
        recs.append({
            "recommendation": "TR8_MORE_FRONTIER_DATA", "priority": "secondary",
            "rationale": (f"{len(need_more)} TR6 wins use clean single-occurrence lemmas "
                          f"({need_more}) that are allowlist-expansion candidates but lack repeat / "
                          "namespace-parametric evidence. A larger fresh sweep would tell whether "
                          "they recur (→ static) or stay one-off (→ dynamic)."),
            "expected_benefit": "resolves the allowlist-expansion-vs-dynamic question for the tail.",
            "risk": "more compute; may still be inconclusive.",
            "required_validation": "TR6-style ranked live sweep over a larger fresh pool, count "
                                   "recurrence of these lemmas.",
            "next_task": "TR8 — larger fresh-frontier sweep focused on the candidate lemmas",
        })

    # note on the cohort artifact
    cohort_note = (f"The headline 'RC4R 0 fresh delta' is substantially a SELECTION ARTIFACT: "
                   f"{dist['tr6_fresh_win_in_rc4r_fresh']}/18 TR6 wins are in the RC4R fresh "
                   "frontier (it excluded all RC4D-used theorems, which is where 14/18 TR6 wins "
                   "went), and the fresh set over-samples the loose RC4A gate. RC4 actually covers "
                   f"{cov['would_rc4_cover']}/18 TR6 wins as its known wins.")

    out = {
        "generated_by": "scripts/tr7_rc5_recommendations.py",
        "primary_recommendation": recs[0]["recommendation"],
        "recommendations": recs,
        "cohort_artifact_note": cohort_note,
        "static_compatible_pct": pct_static, "dynamic_only_pct": pct_dyn,
        "distribution_verdict": dist["verdict"],
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR7 RC5 recommendations", "",
          f"- **primary: {recs[0]['recommendation']}**",
          f"- static-compatible {pct_static:.0%} / dynamic-only {pct_dyn:.0%} / "
          f"distribution {dist['verdict']}", "",
          f"> {cohort_note}", ""]
    for r in recs:
        md += [f"## {r['recommendation']} ({r['priority']})",
               f"- **rationale:** {r['rationale']}",
               f"- **expected benefit:** {r['expected_benefit']}",
               f"- **risk:** {r['risk']}",
               f"- **required validation:** {r['required_validation']}",
               f"- **next task:** {r['next_task']}", ""]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr7-rc5] primary={recs[0]['recommendation']} | static {pct_static} dyn {pct_dyn}")
    print(f"[tr7-rc5] secondaries={[r['recommendation'] for r in recs[1:]]}")


if __name__ == "__main__":
    main()
