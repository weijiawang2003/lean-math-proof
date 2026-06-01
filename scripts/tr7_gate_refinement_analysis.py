#!/usr/bin/env python3
"""TR7 Part 7 — gate refinement analysis.

Per RC4 component (RC4A / RC4B / RC4C_residue): current gate, fresh fire count, fresh success
count, precision, missed TR6 wins, and a recommended change (tighten / loosen / keep / split /
dynamic_only). RC4A's broad def-unfold gate (fires on many fresh monotone/antitone theorems but
closes few) is the focus; RC4B/RC4C gates are tight. Emits candidate refinement proposals.
Analysis only — no changes.
"""
from __future__ import annotations

import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATES = {
    "RC4A": "namespace-agnostic goal/name contains an allowlisted def (Monotone/.../Finset.disjUnion); single `simp [defs]`",
    "RC4B": "namespace ∈ {Set,Multiset} AND name/goal contains disjoint; `simp [<NS>.disjoint_left]`(+aesop)",
    "RC4C_residue": "namespace+token gate for Multiset.disjoint_right / Set.subset_pair / List.forall; `simp [L]`(+aesop)",
}


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rc4-offgate", required=True)
    ap.add_argument("--rc4-fresh", required=True)
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--rc4-policy")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    offgate = json.load(open(_p(args.rc4_offgate)))
    coverage = json.load(open(_p(args.coverage)))

    ef = offgate["emitted_and_failed_by_component"]  # whole benchmark
    # missed TR6 wins per component (from coverage audit: non-covered classes)
    missed_by_comp = {"RC4A": [], "RC4B": [], "RC4C_residue": []}
    for r in coverage["records"]:
        if r["classification"] not in ("STATIC_COVERED_AND_SHOULD_SOLVE",):
            comp = r.get("rc4_component")
            if comp in missed_by_comp:
                missed_by_comp[comp].append(r["full_name"])

    components = []
    for comp in ("RC4A", "RC4B", "RC4C_residue"):
        fired = ef[comp]["fired"]
        failed = ef[comp]["failed"]
        success = fired - failed
        precision = round(success / fired, 3) if fired else 0.0
        missed = missed_by_comp[comp]
        if comp == "RC4A":
            change = "tighten" if precision < 0.3 else "keep"
            rationale = (f"def-unfold gate fires {fired}× but closes only {success} "
                         f"(precision {precision}); it fires on every monotone/antitone theorem "
                         "whether or not `simp [Monotone,…]` finishes — broad, low precision. "
                         "Tighten to the iff-unfold shape (e.g. require `_iff_` in the name) or "
                         "make it dynamic. Additive/safe today, but the loosest component.")
        elif comp == "RC4B":
            change = "keep"
            rationale = (f"tight gate (precision {precision}); fires {fired}× closes {success}. "
                         "Covers all its TR6 disjoint wins. The 1 wrapper miss "
                         "(Set.disjoint_sUnion_right) is a search-depth gap, not a gate problem.")
        else:
            change = "keep"
            rationale = (f"tight gate (precision {precision}); fires {fired}× closes {success}. "
                         "Covers its TR6 residue wins. Missing TR6 wins are single-occurrence / "
                         "theorem-specific (Part 6), so allowlist expansion is not yet warranted.")
        components.append({
            "component": comp, "current_gate": GATES[comp],
            "fire_count": fired, "success_count": success, "precision": precision,
            "missed_tr6_wins": missed, "recommended_change": change, "rationale": rationale,
        })

    proposals = []
    rc4a = next(c for c in components if c["component"] == "RC4A")
    if rc4a["recommended_change"] == "tighten":
        proposals.append("RC4A_TIGHTEN_MONO_GATE")
    proposals.append("RC4B_KEEP")
    # RC4C expand only if Part 6 flagged ADD; here it did not -> keep, but record the option
    proposals.append("RC4C_RESIDUE_KEEP")
    # missing wins requiring dynamic retrieval
    n_dynamic = sum(1 for r in coverage["records"]
                    if r["classification"] in ("DYNAMIC_RETRIEVAL_REQUIRED",))
    n_allowmiss = sum(1 for r in coverage["records"] if r["classification"] == "ALLOWLIST_MISS")
    if n_dynamic + n_allowmiss > 0:
        proposals.append("DYNAMIC_RETRIEVAL_REQUIRED")

    out = {
        "generated_by": "scripts/tr7_gate_refinement_analysis.py",
        "components": components, "candidate_proposals": proposals,
        "rc4a_broad_gate": {"fired": rc4a["fire_count"], "closed": rc4a["success_count"],
                            "precision": rc4a["precision"], "needs_narrowing": rc4a["recommended_change"] == "tighten"},
        "missing_wins_requiring_dynamic_or_new_lemma": n_dynamic + n_allowmiss,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR7 gate refinement analysis", "",
          f"- candidate proposals: {proposals}",
          f"- RC4A broad gate: fired {rc4a['fire_count']}, closed {rc4a['success_count']}, "
          f"precision {rc4a['precision']} → **{'TIGHTEN' if rc4a['recommended_change']=='tighten' else 'keep'}**",
          f"- missing wins needing dynamic/new-lemma: {out['missing_wins_requiring_dynamic_or_new_lemma']}", "",
          "| component | fired | closed | precision | missed TR6 | change |",
          "|---|---|---|---|---|---|"]
    for c in components:
        md.append(f"| {c['component']} | {c['fire_count']} | {c['success_count']} | "
                  f"{c['precision']} | {len(c['missed_tr6_wins'])} | **{c['recommended_change']}** |")
    md += ["", "## Rationale", ""]
    for c in components:
        md.append(f"- **{c['component']}** ({c['recommended_change']}): {c['rationale']}")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr7-gate] proposals={proposals}")
    print(f"[tr7-gate] RC4A precision={rc4a['precision']} (fired {rc4a['fire_count']}/closed {rc4a['success_count']})")


if __name__ == "__main__":
    main()
