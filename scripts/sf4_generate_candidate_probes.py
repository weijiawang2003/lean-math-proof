#!/usr/bin/env python3
"""SF4 Part 5 — generate candidate probes from failure clusters.

Conservative, cluster-driven probe families. NO source-specific rw bridges (those
would be marked SOURCE_SPECIFIC_DIAGNOSTIC and never credited). Each probe carries
a narrow gate (namespace + name-feature), a risk level, and promotion_allowed=false.
Pure analysis (no Lean).
"""
from __future__ import annotations

import argparse
import json
import os

# family -> list of (tactic_or_sequence, risk)
FAMILY_PROBES = {
    "set_ite_simp_aesop": [("simp [Set.ite] <;> aesop", "low"),
                           ("simp [Set.ite, Set.ext_iff] <;> aesop", "medium")],
    "set_ite_ext": [("ext x <;> simp [Set.ite]", "low"),
                    ("ext x <;> by_cases h : x ∈ _ <;> simp_all [Set.ite]", "high")],
    "set_ext_aesop": [("ext x <;> aesop", "low"), ("ext x <;> simp_all", "medium")],
    "set_ext_simp": [("ext x <;> simp", "low")],
    "set_iff_constructor_aesop": [("constructor <;> intro h <;> aesop", "medium"),
                                  ("constructor <;> intro h <;> simp_all", "medium")],
    "set_subset_antisymm": [("apply Set.Subset.antisymm <;> intro x <;> aesop", "high"),
                            ("apply Set.Subset.antisymm <;> intro x <;> simp_all", "high")],
    "multiset_tofinset_simp_aesop": [("simp [Multiset.mem_toFinset] <;> aesop", "medium"),
                                     ("simp [Finset.mem_coe, Multiset.mem_toFinset] <;> aesop", "medium")],
    "arith_omega": [("omega", "low")],
    "arith_nlinarith": [("nlinarith", "high"), ("omega <;> simp_all", "high")],
    "generic_aesop_simpall": [("aesop", "low"), ("simp_all", "low")],
}

# gate name-feature per family (for off-gate discipline)
FAMILY_GATE_FEATURE = {
    "set_ite_simp_aesop": ["ite", "dite", ".if"],
    "set_ite_ext": ["ite", "dite"],
    "set_ext_aesop": ["inter", "union", "diff", "compl", "singleton", "pair", "powerset"],
    "set_ext_simp": ["inter", "union", "diff", "compl"],
    "set_iff_constructor_aesop": ["_iff", "iff_"],
    "set_subset_antisymm": ["subset", "ssubset"],
    "multiset_tofinset_simp_aesop": ["toFinset"],
    "arith_omega": [], "arith_nlinarith": [],
    "generic_aesop_simpall": [],
}
FAMILY_NS = {
    "set_ite_simp_aesop": ["Set"], "set_ite_ext": ["Set"], "set_ext_aesop": ["Set"],
    "set_ext_simp": ["Set"], "set_iff_constructor_aesop": ["Set"], "set_subset_antisymm": ["Set"],
    "multiset_tofinset_simp_aesop": ["Multiset", "Finset"],
    "arith_omega": ["Nat", "Int"], "arith_nlinarith": ["Nat", "Int"],
    "generic_aesop_simpall": [],
}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args(argv)

    clusters = json.load(open(args.clusters))["clusters"]
    probes, seen = [], set()
    for c in clusters:
        for fam in c["candidate_families"]:
            for tac, risk in FAMILY_PROBES.get(fam, []):
                key = (fam, tac)
                if key in seen:
                    # attach this cluster id too
                    for pr in probes:
                        if pr["family"] == fam and pr["tactic_or_sequence"] == tac:
                            if c["cluster_id"] not in pr["cluster_id"]:
                                pr["cluster_id"].append(c["cluster_id"])
                    continue
                seen.add(key)
                probes.append({
                    "family": fam,
                    "tactic_or_sequence": tac,
                    "is_sequence": "<;>" in tac,
                    "cluster_id": [c["cluster_id"]],
                    "gate": {"namespaces": FAMILY_NS.get(fam, []),
                             "name_features": FAMILY_GATE_FEATURE.get(fam, []),
                             "max_emissions_per_theorem": 1},
                    "risk": risk,
                    "source": "cluster_generated",
                    "source_specific": False,
                    "promotion_allowed": False,
                })

    out = {"clusters_input": args.clusters, "num_probes": len(probes),
           "families": sorted({p["family"] for p in probes}),
           "warning": "All probes generic/cluster-driven. No SOURCE_SPECIFIC rw bridges. "
                      "High-risk broad tactics (subset_antisymm, nlinarith, generic) flagged; "
                      "credit requires SX4 TRUE_DELTA + 0 off-gate.",
           "probes": probes}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# SF4 candidate probes", "",
         f"- probes generated: **{len(probes)}**",
         f"- families: {out['families']}", "",
         "| family | tactic/sequence | seq? | risk | gate ns | gate features |",
         "|---|---|---|---|---|---|"]
    for pr in probes:
        L.append(f"| {pr['family']} | `{pr['tactic_or_sequence']}` | {'Y' if pr['is_sequence'] else 'N'} | "
                 f"{pr['risk']} | {','.join(pr['gate']['namespaces']) or '*'} | "
                 f"{','.join(pr['gate']['name_features']) or '*'} |")
    L += ["", f"> {out['warning']}"]
    open(args.out_md, "w").write("\n".join(L))
    print(f"[sf4-probes] probes={len(probes)} families={len(out['families'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
