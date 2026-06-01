#!/usr/bin/env python3
"""TR7 Part 4 — static coverage audit on the TR6 fresh wins (core diagnostic).

For each TR6 fresh true delta, determine whether the RC4 static wrapper could cover it:
  1. would any RC4 static action fire (name-prefix gate)?
  2. is the TR6 winning lemma in the RC4 allowlist?
  3. is the TR6 winning tactic representable in the RC4 schema (simp[L] / simp[L]<;>aesop / def-unfold)?
  4. is it already covered by RC4A/B/C_residue (an RC4R known win)?
Classify each into STATIC_COVERED_AND_SHOULD_SOLVE / STATIC_GATE_MISS / ALLOWLIST_MISS /
WRAPPER_REPRESENTATION_MISS / RC4C_RESIDUE_EXCLUDED / DYNAMIC_RETRIEVAL_REQUIRED / UNKNOWN.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# RC4 static allowlist (the 14 lemmas/defs in the validated wrapper)
RC4_ALLOWLIST = {
    "Monotone", "MonotoneOn", "Antitone", "AntitoneOn", "StrictMono", "StrictMonoOn",
    "StrictAnti", "StrictAntiOn", "Finset.disjUnion",                      # RC4A
    "Set.disjoint_left", "Multiset.disjoint_left",                          # RC4B
    "Multiset.disjoint_right", "Set.subset_pair_iff_eq", "List.forall_iff_forall_mem",  # RC4C_residue
}
# lemmas deliberately excluded from RC4C_residue (depth-1 simp_only duplicate)
RC4C_EXCLUDED = {"Finset.biUnion_subset"}
# TR6 families that are representable as a static RC4 schema action
STATIC_REPRESENTABLE_FAMILIES = {"d2_simp_aesop", "d1_simp_lemma", "def_unfold_simp"}
# families that are inherently non-static / theorem-specific
DYNAMIC_FAMILIES = {"d2_rw_aesop", "d1_exact", "d1_tauto", "d1_rw", "d2_rw_simp"}


def _p(*a):
    return os.path.join(_REPO, *a)


def _in_allowlist(lemma):
    if not lemma:
        return False
    if lemma in RC4_ALLOWLIST:
        return True
    # allow the def short-name form (RC4A defs)
    return lemma.split(".")[-1] in RC4_ALLOWLIST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--rc4-wrapper")
    ap.add_argument("--rc4-policy")
    ap.add_argument("--rc4-manifest")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(_p(args.corpus))]
    wins = [r for r in rows if r["is_tr6_fresh_win"]]

    records = []
    for r in wins:
        lemma = r.get("tr6_winning_lemma")
        fam = r.get("tr6_winning_family")
        gate = r["rc4_static_gate_fired"]
        rc4_res = r["rc4_static_result"]
        in_allow = _in_allowlist(lemma)
        representable = fam in STATIC_REPRESENTABLE_FAMILIES
        known = "rc4_known_wins" in r["rc4r_sets"]

        # classification
        if rc4_res == "solved":
            cls = "STATIC_COVERED_AND_SHOULD_SOLVE"
        elif rc4_res == "failed" and gate:
            cls = "WRAPPER_REPRESENTATION_MISS"      # gate fires, RC4 action exists, search fails
        elif lemma in RC4C_EXCLUDED:
            cls = "RC4C_RESIDUE_EXCLUDED"
        elif not gate:
            # no RC4 action fires at all
            if in_allow:
                cls = "STATIC_GATE_MISS"             # allowlisted lemma but gate prefix misses
            else:
                cls = "ALLOWLIST_MISS"
        else:
            # gate fires but theorem not in the RC4R benchmark (not_applicable); decide by lemma
            if in_allow:
                cls = "STATIC_COVERED_AND_SHOULD_SOLVE"   # RC4 action == TR6 winning lemma
            elif not representable:
                cls = "DYNAMIC_RETRIEVAL_REQUIRED"        # won via theorem-specific rw/exact/tauto
            else:
                cls = "ALLOWLIST_MISS"                    # representable form, lemma not allowlisted
        records.append({
            "full_name": r["full_name"], "namespace": r["namespace"],
            "tr6_winning_lemma": lemma, "tr6_winning_family": fam,
            "tr6_winning_program": r.get("tr6_winning_program"),
            "rc4_gate_fired": gate, "rc4_component": r["rc4_static_component"],
            "lemma_in_allowlist": in_allow, "schema_representable": representable,
            "rc4_benchmark_result": rc4_res, "is_rc4r_known_win": known,
            "classification": cls,
        })

    hist = Counter(r["classification"] for r in records)
    covered = [r for r in records if r["classification"] == "STATIC_COVERED_AND_SHOULD_SOLVE"]
    allow_miss = [r for r in records if r["classification"] == "ALLOWLIST_MISS"]
    gate_miss = [r for r in records if r["classification"] == "STATIC_GATE_MISS"]
    wrap_miss = [r for r in records if r["classification"] == "WRAPPER_REPRESENTATION_MISS"]
    dyn = [r for r in records if r["classification"] == "DYNAMIC_RETRIEVAL_REQUIRED"]
    excl = [r for r in records if r["classification"] == "RC4C_RESIDUE_EXCLUDED"]

    out = {
        "generated_by": "scripts/tr7_static_coverage_audit.py",
        "num_tr6_fresh_wins": len(wins),
        "classification_histogram": dict(hist),
        "summary": {
            "static_covered": len(covered),
            "allowlist_miss": len(allow_miss),
            "gate_miss": len(gate_miss),
            "wrapper_representation_miss": len(wrap_miss),
            "dynamic_retrieval_required": len(dyn),
            "rc4c_residue_excluded": len(excl),
            "would_rc4_cover": len(covered),
            "missing_due_to_allowlist": len(allow_miss) + len(excl),
            "missing_due_to_gate": len(gate_miss),
            "require_dynamic": len(dyn),
        },
        "allowlist_miss_lemmas": sorted(set(r["tr6_winning_lemma"] for r in allow_miss if r["tr6_winning_lemma"])),
        "records": records,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR7 static coverage audit (core diagnostic)", "",
          f"- TR6 fresh wins audited: {len(wins)}",
          f"- classification: {dict(hist)}", "",
          f"- **RC4 static would cover: {len(covered)}/{len(wins)}**",
          f"- missing due to allowlist (incl. RC4C-excluded): {len(allow_miss)+len(excl)}",
          f"- missing due to gate: {len(gate_miss)}",
          f"- require dynamic retrieval: {len(dyn)}",
          f"- wrapper-representation miss: {len(wrap_miss)}", "",
          "## Per-win classification", "",
          "| theorem | ns | lemma | family | gate | in_allow | rc4 | class |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sorted(records, key=lambda x: x["classification"]):
        md.append(f"| `{r['full_name']}` | {r['namespace']} | `{r['tr6_winning_lemma']}` | "
                  f"{r['tr6_winning_family']} | {r['rc4_gate_fired']} | {r['lemma_in_allowlist']} | "
                  f"{r['rc4_benchmark_result']} | {r['classification']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr7-coverage] {dict(hist)}")
    print(f"[tr7-coverage] covered={len(covered)} allowlist_miss={len(allow_miss)} "
          f"gate_miss={len(gate_miss)} wrap_miss={len(wrap_miss)} dynamic={len(dyn)} excluded={len(excl)}")


if __name__ == "__main__":
    main()
