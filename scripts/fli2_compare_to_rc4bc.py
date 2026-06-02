#!/usr/bin/env python3
"""FLI2 Part 8 — compare FLI2 discovered deployment rescues to the manually-validated RC4B/RC4C.

RC4B/RC4C were hand-built lemma-enabling static wrappers (disjoint_left bridge; selected
`simp [L]` enablers). FLI2 tries to *discover* analogous deployments from failure analysis. Reports
family overlap, new families, and whether FLI2 looks like a scalable RC-candidate generator.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RC4_FAMILIES = {
    "disjoint_left": "RC4B disjoint_left bridge (Set/Multiset)",
    "disjoint_right": "RC4C residue (Multiset.disjoint_right)",
    "subset_pair": "RC4C residue (Set.subset_pair_iff_eq)",
    "forall": "RC4C residue (List.forall_iff_forall_mem)",
    "biunion_subset": "RC4C (Finset.biUnion_subset)",
}


def _p(*a):
    return os.path.join(_REPO, *a)


def _fam_token(lemma):
    s = (lemma or "").lower()
    for k in RC4_FAMILIES:
        if k.replace("_", "") in s.replace("_", ""):
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fli2-attribution", required=True)
    ap.add_argument("--rc4bc", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    attr = [json.loads(l) for l in open(_p(args.fli2_attribution)) if l.strip()]
    true_r = [r for r in attr if r["classification"] == "TRUE_RETRIEVAL_GAP_RESCUE"]
    rc4bc = json.load(open(_p(args.rc4bc))) if os.path.exists(_p(args.rc4bc)) else {}

    overlap, new = [], []
    for r in true_r:
        ft = _fam_token(r["lemma"])
        (overlap if ft else new).append({"theorem": r["theorem"], "lemma": r["lemma"],
                                         "tactic": r["tactic"],
                                         "rc4_family": RC4_FAMILIES.get(ft) if ft else None})
    new_families = sorted({(r["namespace"], (r["lemma"] or "").split(".")[-1].split("_")[0])
                           for r in true_r if not _fam_token(r["lemma"])})

    rc4b_decision = (rc4bc.get("readiness", {}).get("rc4b", {}).get("committed_decision"))
    rc4c_decision = (rc4bc.get("readiness", {}).get("rc4c", {}).get("committed_decision"))

    out = {"generated_by": "scripts/fli2_compare_to_rc4bc.py",
           "rc4b_committed_decision": rc4b_decision,
           "rc4c_committed_decision": rc4c_decision,
           "rc4_validation_method": "manual hand-built static wrapper + literal-RC2 validation",
           "fli2_method": "automated failure-analysis → gated retrieved-lemma deployment → at-position rescue",
           "fli2_true_rescues": len(true_r),
           "overlap_with_rc4bc_families": overlap,
           "new_families_beyond_rc4bc": new,
           "new_family_signatures": [f"{ns}:{tok}" for ns, tok in new_families],
           "is_scalable_rc_candidate_generator": len(true_r) >= 1,
           "assessment": (
               "FLI2 discovers the same KIND of object RC4B/RC4C were hand-built for — a small "
               "gated `simp [L]`/closer action that deploys an existing lemma — but sourced "
               "automatically from failure analysis rather than manual curation. "
               + (f"{len(true_r)} at-position rescue(s) found; "
                  f"{len(overlap)} overlap RC4-style families, {len(new)} are new. "
                  if true_r else "No at-position rescues beyond the FLI1 case in this batch. ")
               + "Whether it becomes an RC-candidate generator depends on each rescue passing the "
                 "full literal-RC2 additive validation (off-gate/floors/determinism) used for "
                 "RC4A–RC4D; FLI2 only produces candidates, it does not validate or promote them.")}
    with open(_p(args.out_json), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    md = ["# FLI2 vs RC4B/RC4C", "",
          f"- RC4B decision: `{rc4b_decision}` | RC4C decision: `{rc4c_decision}`",
          f"- RC4 method: {out['rc4_validation_method']}",
          f"- FLI2 method: {out['fli2_method']}",
          f"- FLI2 true rescues: {out['fli2_true_rescues']}",
          f"- overlap with RC4 families: {len(overlap)} | new families: {len(new)} "
          f"{out['new_family_signatures']}", "",
          "## Overlap cases", ""]
    for o in overlap:
        md.append(f"- `{o['theorem']}` via `{o['tactic']}` ↔ {o['rc4_family']}")
    md += ["", "## New (beyond RC4B/RC4C)", ""]
    for o in new:
        md.append(f"- `{o['theorem']}` via `{o['tactic']}` (lemma `{o['lemma']}`)")
    md += ["", "## Assessment", "", out["assessment"]]
    with open(_p(args.out_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli2-rc4bc] true_rescues={len(true_r)} overlap={len(overlap)} new={len(new)}")


if __name__ == "__main__":
    main()
