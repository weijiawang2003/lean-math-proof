#!/usr/bin/env python3
"""FLI3 Part 2 — extract FLI2 rescue candidates for literal validation.

All robust TRUE_RETRIEVAL_GAP_RESCUE cases (deduped to one best action per theorem) + selected
high-quality PARTIAL_PROGRESS cases (carried separately, never mixed into rescue attribution).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# theorem → candidate family (per the FLI2 finding / task spec)
DEF_UNFOLD_FINSET = {"Finset.mem_filterMap", "Finset.mem_map", "Finset.mem_preimage",
                     "Finset.card_subtype"}


def _p(*a):
    return os.path.join(_REPO, *a)


def _family(theorem, lemma):
    if theorem == "Finset.card_le_one_iff":
        return "FINSET_CARD_BRIDGE"
    if theorem in DEF_UNFOLD_FINSET or (theorem.startswith("Finset.") and lemma and
                                        lemma.split(".")[-1] in ("filterMap", "map", "preimage", "subtype")):
        return "FINSET_MEM_DEF_UNFOLD"
    if theorem.startswith("List.") and lemma and "bidirectionalRec" in (lemma or ""):
        return "LIST_DEF_UNFOLD"
    ns = theorem.split(".")[0]
    return f"{ns.upper()}_OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fli2-attribution", required=True)
    ap.add_argument("--fli2-results", required=True)
    ap.add_argument("--fli2-rules", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    attr = [json.loads(l) for l in open(_p(args.fli2_attribution)) if l.strip()]
    true = [r for r in attr if r["classification"] == "TRUE_RETRIEVAL_GAP_RESCUE"]
    partials = [r for r in attr if r["classification"] == "PARTIAL_PROGRESS"]

    # one best action per rescued theorem (prefer simplest template that solved & robust)
    PREF = {"SIMPLE_SIMP": 0, "EXACT_LEMMA": 1, "EXT_SIMP": 2, "SIMP_AESOP": 3,
            "CONSTRUCTOR_SIMP": 4, "INTRO_SIMP_AESOP": 5}
    by_thm = {}
    for r in true:
        if not r.get("robust"):
            continue
        t = r["theorem"]
        if t not in by_thm or PREF.get(r["template"], 9) < PREF.get(by_thm[t]["template"], 9):
            by_thm[t] = r

    candidates = []
    for i, (t, r) in enumerate(sorted(by_thm.items()), 1):
        candidates.append({
            "candidate_id": f"FLI3-C{i:02d}", "theorem": t, "namespace": r["namespace"],
            "lemma": r["lemma"], "template": r["template"], "tactic": r["tactic"],
            "pattern": r.get("expected_pattern"), "source": "FLI2_TRUE_RETRIEVAL_GAP_RESCUE",
            "controls_failed": r.get("control_solved") == [], "robust": bool(r.get("robust")),
            "non_vacuous": True, "source_trace": "fli2_rescue_attribution",
            "candidate_family": _family(t, r["lemma"]),
        })

    # selected partials: meaningful residual change, good namespace, lemma present, not unknown
    GOOD = {"Finset", "List", "Multiset", "Set", "Nat"}
    part_sel, seen = [], set()
    for r in partials:
        t = r["theorem"]
        if t in seen or t in by_thm:
            continue
        ns = (r["namespace"] or "").split(".")[0]
        ra, rb = r.get("residual_after"), r.get("residual_before")
        if ns in GOOD and r.get("lemma") and ra and rb and ra.strip() != rb.strip():
            seen.add(t)
            part_sel.append({
                "candidate_id": f"FLI3-P{len(part_sel)+1:02d}", "theorem": t,
                "namespace": r["namespace"], "lemma": r["lemma"], "template": r.get("template"),
                "tactic": r["tactic"], "pattern": r.get("expected_pattern"),
                "source": "FLI2_PARTIAL_PROGRESS", "candidate_family": _family(t, r.get("lemma")),
                "residual_before": (rb or "")[:200], "residual_after": (ra or "")[:200]})

    out = {"generated_by": "scripts/fli3_extract_rescue_candidates.py",
           "num_rescue_candidates": len(candidates), "num_partial_candidates": len(part_sel),
           "rescue_candidates": candidates, "partial_candidates": part_sel[:20]}
    with open(_p(args.out_json), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    summary = {"generated_by": "scripts/fli3_extract_rescue_candidates.py",
               "num_rescue_candidates": len(candidates),
               "rescue_by_family": dict(Counter(c["candidate_family"] for c in candidates).most_common()),
               "rescue_theorems": [c["theorem"] for c in candidates],
               "num_partial_candidates": len(part_sel),
               "partial_by_family": dict(Counter(c["candidate_family"] for c in part_sel).most_common())}
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI3 rescue candidate summary", "",
          f"- **rescue candidates (robust TRUE rescues): {len(candidates)}** by family "
          f"{summary['rescue_by_family']}",
          f"- partial-progress candidates (separate): {len(part_sel)} by family "
          f"{summary['partial_by_family']}", "",
          "| id | theorem | family | lemma | tactic |", "|---|---|---|---|---|"]
    for c in candidates:
        md.append(f"| {c['candidate_id']} | `{c['theorem']}` | {c['candidate_family']} | "
                  f"`{c['lemma']}` | `{c['tactic']}` |")
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli3-extract] rescues={len(candidates)} families={summary['rescue_by_family']} "
          f"partials={len(part_sel)}")


if __name__ == "__main__":
    main()
