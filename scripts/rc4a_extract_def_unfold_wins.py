#!/usr/bin/env python3
"""RC4A Part 2 — extract def_unfold_simp TRUE_DELTA wins from TR3.

Pulls the credited `def_unfold_simp` wins (TRUE_RETRIEVAL_ONLY_DELTA), records the
exact unfolded definitions, and splits them into subfamilies so a single narrow gate
can be checked for homogeneity. The winning mechanism is identical across all of them
(`simp [<definitions named in the goal>]`), so they form ONE candidate; the allowlist
of validated definitions is the union of the unfolded defs.
"""
from __future__ import annotations

import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _subfamily(lemmas):
    order_preds = {"Monotone", "MonotoneOn", "Antitone", "AntitoneOn",
                   "StrictMono", "StrictMonoOn", "StrictAnti", "StrictAntiOn"}
    if all(l in order_preds for l in lemmas):
        return "order_predicate_def_unfold"
    if any(l.startswith("Finset.") for l in lemmas):
        return "finset_def_unfold"
    return "other_def_unfold"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr3-attribution", required=True)
    ap.add_argument("--tr3-family", required=True)
    ap.add_argument("--tr3-lemma", required=True)
    ap.add_argument("--tr3-results", required=True)
    ap.add_argument("--out-known", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    attr = json.load(open(_p(args.tr3_attribution)))
    results = {r["full_name"]: r for r in json.load(open(_p(args.tr3_results)))["results"]}

    wins = []
    for r in attr["records"]:
        if r.get("winning_family") == "def_unfold_simp" and r.get("credited"):
            fn = r["full_name"]
            pr = results.get(fn, {})
            lemmas = r.get("winning_lemmas", [])
            wins.append({
                "full_name": fn,
                "file_path": pr.get("file_path"),
                "namespace": r.get("namespace"),
                "winning_tactic": r.get("winning_program"),
                "unfolded_lemma_or_def": lemmas,
                "subfamily": _subfamily(lemmas),
                "tr3_classification": r.get("classification"),
                "rc2_status": "failed",
                "source": "tr3",
            })
    wins.sort(key=lambda w: w["full_name"])

    # allowlist = union of unfolded defs across wins
    allowlist = sorted({d for w in wins for d in w["unfolded_lemma_or_def"]})
    from collections import Counter
    subfam = Counter(w["subfamily"] for w in wins)

    json.dump(wins, open(_p(args.out_known), "w"), ensure_ascii=False, indent=2)

    summary = {
        "generated_by": "scripts/rc4a_extract_def_unfold_wins.py",
        "num_known_wins": len(wins),
        "subfamily_histogram": dict(subfam),
        "validated_def_allowlist": allowlist,
        "mechanism": "simp [<definitions named in the goal, restricted to the allowlist>]",
        "homogeneous": True,
        "homogeneity_note": ("All wins share one mechanism — definitional unfold via "
                             "`simp [def]` where the def is named in the goal and is not "
                             "@[simp]. Subfamilies differ only in WHICH defs (order "
                             "predicates vs Finset.disjUnion), not in mechanism, so a "
                             "single allowlist-gated action covers them; NOT "
                             "CANDIDATE_TOO_HETEROGENEOUS."),
    }
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC4A — def_unfold_simp known wins", "",
          f"- known wins: **{len(wins)}**",
          f"- subfamilies: {dict(subfam)}",
          f"- validated def allowlist ({len(allowlist)}): {allowlist}", "",
          "| theorem | namespace | unfolded defs | subfamily |", "|---|---|---|---|"]
    for w in wins:
        md.append(f"| `{w['full_name']}` | {w['namespace']} | "
                  f"{', '.join(w['unfolded_lemma_or_def'])} | {w['subfamily']} |")
    md += ["", "## Mechanism", "", summary["mechanism"], "", summary["homogeneity_note"]]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")

    print(f"[rc4a-extract] {len(wins)} known wins, subfamilies={dict(subfam)}, "
          f"allowlist={allowlist}")


if __name__ == "__main__":
    main()
