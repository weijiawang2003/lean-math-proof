#!/usr/bin/env python3
"""FLI2 Part 6 — attribution / rescue classification over live deployment results.

Per solved/attempted action → TRUE_RETRIEVAL_GAP_RESCUE / PARTIAL_PROGRESS / CONTROL_DUPLICATE /
BASELINE_DUPLICATE / SELF_IMPORT_VACUOUS / UNKNOWN_NAME_OR_IMPORT_GAP / NO_RESCUE / NEEDS_REVIEW,
then aggregated to a per-theorem verdict (a theorem is rescued iff it has ≥1 robust TRUE rescue).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _progressed(r):
    rb, ra = r.get("residual_before"), r.get("residual_after")
    if not ra or r.get("solved"):
        return False
    if rb and ra and ra.strip() == rb.strip():
        return False
    # require the post-goal to be no larger (changed and not obviously bigger)
    return ra is not None and (len(ra) <= len(rb or "") * 1.2)


def classify(r):
    if r.get("vacuous_self"):
        return "SELF_IMPORT_VACUOUS"
    st = r.get("status")
    if st == "unknown_name":
        return "UNKNOWN_NAME_OR_IMPORT_GAP"
    if st in ("infra_error",):
        return "NEEDS_REVIEW"
    if r.get("solved"):
        if r.get("control_solved"):
            return "CONTROL_DUPLICATE"
        if r.get("is_rescue_candidate") and r.get("robust"):
            return "TRUE_RETRIEVAL_GAP_RESCUE"
        if r.get("is_rescue_candidate") and not r.get("robust"):
            return "NEEDS_REVIEW"  # flaky win
        return "CONTROL_DUPLICATE"
    if _progressed(r):
        return "PARTIAL_PROGRESS"
    return "NO_RESCUE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(_p(args.results)) if l.strip()]
    out = []
    for r in rows:
        cls = classify(r)
        rec = {
            "action_id": r.get("action_id"), "case_id": r.get("case_id"),
            "theorem": r["theorem"], "namespace": r.get("namespace"),
            "lemma": r.get("lemma"), "template": r.get("template"), "tactic": r.get("tactic"),
            "classification": cls, "status": r.get("status"),
            "control_solved": r.get("control_solved"), "robust": r.get("robust"),
            "expected_pattern": None,
            "residual_before": r.get("residual_before"), "residual_after": r.get("residual_after"),
            "why": "",
        }
        if cls == "TRUE_RETRIEVAL_GAP_RESCUE":
            rec["why"] = (f"controls {r.get('control_solved')} failed; `{r['tactic']}` deploying "
                          f"`{r.get('lemma')}` closes `{r['theorem']}` at position; robust")
            rec["suggests_reusable_rule"] = True
        elif cls == "PARTIAL_PROGRESS":
            rec["why"] = "goal changed/simplified but not closed"
        elif cls == "CONTROL_DUPLICATE":
            rec["why"] = f"bare control(s) {r.get('control_solved')} also solve"
        out.append(rec)

    # per-theorem verdict (best classification)
    RANK = ["TRUE_RETRIEVAL_GAP_RESCUE", "PARTIAL_PROGRESS", "CONTROL_DUPLICATE",
            "BASELINE_DUPLICATE", "SELF_IMPORT_VACUOUS", "UNKNOWN_NAME_OR_IMPORT_GAP",
            "NEEDS_REVIEW", "NO_RESCUE"]
    by_thm = defaultdict(list)
    for r in out:
        by_thm[r["theorem"]].append(r["classification"])
    thm_verdict = {t: min(cs, key=lambda c: RANK.index(c) if c in RANK else 99)
                   for t, cs in by_thm.items()}

    true_rescues = [r for r in out if r["classification"] == "TRUE_RETRIEVAL_GAP_RESCUE"]
    true_thms = sorted({r["theorem"] for r in true_rescues})
    with open(_p(args.out_jsonl), "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"generated_by": "scripts/fli2_classify_rescues.py",
               "num_actions": len(out),
               "action_classification": dict(Counter(r["classification"] for r in out).most_common()),
               "theorem_verdict": dict(Counter(thm_verdict.values()).most_common()),
               "true_retrieval_gap_rescue_actions": len(true_rescues),
               "true_retrieval_gap_rescue_theorems": len(true_thms),
               "true_rescue_theorem_list": true_thms,
               "partial_progress_theorems": sorted({t for t, v in thm_verdict.items()
                                                    if v == "PARTIAL_PROGRESS"}),
               "control_duplicate_theorems": sorted({t for t, v in thm_verdict.items()
                                                     if v == "CONTROL_DUPLICATE"}),
               "true_rescues": [{"theorem": r["theorem"], "lemma": r["lemma"],
                                 "tactic": r["tactic"], "namespace": r["namespace"],
                                 "controls_failed": r["control_solved"] == []} for r in true_rescues]}
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI2 rescue attribution summary", "",
          f"- actions classified: {summary['num_actions']}",
          f"- action classes: {summary['action_classification']}",
          f"- **per-theorem verdict: {summary['theorem_verdict']}**",
          f"- **TRUE_RETRIEVAL_GAP_RESCUE: {summary['true_retrieval_gap_rescue_theorems']} theorems "
          f"({summary['true_retrieval_gap_rescue_actions']} actions)**", "",
          "## True rescues", "", "| theorem | lemma | tactic |", "|---|---|---|"]
    for r in summary["true_rescues"]:
        md.append(f"| `{r['theorem']}` | `{r['lemma']}` | `{r['tactic']}` |")
    md += ["", f"- partial-progress theorems: {summary['partial_progress_theorems']}",
           f"- control-duplicate theorems: {len(summary['control_duplicate_theorems'])}"]
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli2-classify] TRUE_RESCUE thms={summary['true_retrieval_gap_rescue_theorems']} "
          f"actions={len(true_rescues)} | verdict={summary['theorem_verdict']}")


if __name__ == "__main__":
    main()
