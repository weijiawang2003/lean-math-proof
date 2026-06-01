#!/usr/bin/env python3
"""RC5S Part 9 — attribution + safety classification.

Per theorem (and per run-level safety aggregate), classify the safe-stage outcome:
  SAFE_TRUE_DYNAMIC_WIN / SAFE_NEW_DYNAMIC_WIN / LOST_WIN_DUE_TO_POLICY / TIMEOUT_BOUNDED /
  OFF_POLICY_BLOCKED / UNSAFE_PROGRAM_QUARANTINED / NO_WIN_SAFE_BUDGET.
Reports recovered/lost prior wins, new wins, bounded timeouts, and remaining safety issues.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RC5H_WINNERS = {"Finset.biUnion_subset_iff_forall_subset", "Multiset.add_bind", "Finset.image_subset_iff"}
FILTER_REPORT = "project/evolve/experiments/rc5_safety/out/rc5s_filter_report.json"


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--b5", required=True)
    ap.add_argument("--b10")
    ap.add_argument("--rc5h-attribution",
                    default="project/evolve/experiments/rc5_hybrid/out/rc5h_hybrid_attribution.json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    b5 = {r["full_name"]: r for r in json.load(open(_p(args.b5)))["results"]}
    b10 = {}
    if args.b10 and os.path.exists(_p(args.b10)):
        b10 = {r["full_name"]: r for r in json.load(open(_p(args.b10)))["results"]}
    plan = {t["full_name"]: t for t in json.load(open(_p(args.plan)))["theorems"]}
    filt = json.load(open(_p(FILTER_REPORT))) if os.path.exists(_p(FILTER_REPORT)) else {}

    # merged success across B5/B10
    merged = {}
    for fn in plan:
        r5 = b5.get(fn, {})
        r10 = b10.get(fn, {})
        win = r5 if r5.get("success") else (r10 if r10.get("success") else (r5 or r10))
        merged[fn] = win

    records = []
    for fn, r in merged.items():
        success = bool(r.get("success"))
        killed = bool(r.get("killed_by_timeout"))
        control_wins = r.get("control_wins") or []
        if success and not control_wins:
            cls = "SAFE_TRUE_DYNAMIC_WIN" if fn in RC5H_WINNERS else "SAFE_NEW_DYNAMIC_WIN"
        elif success and control_wins:
            cls = "NO_WIN_SAFE_BUDGET"  # solved by a bare control, not the dynamic program
        elif killed:
            cls = "TIMEOUT_BOUNDED"
        elif fn in RC5H_WINNERS:
            cls = "LOST_WIN_DUE_TO_POLICY"  # was a winner, not reproduced under safe stage
        else:
            cls = "NO_WIN_SAFE_BUDGET"
        records.append({"full_name": fn, "namespace": plan[fn].get("namespace"),
                        "category": plan[fn].get("category"),
                        "success": success, "killed_by_timeout": killed,
                        "wall_seconds": r.get("wall_seconds"),
                        "winning_program": (r.get("winning_program") or {}).get("tactic"),
                        "classification": cls})

    hist = Counter(r["classification"] for r in records)
    recovered = sorted(r["full_name"] for r in records
                       if r["classification"] == "SAFE_TRUE_DYNAMIC_WIN")
    lost = sorted(r["full_name"] for r in records if r["classification"] == "LOST_WIN_DUE_TO_POLICY")
    new = sorted(r["full_name"] for r in records if r["classification"] == "SAFE_NEW_DYNAMIC_WIN")
    bounded = sorted(r["full_name"] for r in records if r["classification"] == "TIMEOUT_BOUNDED")

    # run-level safety aggregates
    off_policy_blocked = filt.get("off_policy_removed", 0)
    quarantined = filt.get("stall_risk_removed", 0)
    out = {
        "generated_by": "scripts/rc5s_attribution_and_safety.py",
        "classification_histogram": dict(hist),
        "recovered_prior_wins": recovered, "num_recovered": len(recovered),
        "lost_prior_wins": lost, "num_lost": len(lost),
        "new_wins": new, "num_new": len(new),
        "bounded_timeouts": bounded, "num_bounded_timeouts": len(bounded),
        "off_policy_blocked_pre_execution": off_policy_blocked,
        "unsafe_programs_quarantined": quarantined,
        "rc5h_winners_total": len(RC5H_WINNERS),
        "remaining_safety_issues": (["bounded timeouts present (acceptable)"] if bounded else [])
                                   + (["lost prior wins"] if lost else []),
        "records": records,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5S attribution & safety", "",
          f"- classification: {dict(hist)}",
          f"- **recovered prior wins: {len(recovered)}/{len(RC5H_WINNERS)}** {recovered}",
          f"- lost prior wins: {len(lost)} {lost}",
          f"- new safe wins: {len(new)} {new}",
          f"- bounded timeouts: {len(bounded)}",
          f"- off-policy blocked (pre-execution): {off_policy_blocked} | "
          f"unsafe quarantined: {quarantined}", "",
          "| theorem | category | class | success | killed | wall(s) |", "|---|---|---|---|---|---|"]
    for r in sorted(records, key=lambda x: (x["classification"], x["full_name"])):
        md.append(f"| `{r['full_name']}` | {r.get('category')} | {r['classification']} | "
                  f"{r['success']} | {r['killed_by_timeout']} | {r.get('wall_seconds')} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5s-attrib] {dict(hist)}")
    print(f"[rc5s-attrib] recovered={len(recovered)}/{len(RC5H_WINNERS)} {recovered} lost={lost} new={new} bounded={len(bounded)}")


if __name__ == "__main__":
    main()
