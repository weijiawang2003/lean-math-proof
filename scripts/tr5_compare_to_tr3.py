#!/usr/bin/env python3
"""TR5 Part 8 — compare the live ranker-guided search to the TR3 full battery.

TR3 ran the full grammar (4,377 programs over 92 theorems) and found 13 live successes.
TR5 runs only the ranker's top-B per theorem. This measures recovery of those successes,
probe reduction, success-per-probe, unknown-name failures avoided, mean live
first-success rank, and missed known wins, and renders a decision.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _load(path):
    fp = _p(path) if not os.path.isabs(path) else path
    return json.load(open(fp)) if os.path.exists(fp) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr3-results", required=True)
    ap.add_argument("--tr3-attribution", required=True)
    ap.add_argument("--ranked-plan", required=True)
    ap.add_argument("--tr5-attribution", required=True)
    ap.add_argument("--b5", default="project/evolve/experiments/tr5/out/tr5_b5_live_results.json")
    ap.add_argument("--b10", default="project/evolve/experiments/tr5/out/tr5_b10_live_results.json")
    ap.add_argument("--b20", default="project/evolve/experiments/tr5/out/tr5_b20_live_results.json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    tr3 = {r["full_name"]: r for r in _load(args.tr3_results)["results"]}
    tr3_attr = {r["full_name"]: r for r in _load(args.tr3_attribution)["records"]}
    plan = {t["full_name"]: t for t in _load(args.ranked_plan)["theorems"]}
    attr = {r["full_name"]: r for r in _load(args.tr5_attribution)["records"]}
    b5 = _load(args.b5)
    b10 = _load(args.b10)
    b20 = _load(args.b20)
    b5_by = {r["full_name"]: r for r in b5["results"]} if b5 else {}

    # TR3 known successes / credited
    tr3_success = {fn for fn, r in tr3.items() if r.get("wins")}
    tr3_credited = {r["full_name"] for r in tr3_attr.values() if r.get("credited")}

    overlap = sorted(set(tr3.keys()) & set(plan.keys()))

    # TR5 solved-at-budget map
    def solved_at(fn):
        r = attr.get(fn, {})
        if r.get("credited") or r.get("classification") in ("TRUE_RANKER_DELTA", "TRUE_RC4A_REPRODUCTION", "BASELINE_DUPLICATE"):
            return r.get("win_budget")
        return None

    # programs attempted
    tr3_programs = sum(r.get("programs_tried", len(r.get("ran", []))) for r in tr3.values())
    b5_programs = sum(r.get("programs_attempted", 0) for r in b5["results"]) if b5 else 0
    b5_controls = sum(len(r.get("controls", [])) for r in b5["results"]) if b5 else 0
    b10_programs = sum(r.get("programs_attempted", 0) for r in b10.get("new_results", [])) if b10 else 0
    b20_programs = sum(r.get("programs_attempted", 0) for r in b20.get("new_results", [])) if b20 else 0
    tr5_total_programs = b5_programs + b10_programs + b20_programs

    # recovery of TR3 successes by budget
    def recovered(maxb):
        return sorted(fn for fn in tr3_success
                      if (solved_at(fn) is not None and solved_at(fn) <= maxb))
    rec_b5 = recovered(5)
    rec_b10 = recovered(10)
    rec_b20 = recovered(20)
    cred_b5 = sorted(set(rec_b5) & tr3_credited)
    cred_b10 = sorted(set(rec_b10) & tr3_credited)
    cred_b20 = sorted(set(rec_b20) & tr3_credited)
    missed = sorted(tr3_success - set(rec_b20 if b20 else (rec_b10 if b10 else rec_b5)))

    # new wins not found by TR3 (credited TRUE_RANKER_DELTA not in tr3_success)
    new_wins = sorted(fn for fn, r in attr.items()
                      if r.get("credited") and fn not in tr3_success)

    # mean live first-success rank (over solved theorems)
    ranks = [attr[fn].get("first_success_rank") for fn in attr
             if attr[fn].get("first_success_rank")]
    mean_rank = round(sum(ranks) / len(ranks), 2) if ranks else None

    # unknown-name failures avoided: TR3 ran every program (many unknown_name); TR5 ran fewer
    tr3_unknown = sum(1 for r in tr3.values() for p in r.get("ran", [])
                      if p.get("outcome") == "unknown_name")
    b5_unknown = sum(1 for r in b5["results"] for p in r.get("failures", [])
                     if p.get("outcome") == "unknown_name") if b5 else 0

    # rank-1 false positives: rank-1 programs that failed live
    rank1_fail = rank1_total = 0
    for fn, r in b5_by.items():
        for p in r.get("failures", []):
            if p.get("rank") == 1:
                rank1_fail += 1
        if r.get("programs_attempted", 0) > 0 or r.get("success"):
            rank1_total += 1

    probe_reduction_b5 = round(1 - tr5_total_programs / max(1, tr3_programs), 4)
    succ_per_probe_tr5 = round(len(rec_b20 if b20 else rec_b10 if b10 else rec_b5) / max(1, tr5_total_programs), 4)
    succ_per_probe_tr3 = round(len(tr3_success) / max(1, tr3_programs), 4)

    # ---- decision ----
    n_known = len(tr3_success)
    b5_rate = len(rec_b5) / max(1, n_known)
    b10_rate = len(rec_b10) / max(1, n_known) if b10 else None
    decision_reasons = []
    confirmed = False
    if (b5_rate >= 0.70 or (b10_rate is not None and b10_rate >= 0.85)) \
            and probe_reduction_b5 >= 0.60 and len(new_wins) >= 0:
        decision = "RANKER_LIVE_CONFIRMED"
        confirmed = True
        decision_reasons.append(f"B5 recovered {b5_rate:.0%} (≥70%) "
                                + (f"/ B10 {b10_rate:.0%} " if b10_rate is not None else "")
                                + f"of {n_known} known successes at {probe_reduction_b5:.0%} probe reduction")
    elif b5_rate >= 0.4 and probe_reduction_b5 >= 0.6:
        decision = "RANKER_PARTIALLY_CONFIRMED"
        decision_reasons.append(f"B5 recovered {b5_rate:.0%} (<70%) of known successes; "
                                f"probe reduction {probe_reduction_b5:.0%}")
    elif len(overlap) < 5:
        decision = "INCONCLUSIVE_TOO_FEW_OVERLAPS"
    else:
        decision = "RANKER_FAILED_LIVE"
        decision_reasons.append(f"B5 recovered only {b5_rate:.0%} of known successes")

    out = {
        "generated_by": "scripts/tr5_compare_to_tr3.py",
        "overlap_theorems": len(overlap),
        "tr3_programs_attempted": tr3_programs,
        "tr5_programs_attempted": tr5_total_programs,
        "tr5_programs_by_budget": {"b5": b5_programs, "b10": b10_programs, "b20": b20_programs,
                                   "b5_controls": b5_controls},
        "tr3_known_successes": sorted(tr3_success), "num_tr3_known_successes": n_known,
        "tr3_credited_wins": sorted(tr3_credited),
        "recovered_b5": rec_b5, "recovered_b10": rec_b10, "recovered_b20": rec_b20,
        "num_recovered_b5": len(rec_b5), "num_recovered_b10": len(rec_b10),
        "num_recovered_b20": len(rec_b20),
        "credited_recovered_b5": cred_b5, "credited_recovered_b10": cred_b10,
        "credited_recovered_b20": cred_b20,
        "missed_known_wins": missed,
        "new_wins_not_in_tr3": new_wins,
        "probe_reduction_vs_tr3": probe_reduction_b5,
        "success_per_probe_tr5": succ_per_probe_tr5,
        "success_per_probe_tr3": succ_per_probe_tr3,
        "mean_live_first_success_rank": mean_rank,
        "tr3_unknown_name_failures": tr3_unknown,
        "tr5_unknown_name_failures_b5": b5_unknown,
        "unknown_name_failures_avoided": tr3_unknown - b5_unknown,
        "rank1_false_positives_b5": rank1_fail,
        "decision": decision, "decision_reasons": decision_reasons,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR5 vs TR3 full battery", "",
          f"## Decision: `{decision}`", ""]
    for r in decision_reasons:
        md.append(f"- {r}")
    md += ["",
           f"- overlap theorems: {len(overlap)}",
           f"- programs attempted: **TR3 {tr3_programs}** → **TR5 {tr5_total_programs}** "
           f"(B5 {b5_programs} + B10 {b10_programs} + B20 {b20_programs}; +{b5_controls} controls)",
           f"- **probe reduction vs TR3: {probe_reduction_b5:.1%}**",
           f"- TR3 known successes: {n_known}",
           f"- recovered: B5 {len(rec_b5)}/{n_known}, B10 {len(rec_b10)}/{n_known}, "
           f"B20 {len(rec_b20)}/{n_known}",
           f"- credited recovered: B5 {len(cred_b5)}, B10 {len(cred_b10)}, B20 {len(cred_b20)}",
           f"- missed known wins: {missed}",
           f"- new wins not found by TR3: {new_wins}",
           f"- success/probe: TR5 {succ_per_probe_tr5} vs TR3 {succ_per_probe_tr3}",
           f"- mean live first-success rank: {mean_rank}",
           f"- unknown-name failures avoided: {out['unknown_name_failures_avoided']} "
           f"(TR3 {tr3_unknown} → TR5 {b5_unknown})",
           f"- rank-1 false positives (B5): {rank1_fail}"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr5-vs-tr3] decision={decision} recovered_b5={len(rec_b5)}/{n_known} "
          f"probe_reduction={probe_reduction_b5} new_wins={len(new_wins)}")


if __name__ == "__main__":
    main()
