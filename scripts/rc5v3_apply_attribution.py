#!/usr/bin/env python3
"""RC5V3 Part 10 — attribution over the incremental B1/B3/B5 dynamic results.

Merges the three budget result files into one per-theorem outcome (earliest budget that solved it),
then classifies each dynamic success using the bare controls + RC2/RC4 status + freshness:
  FRESH_TRUE_RC5V3_DELTA / RC4_DUPLICATE / BASELINE_DUPLICATE / RC2_ALREADY_SOLVED /
  SOURCE_SPECIFIC_DYNAMIC_WIN / OPEN_FLAKE / TIMEOUT_BOUNDED / NEEDS_REVIEW / NO_DYNAMIC_WIN.
Breaks fresh deltas out by budget B1/B3/B5, namespace, family, rank.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_SPECIFIC = {"d1_exact", "d1_rw", "d1_tauto", "d2_rw_aesop", "d2_rw_simp"}


def _p(*a):
    return os.path.join(_REPO, *a)


def _load_results(path):
    return {r["full_name"]: r for r in json.load(open(_p(path)))["results"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rc2-results", required=True)
    ap.add_argument("--static-results", required=True)
    ap.add_argument("--dynamic-b1", required=True)
    ap.add_argument("--dynamic-b3", required=True)
    ap.add_argument("--dynamic-b5", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rc2 = _load_results(args.rc2_results)
    static = _load_results(args.static_results)
    b1 = _load_results(args.dynamic_b1)
    b3 = _load_results(args.dynamic_b3)
    b5 = _load_results(args.dynamic_b5)
    plan = {t["full_name"]: t for t in json.load(open(_p(args.plan)))["theorems"]}
    elig = {}
    ep = _p("project/evolve/experiments/rc5_v3/cases/rc5v3_dynamic_eligible.json")
    if os.path.exists(ep):
        elig = {e["full_name"]: e for e in json.load(open(ep))["results"]}

    # merge: earliest budget that solved each theorem wins; otherwise last attempt seen.
    BUDGET_ORDER = [("B1", b1), ("B3", b3), ("B5", b5)]
    merged = {}
    for fn in plan:
        chosen = None
        chosen_budget = None
        last = None
        for label, store in BUDGET_ORDER:
            r = store.get(fn)
            if r is None:
                continue
            last = (label, r)
            if r.get("success"):
                chosen = r
                chosen_budget = label
                break
        if chosen is None and last is not None:
            chosen_budget, chosen = last
        if chosen is None:
            continue
        merged[fn] = (chosen_budget, chosen)

    records = []
    for fn, (budget, d) in merged.items():
        success = bool(d.get("success"))
        killed = bool(d.get("killed_by_timeout"))
        wp = d.get("winning_program") or {}
        fam = wp.get("family")
        controls = d.get("control_wins") or [c["tactic"] for c in d.get("controls", []) if c.get("solved")]
        rc2_solved = rc2.get(fn, {}).get("status") == "solved"
        rc4_solved = static.get(fn, {}).get("status") == "solved"
        fresh = elig.get(fn, {}).get("freshness_status", "strict_fresh")
        lemmas = wp.get("used_lemmas") or wp.get("lemmas") or []
        src_specific = (fam in SOURCE_SPECIFIC) or any(L == fn for L in lemmas)

        if not success:
            cls = "TIMEOUT_BOUNDED" if killed else (
                "OPEN_FLAKE" if d.get("setup_error") and "exceeded" in (d.get("setup_error") or "")
                else "NO_DYNAMIC_WIN")
        elif rc2_solved:
            cls = "RC2_ALREADY_SOLVED"
        elif rc4_solved:
            cls = "RC4_DUPLICATE"
        elif controls:
            cls = "BASELINE_DUPLICATE"
        elif src_specific:
            cls = "SOURCE_SPECIFIC_DYNAMIC_WIN"
        elif fresh in ("strict_fresh", "soft_fresh"):
            cls = "FRESH_TRUE_RC5V3_DELTA"
        else:
            cls = "NEEDS_REVIEW"
        records.append({"full_name": fn,
                        "namespace": d.get("namespace") or plan.get(fn, {}).get("namespace"),
                        "budget_solved": budget if success else None,
                        "success": success, "killed_by_timeout": killed,
                        "winning_program": wp.get("tactic"), "winning_family": fam,
                        "winning_lemmas": lemmas, "first_success_rank": d.get("first_success_rank"),
                        "freshness": fresh, "control_wins": controls, "classification": cls})

    hist = Counter(r["classification"] for r in records)
    fresh_delta = [r for r in records if r["classification"] == "FRESH_TRUE_RC5V3_DELTA"]
    by_ns = Counter(r["namespace"] for r in fresh_delta)
    by_fam = Counter(r["winning_family"] for r in fresh_delta)
    by_rank = Counter(r["first_success_rank"] for r in fresh_delta)
    by_budget = Counter(r["budget_solved"] for r in fresh_delta)
    out = {"generated_by": "scripts/rc5v3_apply_attribution.py",
           "num_dynamic_attempts": len(records), "classification_histogram": dict(hist),
           "fresh_true_deltas": len(fresh_delta),
           "fresh_true_delta_targets": sorted(r["full_name"] for r in fresh_delta),
           "fresh_delta_by_budget": dict(by_budget),
           "duplicates": hist.get("RC4_DUPLICATE", 0) + hist.get("BASELINE_DUPLICATE", 0)
           + hist.get("RC2_ALREADY_SOLVED", 0),
           "source_specific": hist.get("SOURCE_SPECIFIC_DYNAMIC_WIN", 0),
           "fresh_delta_by_namespace": dict(by_ns), "fresh_delta_by_family": dict(by_fam),
           "fresh_delta_by_rank": {str(k): v for k, v in by_rank.items()},
           "records": records}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V3 attribution", "",
          f"- dynamic attempts: {len(records)} | classifications: {dict(hist)}",
          f"- **FRESH_TRUE_RC5V3_DELTA: {len(fresh_delta)}** {out['fresh_true_delta_targets']}",
          f"- fresh delta by budget: {dict(by_budget)}",
          f"- duplicates: {out['duplicates']} | source-specific: {out['source_specific']}",
          f"- fresh delta by namespace: {dict(by_ns)} | by family: {dict(by_fam)} | by rank: {out['fresh_delta_by_rank']}",
          "", "| theorem | ns | budget | class | rank | winning program |", "|---|---|---|---|---|---|"]
    for r in sorted(records, key=lambda x: (x["classification"] != "FRESH_TRUE_RC5V3_DELTA", x["full_name"])):
        if r["classification"] == "NO_DYNAMIC_WIN":
            continue
        md.append(f"| `{r['full_name']}` | {r['namespace']} | {r.get('budget_solved')} | "
                  f"{r['classification']} | {r.get('first_success_rank')} | `{r['winning_program'] or ''}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v3-attrib] {dict(hist)}")
    print(f"[rc5v3-attrib] FRESH_TRUE_RC5V3_DELTA={len(fresh_delta)} by_budget={dict(by_budget)} "
          f"{out['fresh_true_delta_targets']}")


if __name__ == "__main__":
    main()
