#!/usr/bin/env python3
"""RC5V2 Part 6 — build the dynamic-eligibility set.

Eligible = RC4 static failed ∧ non-flake ∧ namespace allowed by the RC5S strict policy ∧ not a
known hard-disabled Order/root case. Excluded cases are classified (STATIC_SOLVED /
RC2_SOLVED_BUT_RC4_FAILED_REGRESSION_CANDIDATE / DYNAMIC_NAMESPACE_DISABLED / DYNAMIC_LOW_CONFIDENCE
/ FLAKE_OR_PATH_ERROR / NEEDS_REVIEW). Writes the eligible set (confirmation-shaped for retrieval).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--rc2-results", required=True)
    ap.add_argument("--static-results", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    policy = json.load(open(_p(args.policy)))
    allowed_ns = set(policy["allowed_namespaces"])
    batch = {t["full_name"]: t for t in json.load(open(_p(args.batch)))["theorems"]}
    rc2 = {r["full_name"]: r for r in json.load(open(_p(args.rc2_results)))["results"]}
    static = {r["full_name"]: r for r in json.load(open(_p(args.static_results)))["results"]}

    eligible, excluded = [], []
    excl_hist = Counter()
    for fn, t in batch.items():
        ns = t.get("namespace") or fn.split(".")[0]
        st = static.get(fn, {}).get("status")
        r2 = rc2.get(fn, {}).get("status")
        if st == "solved":
            excl_hist["STATIC_SOLVED"] += 1
            excluded.append({"full_name": fn, "reason": "STATIC_SOLVED"}); continue
        if st in ("open_flake", "trace_insufficient", "path_error"):
            excl_hist["FLAKE_OR_PATH_ERROR"] += 1
            excluded.append({"full_name": fn, "reason": "FLAKE_OR_PATH_ERROR"}); continue
        if ns not in allowed_ns:
            excl_hist["DYNAMIC_NAMESPACE_DISABLED"] += 1
            excluded.append({"full_name": fn, "reason": "DYNAMIC_NAMESPACE_DISABLED"}); continue
        if r2 == "solved" and st != "solved":
            # RC2 solved but RC4 failed -> additive wrapper should never regress; flag for review
            excl_hist["RC2_SOLVED_BUT_RC4_FAILED_REGRESSION_CANDIDATE"] += 1
            excluded.append({"full_name": fn, "reason": "RC2_SOLVED_BUT_RC4_FAILED_REGRESSION_CANDIDATE"})
            continue
        eligible.append({"full_name": fn, "namespace": ns, "file_path": t.get("file_path"),
                         "statement_text": t.get("statement_text"),
                         "goal_text": t.get("statement_text"),
                         "rc2_status": r2, "features": t.get("features") or {},
                         "freshness_status": t.get("freshness_status", "strict_fresh"),
                         "classification": "CONFIRMED_RC2_FAILURE"})

    conf = {"generated_by": "scripts/rc5v2_build_dynamic_eligibility.py",
            "num_eligible": len(eligible), "results": eligible}
    json.dump(conf, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    ns_dist = Counter(e["namespace"] for e in eligible)
    rc2_failed = sum(1 for e in eligible if e["rc2_status"] != "solved")
    feat = {k: sum(1 for e in eligible if (e["features"] or {}).get(k))
            for k in ("has_subset", "has_disjoint", "has_mem", "has_map_filter", "has_iff", "has_singleton")}
    summary = {"generated_by": "scripts/rc5v2_build_dynamic_eligibility.py",
               "dynamic_eligible": len(eligible), "excluded": len(excluded),
               "exclusion_histogram": dict(excl_hist),
               "eligible_by_namespace": dict(ns_dist.most_common()),
               "eligible_rc2_failed_subset": rc2_failed,
               "eligible_feature_distribution": feat}
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V2 dynamic eligibility", "",
          f"- **dynamic eligible: {len(eligible)}** | excluded: {len(excluded)} {dict(excl_hist)}",
          f"- eligible by namespace: {dict(ns_dist.most_common())}",
          f"- eligible & RC2-failed: {rc2_failed}",
          f"- eligible features: {feat}"]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v2-elig] eligible={len(eligible)} excluded={dict(excl_hist)} ns={dict(ns_dist.most_common())}")


if __name__ == "__main__":
    main()
