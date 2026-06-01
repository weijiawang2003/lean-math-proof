#!/usr/bin/env python3
"""RC5V2 Part 10 — attribution.

Classifies each dynamic B5 success using the bare controls already run in the B5 stage + RC2/RC4
status + freshness:
  FRESH_TRUE_RC5V2_DELTA / TRUE_RC5V2_DELTA_KNOWN_CONTROL / STATIC_DUPLICATE / BASELINE_DUPLICATE /
  RC2_ALREADY_SOLVED / SOURCE_SPECIFIC_DYNAMIC_WIN / NO_DYNAMIC_WIN / OPEN_FLAKE / TIMEOUT_BOUNDED.
All RC5V2 batch theorems are strict_fresh, so a true delta is fresh unless the theorem carries a
known_control freshness tag.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rc2-results", required=True)
    ap.add_argument("--static-results", required=True)
    ap.add_argument("--dynamic-results", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rc2 = {r["full_name"]: r for r in json.load(open(_p(args.rc2_results)))["results"]}
    static = {r["full_name"]: r for r in json.load(open(_p(args.static_results)))["results"]}
    dyn = {r["full_name"]: r for r in json.load(open(_p(args.dynamic_results)))["results"]}
    plan = {t["full_name"]: t for t in json.load(open(_p(args.plan)))["theorems"]}
    elig = {}
    ep = _p("project/evolve/experiments/rc5_v2/cases/rc5v2_dynamic_eligible.json")
    if os.path.exists(ep):
        elig = {e["full_name"]: e for e in json.load(open(ep))["results"]}

    records = []
    for fn, d in dyn.items():
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
            cls = "TIMEOUT_BOUNDED" if killed else ("OPEN_FLAKE" if d.get("setup_error") and "exceeded" in (d.get("setup_error") or "") else "NO_DYNAMIC_WIN")
        elif rc2_solved:
            cls = "RC2_ALREADY_SOLVED"
        elif rc4_solved:
            cls = "STATIC_DUPLICATE"
        elif controls:
            cls = "BASELINE_DUPLICATE"
        elif src_specific:
            cls = "SOURCE_SPECIFIC_DYNAMIC_WIN"
        elif fresh in ("strict_fresh", "soft_fresh"):
            cls = "FRESH_TRUE_RC5V2_DELTA"
        else:
            cls = "TRUE_RC5V2_DELTA_KNOWN_CONTROL"
        records.append({"full_name": fn, "namespace": d.get("namespace") or plan.get(fn, {}).get("namespace"),
                        "success": success, "killed_by_timeout": killed,
                        "winning_program": wp.get("tactic"), "winning_family": fam,
                        "winning_lemmas": lemmas, "first_success_rank": d.get("first_success_rank"),
                        "freshness": fresh, "control_wins": controls, "classification": cls})

    hist = Counter(r["classification"] for r in records)
    fresh_delta = [r for r in records if r["classification"] == "FRESH_TRUE_RC5V2_DELTA"]
    known_delta = [r for r in records if r["classification"] == "TRUE_RC5V2_DELTA_KNOWN_CONTROL"]
    by_ns = Counter(r["namespace"] for r in fresh_delta)
    by_fam = Counter(r["winning_family"] for r in fresh_delta)
    by_rank = Counter(r["first_success_rank"] for r in fresh_delta)
    out = {"generated_by": "scripts/rc5v2_apply_attribution.py",
           "num_dynamic_attempts": len(records), "classification_histogram": dict(hist),
           "fresh_true_deltas": len(fresh_delta),
           "fresh_true_delta_targets": [r["full_name"] for r in fresh_delta],
           "known_control_deltas": len(known_delta),
           "duplicates": hist.get("STATIC_DUPLICATE", 0) + hist.get("BASELINE_DUPLICATE", 0) + hist.get("RC2_ALREADY_SOLVED", 0),
           "source_specific": hist.get("SOURCE_SPECIFIC_DYNAMIC_WIN", 0),
           "fresh_delta_by_namespace": dict(by_ns), "fresh_delta_by_family": dict(by_fam),
           "fresh_delta_by_rank": {str(k): v for k, v in by_rank.items()},
           "records": records}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V2 attribution", "",
          f"- dynamic attempts: {len(records)} | classifications: {dict(hist)}",
          f"- **FRESH_TRUE_RC5V2_DELTA: {len(fresh_delta)}** {out['fresh_true_delta_targets']}",
          f"- known/control deltas: {len(known_delta)} | duplicates: {out['duplicates']} | "
          f"source-specific: {out['source_specific']}",
          f"- fresh delta by namespace: {dict(by_ns)} | by family: {dict(by_fam)}", "",
          "| theorem | ns | class | rank | winning program |", "|---|---|---|---|---|"]
    for r in sorted(records, key=lambda x: (x["classification"] != "FRESH_TRUE_RC5V2_DELTA", x["full_name"])):
        if r["classification"] in ("NO_DYNAMIC_WIN",):
            continue
        md.append(f"| `{r['full_name']}` | {r['namespace']} | {r['classification']} | "
                  f"{r.get('first_success_rank')} | `{r['winning_program'] or ''}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v2-attrib] {dict(hist)}")
    print(f"[rc5v2-attrib] FRESH_TRUE_RC5V2_DELTA={len(fresh_delta)} {out['fresh_true_delta_targets']}")


if __name__ == "__main__":
    main()
