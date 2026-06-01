#!/usr/bin/env python3
"""TR7 Part 8 — dynamic-retrieval vs static-wrapper classification.

Combines the static coverage audit + missing-allowlist recommendation + gate analysis to put
each TR6 fresh win into one of:
  STATIC_WRAPPER_COMPATIBLE_NOW
  STATIC_WRAPPER_COMPATIBLE_WITH_ALLOWLIST_EXPANSION
  STATIC_WRAPPER_COMPATIBLE_WITH_GATE_REFINEMENT
  STATIC_WRAPPER_COMPATIBLE_WITH_SCHEMA_FIX
  DYNAMIC_RETRIEVAL_PREFERRED
  SEARCH_ONLY_FAMILY
Then summarizes the static-compatible vs dynamic-only split and the recommended RC5 direction.
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
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--allowlist", required=True)
    ap.add_argument("--gate-analysis", required=True)
    ap.add_argument("--tr6-attribution", required=True)
    ap.add_argument("--rc4-fresh", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    cov = {r["full_name"]: r for r in json.load(open(_p(args.coverage)))["records"]}
    allow = {r["full_name"]: r for r in json.load(open(_p(args.allowlist)))["records"]}

    records = []
    for fn, c in cov.items():
        cls = c["classification"]
        a = allow.get(fn, {})
        rec = a.get("recommendation")
        if cls == "STATIC_COVERED_AND_SHOULD_SOLVE":
            ds = "STATIC_WRAPPER_COMPATIBLE_NOW"
        elif cls == "WRAPPER_REPRESENTATION_MISS":
            ds = "STATIC_WRAPPER_COMPATIBLE_WITH_SCHEMA_FIX"
        elif cls == "STATIC_GATE_MISS":
            ds = "STATIC_WRAPPER_COMPATIBLE_WITH_GATE_REFINEMENT"
        elif cls == "RC4C_RESIDUE_EXCLUDED":
            # depth-1 simp lemma, deliberately excluded; re-addable via allowlist (depth-1)
            ds = "STATIC_WRAPPER_COMPATIBLE_WITH_ALLOWLIST_EXPANSION"
        elif cls == "ALLOWLIST_MISS":
            if rec in ("ADD_TO_STATIC_ALLOWLIST", "NEED_MORE_EVIDENCE"):
                ds = "STATIC_WRAPPER_COMPATIBLE_WITH_ALLOWLIST_EXPANSION"
            else:
                ds = "DYNAMIC_RETRIEVAL_PREFERRED"
        elif cls == "DYNAMIC_RETRIEVAL_REQUIRED":
            ds = "DYNAMIC_RETRIEVAL_PREFERRED"
        else:
            ds = "SEARCH_ONLY_FAMILY"
        records.append({"full_name": fn, "namespace": c["namespace"],
                        "static_coverage_class": cls, "allowlist_recommendation": rec,
                        "dynamic_vs_static_class": ds})

    hist = Counter(r["dynamic_vs_static_class"] for r in records)
    n = len(records)
    static_now = hist.get("STATIC_WRAPPER_COMPATIBLE_NOW", 0)
    static_with_work = sum(hist.get(k, 0) for k in (
        "STATIC_WRAPPER_COMPATIBLE_WITH_ALLOWLIST_EXPANSION",
        "STATIC_WRAPPER_COMPATIBLE_WITH_GATE_REFINEMENT",
        "STATIC_WRAPPER_COMPATIBLE_WITH_SCHEMA_FIX"))
    dynamic = sum(hist.get(k, 0) for k in ("DYNAMIC_RETRIEVAL_PREFERRED", "SEARCH_ONLY_FAMILY"))
    pct_static_compatible = round((static_now + static_with_work) / (n or 1), 3)
    pct_dynamic = round(dynamic / (n or 1), 3)

    # RC5 direction
    if pct_static_compatible >= 0.8 and dynamic <= 2:
        direction = "static RC5"
    elif pct_dynamic >= 0.5:
        direction = "dynamic retrieval RC5"
    else:
        direction = "hybrid RC5"

    out = {
        "generated_by": "scripts/tr7_dynamic_vs_static_classification.py",
        "num_wins": n, "classification_histogram": dict(hist),
        "static_compatible_now": static_now,
        "static_compatible_with_work": static_with_work,
        "dynamic_only": dynamic,
        "pct_static_compatible": pct_static_compatible, "pct_dynamic_only": pct_dynamic,
        "recommended_rc5_direction": direction,
        "records": records,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR7 dynamic vs static classification", "",
          f"- TR6 fresh wins: {n} | classes: {dict(hist)}",
          f"- static-compatible now: {static_now} | with work (allowlist/gate/schema): {static_with_work}",
          f"- dynamic-only: {dynamic}",
          f"- **% static-compatible: {pct_static_compatible:.0%} | % dynamic-only: {pct_dynamic:.0%}**",
          f"- **recommended RC5 direction: {direction}**", "",
          "| theorem | ns | static_coverage | dynamic_vs_static |", "|---|---|---|---|"]
    for r in sorted(records, key=lambda x: x["dynamic_vs_static_class"]):
        md.append(f"| `{r['full_name']}` | {r['namespace']} | {r['static_coverage_class']} | "
                  f"{r['dynamic_vs_static_class']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr7-dynstat] {dict(hist)}")
    print(f"[tr7-dynstat] static_compat={pct_static_compatible} dynamic={pct_dynamic} "
          f"direction={direction}")


if __name__ == "__main__":
    main()
