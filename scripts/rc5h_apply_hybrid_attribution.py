#!/usr/bin/env python3
"""RC5H Part 9 — hybrid attribution.

For each dynamic success (across B5/B10/B20), classify using the bare controls already run in
the B5 stage (simp / simp_all / aesop / classical;aesop / exact L / simpa / simp[L]) plus the
RC2 and RC4-static outcomes:
  TRUE_HYBRID_DELTA            RC2 failed ∧ RC4 failed ∧ dynamic solved ∧ controls failed ∧ not source-specific
  STATIC_DUPLICATE             RC4 static already solved
  BASELINE_DUPLICATE           a bare control solved
  RC2_ALREADY_SOLVED           RC2 solved
  DYNAMIC_ONLY_BUT_SOURCE_SPECIFIC   theorem-specific rw/exact/tauto win (valid search, not production-safe)
  UNKNOWN_NAME_FAILURE / OPEN_FLAKE / NO_DYNAMIC_WIN
Only TRUE_HYBRID_DELTA counts as a credited hybrid win.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_SPECIFIC_FAMILIES = {"d1_exact", "d1_rw", "d1_tauto", "d2_rw_aesop", "d2_rw_simp"}


def _p(*a):
    return os.path.join(_REPO, *a)


def _merge_dynamic(b5, b10, b20):
    """Per theorem: the first success across budgets + the B5 controls."""
    merged = {}
    for stage in (b5, b10, b20):
        if not stage:
            continue
        for r in stage.get("results", []):
            fn = r["full_name"]
            cur = merged.get(fn)
            if cur is None:
                merged[fn] = dict(r)
            else:
                # keep controls from B5; upgrade success if a later budget solved
                if not cur.get("success") and r.get("success"):
                    cur.update({k: r[k] for k in ("success", "first_success_rank", "winning_program")})
                if r.get("controls") and not cur.get("controls"):
                    cur["controls"] = r["controls"]
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rc2-results", required=True)
    ap.add_argument("--static-results", required=True)
    ap.add_argument("--dynamic-b5", required=True)
    ap.add_argument("--dynamic-b10")
    ap.add_argument("--dynamic-b20")
    ap.add_argument("--program-plan", required=True)
    ap.add_argument("--policy")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rc2 = {r["full_name"]: r for r in json.load(open(_p(args.rc2_results)))["results"]}
    static = {r["full_name"]: r for r in json.load(open(_p(args.static_results)))["results"]}
    b5 = json.load(open(_p(args.dynamic_b5))) if args.dynamic_b5 and os.path.exists(_p(args.dynamic_b5)) else None
    b10 = json.load(open(_p(args.dynamic_b10))) if args.dynamic_b10 and os.path.exists(_p(args.dynamic_b10)) else None
    b20 = json.load(open(_p(args.dynamic_b20))) if args.dynamic_b20 and os.path.exists(_p(args.dynamic_b20)) else None
    dyn = _merge_dynamic(b5, b10, b20)
    plan = {t["full_name"]: t for t in json.load(open(_p(args.program_plan)))["theorems"]}

    records = []
    for fn, d in dyn.items():
        rc2_solved = rc2.get(fn, {}).get("status") == "solved"
        rc4_solved = static.get(fn, {}).get("status") == "solved"
        success = bool(d.get("success"))
        wp = d.get("winning_program") or {}
        fam = wp.get("family")
        control_wins = d.get("control_wins") or [c["tactic"] for c in d.get("controls", []) if c.get("solved")]
        setup_error = d.get("setup_error")
        budget_hit = d.get("budget")
        rank = d.get("first_success_rank")
        # source-specific: theorem-specific rw/exact/tauto, or the lemma is the theorem itself
        lemmas = wp.get("used_lemmas") or []
        source_specific = (fam in SOURCE_SPECIFIC_FAMILIES) or any(L == fn for L in lemmas)

        if not success:
            if setup_error and "exceeded" in (setup_error or ""):
                cls = "OPEN_FLAKE"
            elif d.get("timeout"):
                cls = "OPEN_FLAKE"
            else:
                cls = "NO_DYNAMIC_WIN"
        elif rc2_solved:
            cls = "RC2_ALREADY_SOLVED"
        elif rc4_solved:
            cls = "STATIC_DUPLICATE"
        elif control_wins:
            cls = "BASELINE_DUPLICATE"
        elif source_specific:
            cls = "DYNAMIC_ONLY_BUT_SOURCE_SPECIFIC"
        else:
            cls = "TRUE_HYBRID_DELTA"
        records.append({
            "full_name": fn, "namespace": d.get("namespace"),
            "rc2_solved": rc2_solved, "rc4_static_solved": rc4_solved,
            "dynamic_success": success, "winning_program": wp.get("tactic"),
            "winning_family": fam, "winning_lemmas": lemmas, "first_success_rank": rank,
            "budget": budget_hit, "control_wins": control_wins,
            "source_specific": source_specific, "classification": cls,
        })

    hist = Counter(r["classification"] for r in records)
    true_delta = [r for r in records if r["classification"] == "TRUE_HYBRID_DELTA"]
    by_ns = Counter(r["namespace"] for r in true_delta)
    by_fam = Counter(r["winning_family"] for r in true_delta)
    by_budget = Counter(r["budget"] for r in true_delta)
    out = {
        "generated_by": "scripts/rc5h_apply_hybrid_attribution.py",
        "num_dynamic_attempts": len(records),
        "classification_histogram": dict(hist),
        "true_hybrid_deltas": len(true_delta),
        "true_hybrid_delta_targets": [r["full_name"] for r in true_delta],
        "dynamic_wins_total": sum(1 for r in records if r["dynamic_success"]),
        "source_specific_wins": sum(1 for r in records if r["classification"] == "DYNAMIC_ONLY_BUT_SOURCE_SPECIFIC"),
        "true_delta_by_namespace": dict(by_ns), "true_delta_by_family": dict(by_fam),
        "true_delta_by_budget": {str(k): v for k, v in by_budget.items()},
        "records": records,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5H hybrid attribution", "",
          f"- dynamic attempts: {len(records)} | classifications: {dict(hist)}",
          f"- **TRUE_HYBRID_DELTA: {len(true_delta)}** {out['true_hybrid_delta_targets']}",
          f"- dynamic wins total: {out['dynamic_wins_total']} | source-specific: {out['source_specific_wins']}",
          f"- true delta by namespace: {dict(by_ns)} | by family: {dict(by_fam)} | "
          f"by budget: {out['true_delta_by_budget']}", "",
          "| theorem | ns | rc2 | rc4 | dyn | class | winning program |", "|---|---|---|---|---|---|---|"]
    for r in sorted(records, key=lambda x: (x["classification"] != "TRUE_HYBRID_DELTA", x["full_name"])):
        md.append(f"| `{r['full_name']}` | {r['namespace']} | {'S' if r['rc2_solved'] else 'F'} | "
                  f"{'S' if r['rc4_static_solved'] else 'F'} | {'S' if r['dynamic_success'] else 'F'} | "
                  f"{r['classification']} | `{r['winning_program'] or ''}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5h-attrib] {dict(hist)}")
    print(f"[rc5h-attrib] TRUE_HYBRID_DELTA={len(true_delta)} {out['true_hybrid_delta_targets']}")


if __name__ == "__main__":
    main()
