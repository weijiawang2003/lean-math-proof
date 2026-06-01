#!/usr/bin/env python3
"""RC5V3 Part 15 — export the dynamic examples (one safe attempt = one row). No ranker retrain.

Each row carries theorem / namespace / tactic / lemma / rank / score / budget slice / result /
attribution / safety flags / freshness / feature bucket. Merges B1/B3/B5 outcomes per program.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc5s_grammar as G

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _load(path):
    return {r["full_name"]: r for r in json.load(open(_p(path)))["results"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--dynamic-b1", required=True)
    ap.add_argument("--dynamic-b3", required=True)
    ap.add_argument("--dynamic-b5", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--yield-analysis", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    plan = {t["full_name"]: t for t in json.load(open(_p(args.plan)))["theorems"]}
    b1, b3, b5 = _load(args.dynamic_b1), _load(args.dynamic_b3), _load(args.dynamic_b5)
    attr = {r["full_name"]: r for r in json.load(open(_p(args.attribution)))["records"]}

    rows = []
    for fn, t in plan.items():
        ns = t.get("namespace")
        acls = attr.get(fn, {}).get("classification")
        budget_solved = attr.get(fn, {}).get("budget_solved")
        # merged per-program outcome across budgets
        outcome_by_tac, win_tac, killed_any = {}, None, False
        for store in (b1, b3, b5):
            r = store.get(fn)
            if not r:
                continue
            killed_any = killed_any or bool(r.get("killed_by_timeout"))
            wt = (r.get("winning_program") or {}).get("tactic")
            if wt:
                win_tac = wt
            for f in r.get("failures", []):
                outcome_by_tac.setdefault(f["tactic"], f.get("outcome"))
        feats = {k for k, v in (t.get("features") or {}).items() if v}
        for p in t.get("programs_ranked", []):
            tac = p.get("tactic")
            lemmas = p.get("used_lemmas") or p.get("lemmas") or []
            is_win = tac == win_tac
            rows.append({
                "full_name": fn, "namespace": ns, "tactic": tac,
                "lemma": lemmas[0] if lemmas else None, "rank": p.get("rank"),
                "score": p.get("ranker_score"),
                "budget_slice": p.get("budget_stage"),
                "rc5s_pattern": p.get("rc5s_pattern") or G.pattern_of(tac),
                "result": "solved" if is_win else outcome_by_tac.get(tac, "not_reached_or_failed"),
                "is_winning_program": is_win,
                "attribution": acls, "budget_solved": budget_solved,
                "policy_status": "POLICY_ALLOWED" if G.classify_program(tac, ns)[1] else "OFF_POLICY",
                "timeout_status": "theorem_killed" if killed_any else "bounded",
                "freshness_status": "strict_fresh",
                "feature_bucket": sorted(feats),
            })
    with open(_p(args.out_jsonl), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"generated_by": "scripts/rc5v3_export_examples.py", "num_examples": len(rows),
               "num_theorems": len(plan),
               "policy_status_histogram": dict(Counter(r["policy_status"] for r in rows)),
               "result_histogram": dict(Counter(r["result"] for r in rows)),
               "budget_slice_histogram": dict(Counter(r["budget_slice"] for r in rows)),
               "winning_programs": sum(1 for r in rows if r["is_winning_program"]),
               "off_policy_examples": sum(1 for r in rows if r["policy_status"] != "POLICY_ALLOWED"),
               "note": "safety/audit data — ranker NOT retrained."}
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V3 dynamic examples summary", "",
          f"- examples: {len(rows)} over {len(plan)} theorems",
          f"- policy status: {summary['policy_status_histogram']}",
          f"- budget slices: {summary['budget_slice_histogram']}",
          f"- **off-policy: {summary['off_policy_examples']}** | winning: {summary['winning_programs']}",
          f"- NOTE: {summary['note']}"]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v3-export] examples={len(rows)} off_policy={summary['off_policy_examples']} "
          f"winning={summary['winning_programs']} budgets={summary['budget_slice_histogram']}")


if __name__ == "__main__":
    main()
