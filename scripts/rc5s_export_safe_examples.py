#!/usr/bin/env python3
"""RC5S Part 11 — export the safe dynamic examples (safety data, NOT ranker training).

One safe attempted program = one row, with theorem / tactic / lemma / policy status / timeout
status / outcome / attribution / safety class. Used to audit the hardened stage; the ranker is
NOT retrained here (only 3 positives — RC5H showed that hurts PR-AUC).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--b5", required=True)
    ap.add_argument("--b10")
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    plan = {t["full_name"]: t for t in json.load(open(_p(args.plan)))["theorems"]}
    b5 = {r["full_name"]: r for r in json.load(open(_p(args.b5)))["results"]}
    b10 = {}
    if args.b10 and os.path.exists(_p(args.b10)):
        b10 = {r["full_name"]: r for r in json.load(open(_p(args.b10)))["results"]}
    attr = {r["full_name"]: r for r in json.load(open(_p(args.attribution)))["records"]}

    rows = []
    for fn, t in plan.items():
        ns = t.get("namespace")
        runs = [b5.get(fn, {}), b10.get(fn, {})]
        outcome_by_tac, win_tac = {}, None
        killed = False
        for run in runs:
            if run.get("killed_by_timeout"):
                killed = True
            wp = run.get("winning_program") or {}
            if wp.get("tactic"):
                win_tac = wp["tactic"]
            for f in run.get("failures", []):
                outcome_by_tac.setdefault(f["tactic"], f.get("outcome"))
        acls = attr.get(fn, {}).get("classification")
        for p in t.get("programs_ranked", []):
            tac = p.get("tactic")
            lemmas = p.get("lemmas") or p.get("used_lemmas") or []
            is_win = (tac == win_tac)
            outcome = "solved" if is_win else outcome_by_tac.get(tac, "not_reached_or_failed")
            klass, allowed = G.classify_program(tac, ns)
            rows.append({
                "full_name": fn, "namespace": ns, "tactic": tac,
                "lemma": lemmas[0] if lemmas else None, "lemmas": lemmas,
                "rc5s_pattern": p.get("rc5s_pattern") or G.pattern_of(tac),
                "budget_stage": p.get("budget_stage"), "rank": p.get("rank"),
                "ranker_score": p.get("ranker_score"),
                "policy_status": "POLICY_ALLOWED" if allowed else klass,
                "timeout_status": "theorem_killed_by_timeout" if killed else "bounded",
                "outcome": outcome, "is_winning_program": is_win,
                "attribution": acls,
                "safety_class": ("SAFE_TRUE_DYNAMIC_WIN" if (is_win and acls == "SAFE_TRUE_DYNAMIC_WIN")
                                 else acls or "NO_WIN_SAFE_BUDGET"),
            })

    with open(_p(args.out_jsonl), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "generated_by": "scripts/rc5s_export_safe_examples.py",
        "num_examples": len(rows),
        "num_theorems": len(plan),
        "policy_status_histogram": dict(Counter(r["policy_status"] for r in rows)),
        "pattern_histogram": dict(Counter(r["rc5s_pattern"] for r in rows)),
        "outcome_histogram": dict(Counter(r["outcome"] for r in rows)),
        "winning_programs": sum(1 for r in rows if r["is_winning_program"]),
        "off_policy_examples": sum(1 for r in rows if r["policy_status"] != "POLICY_ALLOWED"),
        "note": "safety/audit data — NOT ranker training (only 3 positives; RC5H showed retrain hurts).",
    }
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5S safe dynamic examples summary", "",
          f"- examples (one safe attempt = one row): {len(rows)} over {len(plan)} theorems",
          f"- policy status: {summary['policy_status_histogram']}",
          f"- patterns: {summary['pattern_histogram']}",
          f"- **off-policy examples: {summary['off_policy_examples']}** (must be 0)",
          f"- winning programs: {summary['winning_programs']}",
          f"- NOTE: {summary['note']}"]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5s-export] examples={len(rows)} off_policy={summary['off_policy_examples']} "
          f"winning={summary['winning_programs']}")


if __name__ == "__main__":
    main()
