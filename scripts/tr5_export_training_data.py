#!/usr/bin/env python3
"""TR5 Part 10 — export live program-level training examples.

One attempted program (across B5/B10/B20) = one row, in the TR4 schema (so it can augment
the TR4 dataset). Labels: TRUE_RANKER_DELTA / TRUE_RC4B_EVIDENCE / TRUE_RC4C_EVIDENCE /
TRUE_RC4A_REPRODUCTION / RANKER_FALSE_POSITIVE / NO_WIN_UNDER_BUDGET / BASELINE_DUPLICATE /
PRODUCTION_SUBSUMED / OPEN_FLAKE. label_success = solved live; label_credit = solved AND
theorem credited. Does NOT overwrite TR4 data.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tr5_score as S

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _load(path):
    fp = _p(path) if not os.path.isabs(path) else path
    return json.load(open(fp)) if os.path.exists(fp) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked-plan", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--b5", default="project/evolve/experiments/tr5/out/tr5_b5_live_results.json")
    ap.add_argument("--b10", default="project/evolve/experiments/tr5/out/tr5_b10_live_results.json")
    ap.add_argument("--b20", default="project/evolve/experiments/tr5/out/tr5_b20_live_results.json")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    plan = {t["full_name"]: t for t in _load(args.ranked_plan)["theorems"]}
    attr = {r["full_name"]: r for r in _load(args.attribution)["records"]}
    b5 = _load(args.b5)
    b10 = _load(args.b10)
    b20 = _load(args.b20)
    # goal/program lookup from plan
    prog_meta = {}
    goal_of = {}
    for fn, t in plan.items():
        for p in t.get("programs_ranked", []):
            prog_meta[(fn, p["tactic"])] = p
    tr3_plan = {t["full_name"]: t for t in
                (_load("project/evolve/experiments/tr3/out/tr3_depth_program_plan.json") or {}).get("theorems", [])}
    for fn, t in tr3_plan.items():
        goal_of[fn] = t.get("goal_text")

    # gather attempted programs from all budgets
    attempts = []   # (full_name, ns, ran_record, budget)
    def collect(results, budget):
        for r in results:
            fn = r["full_name"]
            ns = r.get("namespace")
            # b5 stores winning + failures separately; reconstruct attempted set:
            # winning program (if any) + failures
            seen = []
            wp = r.get("winning_program")
            if wp:
                seen.append({"tactic": wp["tactic"], "rank": wp["rank"], "family": wp.get("family"),
                             "depth": wp.get("depth"), "used_lemmas": wp.get("used_lemmas", []),
                             "ranker_score": wp.get("ranker_score"), "outcome": "success",
                             "solved": True})
            for f in r.get("failures", []):
                pm = prog_meta.get((fn, f["tactic"]), {})
                seen.append({"tactic": f["tactic"], "rank": f.get("rank"), "family": f.get("family"),
                             "depth": pm.get("depth"), "used_lemmas": pm.get("used_lemmas", []),
                             "ranker_score": pm.get("ranker_score"), "outcome": f.get("outcome"),
                             "solved": False})
            for s in seen:
                attempts.append((fn, ns, s, budget))
    if b5:
        collect(b5["results"], 5)
    if b10:
        collect(b10.get("new_results", []), 10)
    if b20:
        collect(b20.get("new_results", []), 20)

    rows = []
    for fn, ns, s, budget in attempts:
        a = attr.get(fn, {})
        cls = a.get("classification")
        solved = s["solved"]
        if solved and cls in ("TRUE_RANKER_DELTA", "TRUE_RC4A_REPRODUCTION"):
            if a.get("rc4b_evidence"):
                label = "TRUE_RC4B_EVIDENCE"
            elif a.get("rc4c_evidence"):
                label = "TRUE_RC4C_EVIDENCE"
            else:
                label = cls
        elif solved and cls == "BASELINE_DUPLICATE":
            label = "BASELINE_DUPLICATE"
        elif solved and cls == "PRODUCTION_SUBSUMED":
            label = "PRODUCTION_SUBSUMED"
        elif not solved and s.get("rank") == 1:
            label = "RANKER_FALSE_POSITIVE"
        elif not solved:
            label = "NO_WIN_UNDER_BUDGET"
        else:
            label = "OPEN_FLAKE"
        credited = label in ("TRUE_RANKER_DELTA", "TRUE_RC4A_REPRODUCTION",
                             "TRUE_RC4B_EVIDENCE", "TRUE_RC4C_EVIDENCE")
        goal = goal_of.get(fn)
        base = S.build_row(fn, goal, ns, s["tactic"], s.get("used_lemmas"), s.get("family"),
                           s.get("depth") or 1, source="tr5")
        base.update({
            "source": "tr5", "rc2_status": "failed", "budget": budget,
            "rank": s.get("rank"), "ranker_score": s.get("ranker_score"),
            "program_family": s.get("family"), "program_depth": s.get("depth") or 1,
            "outcome": s["outcome"], "tr5_label": label,
            "label_success": 1 if solved else 0,
            "label_credit": 1 if credited else 0,
        })
        rows.append(base)

    os.makedirs(os.path.dirname(_p(args.out_jsonl)), exist_ok=True)
    with open(_p(args.out_jsonl), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    succ = sum(r["label_success"] for r in rows)
    cred = sum(r["label_credit"] for r in rows)
    fp = sum(1 for r in rows if r["tr5_label"] == "RANKER_FALSE_POSITIVE")
    by_label = Counter(r["tr5_label"] for r in rows)
    by_ns = Counter(r["namespace"] for r in rows)
    by_fam = Counter(r["program_family"] for r in rows)
    pos_by_fam = Counter(r["program_family"] for r in rows if r["label_success"])
    # how many improve TR4: credited positives are the scarce class TR4 needs
    improves_tr4 = cred
    summary = {
        "generated_by": "scripts/tr5_export_training_data.py",
        "num_examples": n, "success_positives": succ, "credit_positives": cred,
        "false_positives": fp,
        "label_histogram": dict(by_label),
        "by_namespace": dict(by_ns), "by_family": dict(by_fam),
        "success_by_family": dict(pos_by_fam),
        "examples_that_improve_tr4": improves_tr4,
        "note": "TR4 data NOT overwritten; these augment it. Credit positives are the scarce "
                "class (TR4 had 22) — each live-verified credit broadens the ranker.",
    }
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR5 training delta", "",
          f"- attempted programs (examples): **{n}**",
          f"- success positives: **{succ}** | credit positives: **{cred}** | false positives: {fp}",
          f"- label histogram: {dict(by_label)}",
          f"- by namespace: {dict(by_ns)}",
          f"- success by family: {dict(pos_by_fam)}",
          f"- examples that improve TR4 (credit positives): **{improves_tr4}**"]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[tr5-export] {n} examples, success={succ}, credit={cred}, fp={fp}, labels={dict(by_label)}")


if __name__ == "__main__":
    main()
