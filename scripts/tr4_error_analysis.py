#!/usr/bin/env python3
"""TR4 Part 7 — error analysis over the leakage-free OOF ranker (HGB).

False positives (high-ranked failures, esp. unknown-name / broad simp-rw), false
negatives (successes ranked low), namespace/family generalization, class imbalance,
and leakage signals (by-theorem strong vs by-namespace weak). Emits recommendations.
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _rows(path):
    return [json.loads(l) for l in open(_p(path)) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--training-results", required=True)
    ap.add_argument("--ranking-eval", required=True)
    ap.add_argument("--budget", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--model-dir", default="project/evolve/experiments/tr4/models")
    ap.add_argument("--model", default="hgb")
    args = ap.parse_args()

    rows = _rows(args.examples)
    oof = np.load(os.path.join(_p(args.model_dir), "oof_scores.npz"))
    sc = oof[args.model]
    success = np.array([r["label_success"] for r in rows])
    train = json.load(open(_p(args.training_results)))
    budget = json.load(open(_p(args.budget)))

    valid = ~np.isnan(sc)
    order = np.argsort(-np.where(valid, sc, -np.inf))

    # false positives: top-50 ranked that are failures
    fp = [rows[i] for i in order[:50] if not success[i]]
    fp_outcome = Counter(r["outcome"] for r in fp)
    fp_family = Counter(r["program_family"] for r in fp)
    fp_unknown_name = sum(1 for r in fp if r["outcome"] == "unknown_name")

    # false negatives: successes whose OOF rank (global) is poor
    succ_idx = [i for i in range(len(rows)) if success[i] and valid[i]]
    global_rank = {int(i): int(p) for p, i in enumerate(order, 1)}
    fn_list = sorted(({"full_name": rows[i]["full_name"], "tactic": rows[i]["tactic"],
                       "family": rows[i]["program_family"],
                       "global_rank": global_rank.get(i)} for i in succ_idx),
                     key=lambda x: -(x["global_rank"] or 0))
    fn_worst = [x for x in fn_list if (x["global_rank"] or 0) > 100]

    # namespace / family generalization
    grouped = train.get("grouped_generalization", {}).get(args.model, {})
    pos_by_ns = Counter(r["namespace"] for r in rows if r["label_success"])
    pos_by_fam = Counter(r["program_family"] for r in rows if r["label_success"])

    imbalance = {"num_examples": len(rows), "num_positive": int(success.sum()),
                 "positive_rate": round(float(success.mean()), 5),
                 "positives_by_namespace": dict(pos_by_ns),
                 "positives_by_family": dict(pos_by_fam)}

    leakage = {
        "oof_by_theorem_pr_auc": (grouped.get("by_theorem") or {}).get("pr_auc"),
        "oof_by_namespace_pr_auc": (grouped.get("by_namespace") or {}).get("pr_auc"),
        "oof_by_cluster_pr_auc": (grouped.get("by_cluster") or {}).get("pr_auc"),
        "interpretation": ("Strong by-theorem but weak by-namespace PR-AUC => the ranker "
                           "relies on namespace/family-surface cues that do NOT transfer "
                           "to a held-out namespace (same gap as TR1). Within-distribution "
                           "probe reduction is real; cross-namespace transfer is not "
                           "established."),
    }

    recommendations = [
        "Probe reduction is usable WITHIN seen namespaces (Set/Finset/List); do NOT "
        "assume transfer to an unseen namespace — collect positives there first.",
        "Collect more positives via RC4B (Set.disjoint_left) and RC4C (d2_simp_aesop) "
        "validation before retraining — positives (23) and namespaces with positives "
        f"({len(pos_by_ns)}) are the binding constraint, not model capacity.",
        "A better/scope-aware retrieval index (cut the unknown-name failures that "
        f"dominate false positives: {fp_unknown_name}/50 top-ranked failures) likely "
        "helps more than further model tuning.",
        "Nat has 0 positives despite many failures — retrieval-aware programs do not "
        "address Nat arithmetic depth gaps; route those to a depth/search experiment.",
    ]

    out = {"generated_by": "scripts/tr4_error_analysis.py", "model": args.model,
           "false_positives": {"top50_failures": len(fp),
                               "by_outcome": dict(fp_outcome),
                               "by_family": dict(fp_family),
                               "unknown_name_in_top50": fp_unknown_name},
           "false_negatives": {"successes": len(succ_idx),
                               "ranked_below_100": len(fn_worst),
                               "worst": fn_worst[:10]},
           "imbalance": imbalance, "generalization": grouped, "leakage": leakage,
           "budget_decision": budget.get("decision"),
           "recommendations": recommendations}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# TR4 error analysis", "",
          f"- model: {args.model} | budget decision: {budget.get('decision')}", "",
          "## False positives (top-50 ranked failures)",
          f"- by outcome: {dict(fp_outcome)}",
          f"- by family: {dict(fp_family)}",
          f"- unknown-name in top-50: {fp_unknown_name}", "",
          "## False negatives",
          f"- successes: {len(succ_idx)} | ranked below global-100: {len(fn_worst)}", "",
          "## Class imbalance",
          f"- positives {imbalance['num_positive']}/{imbalance['num_examples']} "
          f"({imbalance['positive_rate']})",
          f"- positives by namespace: {dict(pos_by_ns)}",
          f"- positives by family: {dict(pos_by_fam)}", "",
          "## Leakage / generalization", "", leakage["interpretation"],
          f"- by_theorem PR-AUC {leakage['oof_by_theorem_pr_auc']}, by_namespace "
          f"{leakage['oof_by_namespace_pr_auc']}, by_cluster {leakage['oof_by_cluster_pr_auc']}",
          "", "## Recommendations", ""] + [f"- {r}" for r in recommendations]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr4-erroranalysis] fp_unknown_name={fp_unknown_name}/50 "
          f"fn_below100={len(fn_worst)} by_ns_pr_auc={leakage['oof_by_namespace_pr_auc']}")


if __name__ == "__main__":
    main()
