#!/usr/bin/env python3
"""FLI2 Part 7 — mine reusable lemma-deployment rules from the rescue attribution.

Groups TRUE_RETRIEVAL_GAP_RESCUE (and PARTIAL_PROGRESS) cases by (namespace, lemma-family,
template) into candidate DEPLOYMENT_RULEs, tallies false positives (same-family actions that did
not rescue), assigns risk + promotion_status. Discovery only — nothing is promoted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _lemma_family(L):
    """Coarse family token from a lemma name, e.g. Finset.card_le_one → Finset.card_*."""
    if not L:
        return "closer"
    ns = L.split(".")[0] if "." in L else ""
    short = L.split(".")[-1].lower()
    for key in ("card", "disjoint", "mem", "subset", "singleton", "map", "filter", "bind",
                "biunion", "iunion", "image", "tofinset", "insert", "nonempty"):
        if key in short:
            return f"{ns}.{key}_*" if ns else f"{key}_*"
    return f"{ns}.{short.split('_')[0]}_*" if ns else short


def _rule_name(ns, fam):
    base = fam.split(".")[-1].replace("_*", "").upper()
    return f"{ns.upper()}_{base}_BRIDGE" if ns else f"{base}_BRIDGE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--actions", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    attr = [json.loads(l) for l in open(_p(args.attribution)) if l.strip()]
    true_r = [r for r in attr if r["classification"] == "TRUE_RETRIEVAL_GAP_RESCUE"]
    partials = [r for r in attr if r["classification"] == "PARTIAL_PROGRESS"]

    # group rescues by (namespace, lemma_family)
    groups = defaultdict(lambda: {"rescues": [], "partials": [], "templates": set(), "lemmas": set()})
    for r in true_r:
        ns = (r["namespace"] or "").split(".")[0]
        fam = _lemma_family(r["lemma"])
        g = groups[(ns, fam)]
        g["rescues"].append(r)
        g["templates"].add(r["template"])
        g["lemmas"].add(r["lemma"])
    for r in partials:
        ns = (r["namespace"] or "").split(".")[0]
        fam = _lemma_family(r["lemma"])
        groups[(ns, fam)]["partials"].append(r)

    # false positives: same (ns, family) actions that fired but did NOT rescue
    fp = defaultdict(list)
    for r in attr:
        if r["classification"] in ("CONTROL_DUPLICATE", "NO_RESCUE", "NEEDS_REVIEW"):
            ns = (r["namespace"] or "").split(".")[0]
            fam = _lemma_family(r["lemma"])
            fp[(ns, fam)].append(r["tactic"])

    rules = []
    for (ns, fam), g in sorted(groups.items()):
        n_resc = len(g["rescues"])
        n_part = len(g["partials"])
        n_fp = len(fp.get((ns, fam), []))
        if n_resc >= 1:
            denom = n_resc + n_fp
            fp_rate = round(n_fp / denom, 3) if denom else 0.0
            risk = "low" if fp_rate <= 0.5 and n_resc >= 1 else "medium"
            status = "candidate" if (n_resc >= 1 and fp_rate <= 0.6) else "needs_more_data"
        elif n_part >= 1:
            risk, status, fp_rate = "medium", "needs_more_data", None
        else:
            continue
        rec_actions = sorted(g["templates"]) or ["SIMPLE_SIMP", "SIMP_AESOP"]
        rules.append({
            "rule_id": _rule_name(ns, fam),
            "pattern": (g["rescues"] or g["partials"])[0].get("expected_pattern"),
            "namespace": ns,
            "trigger_conditions": [f"theorem/residual in {ns}",
                                   f"retrieved lemma in family {fam}"],
            "lemma_family": fam,
            "recommended_actions": rec_actions,
            "supporting_rescues": [{"theorem": r["theorem"], "lemma": r["lemma"],
                                    "tactic": r["tactic"]} for r in g["rescues"]],
            "partial_progress_cases": [{"theorem": r["theorem"], "lemma": r["lemma"]}
                                       for r in g["partials"]],
            "false_positive_cases": fp.get((ns, fam), [])[:8],
            "num_rescues": n_resc, "num_partials": n_part, "num_false_positives": n_fp,
            "false_positive_rate": fp_rate, "risk": risk, "promotion_status": status,
        })
    rules.sort(key=lambda r: (-r["num_rescues"], -r["num_partials"], r["rule_id"]))

    out = {"generated_by": "scripts/fli2_mine_deployment_rules.py",
           "num_rules": len(rules),
           "candidate_rules": sum(1 for r in rules if r["promotion_status"] == "candidate"),
           "rules": rules}
    with open(_p(args.out_json), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    summary = {"generated_by": "scripts/fli2_mine_deployment_rules.py",
               "num_rules": len(rules),
               "candidate_rules": out["candidate_rules"],
               "rule_ids": [r["rule_id"] for r in rules],
               "total_supporting_rescues": sum(r["num_rescues"] for r in rules)}
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI2 deployment rule summary", "",
          f"- mined rules: {summary['num_rules']} (candidate: {summary['candidate_rules']})",
          f"- supporting rescues total: {summary['total_supporting_rescues']}", "",
          "| rule | ns | family | actions | rescues | partials | FP | risk | status |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rules:
        md.append(f"| {r['rule_id']} | {r['namespace']} | {r['lemma_family']} | "
                  f"{','.join(r['recommended_actions'])} | {r['num_rescues']} | {r['num_partials']} | "
                  f"{r['num_false_positives']} | {r['risk']} | {r['promotion_status']} |")
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli2-rules] rules={len(rules)} candidate={out['candidate_rules']} "
          f"ids={summary['rule_ids']}")


if __name__ == "__main__":
    main()
