#!/usr/bin/env python3
"""TR3 Part 9 — analyze program families and retrieved-lemma usage.

Family rollup (tried / true-delta wins / baseline dups / parse errors / timeouts /
namespaces / clusters / promotion recommendation) and per-lemma usage (targets,
wins, clusters, classification). Promotion threshold (advisory, NOT applied here):
>=2 TRUE_*_DELTA over literal RC2, 0 off-gate, deterministic, generic, SX4-survived.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirmation", required=True)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--program-plan", required=True)
    ap.add_argument("--program-results", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--out-family-json", required=True)
    ap.add_argument("--out-family-md", required=True)
    ap.add_argument("--out-lemma-json", required=True)
    ap.add_argument("--out-lemma-md", required=True)
    args = ap.parse_args()

    pr = json.load(open(_p(args.program_results)))
    attr = {r["full_name"]: r
            for r in json.load(open(_p(args.attribution)))["records"]}
    plan = {t["full_name"]: t for t in json.load(open(_p(args.program_plan)))["theorems"]}

    # ---- family analysis ----
    fam = defaultdict(lambda: {"family": None, "depth": None, "tried": 0,
                               "true_delta_wins": 0, "solves": 0, "baseline_duplicates": 0,
                               "parse_errors": 0, "timeouts": 0, "unknown_name": 0,
                               "namespaces": set(), "clusters": set(),
                               "true_delta_targets": []})
    for r in pr["results"]:
        ns = r.get("namespace")
        cid = r.get("cluster_id")
        for run in r.get("ran", []):
            if run.get("skipped"):
                continue
            f = run["family"]
            d = fam[f]
            d["family"] = f
            d["depth"] = run.get("depth")
            d["tried"] += 1
            o = run.get("outcome")
            if o == "success":
                d["solves"] += 1
            elif o == "parse_error":
                d["parse_errors"] += 1
            elif o == "timeout":
                d["timeouts"] += 1
            elif o == "unknown_name":
                d["unknown_name"] += 1
            d["namespaces"].add(ns)
            d["clusters"].add(cid)
    # credited wins per family
    for fn, a in attr.items():
        if a.get("credited") and a.get("winning_family"):
            f = a["winning_family"]
            fam[f]["true_delta_wins"] += 1
            fam[f]["true_delta_targets"].append(fn)
            fam[f]["family"] = f
        # baseline duplicates: attribute control-win to no family; track separately

    def reco(d):
        if d["true_delta_wins"] >= 2:
            return "rc_candidate"
        if d["true_delta_wins"] == 1:
            return "experimental"
        if d["tried"] > 0 and (d["parse_errors"] + d["timeouts"]) < d["tried"]:
            return "training_only"
        return "reject"

    fam_out = []
    for f, d in sorted(fam.items(), key=lambda kv: (-kv[1]["true_delta_wins"], -kv[1]["tried"])):
        fam_out.append({
            "family": f, "depth": d["depth"], "tried": d["tried"],
            "solves": d["solves"], "true_delta_wins": d["true_delta_wins"],
            "true_delta_targets": sorted(d["true_delta_targets"]),
            "parse_errors": d["parse_errors"], "timeouts": d["timeouts"],
            "unknown_name": d["unknown_name"],
            "namespaces": sorted(x for x in d["namespaces"] if x),
            "clusters": sorted(x for x in d["clusters"] if x),
            "promotion_recommendation": reco(d),
        })

    json.dump({"generated_by": "scripts/tr3_analyze_results.py",
               "num_families": len(fam_out), "families": fam_out},
              open(_p(args.out_family_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR3 family analysis", "",
          "| family | depth | tried | solves | true_delta | parse_err | timeout | reco |",
          "|---|---|---|---|---|---|---|---|"]
    for d in fam_out:
        md.append(f"| {d['family']} | {d['depth']} | {d['tried']} | {d['solves']} | "
                  f"{d['true_delta_wins']} | {d['parse_errors']} | {d['timeouts']} | "
                  f"{d['promotion_recommendation']} |")
    open(_p(args.out_family_md), "w").write("\n".join(md) + "\n")

    # ---- lemma usage analysis ----
    used = defaultdict(lambda: {"used_in_targets": set(), "wins": set(), "clusters": set()})
    for r in pr["results"]:
        cid = r.get("cluster_id")
        for run in r.get("ran", []):
            for L in run.get("lemmas", []) or []:
                used[L]["used_in_targets"].add(r["full_name"])
                used[L]["clusters"].add(cid)
    for fn, a in attr.items():
        if a.get("credited"):
            for L in a.get("winning_lemmas", []) or []:
                used[L]["wins"].add(fn)

    lemma_out = []
    for L, d in sorted(used.items(), key=lambda kv: (-len(kv[1]["wins"]), -len(kv[1]["used_in_targets"]))):
        nwins = len(d["wins"])
        cls = ("useful_retrieval_lemma" if nwins >= 1
               else ("needs_review" if len(d["used_in_targets"]) >= 3 else "noise"))
        lemma_out.append({
            "lemma": L, "used_in_targets": sorted(d["used_in_targets"]),
            "num_used_in_targets": len(d["used_in_targets"]),
            "wins": sorted(d["wins"]), "num_wins": nwins,
            "clusters": sorted(x for x in d["clusters"] if x),
            "classification": cls,
        })
    json.dump({"generated_by": "scripts/tr3_analyze_results.py",
               "num_lemmas_used": len(lemma_out), "lemmas": lemma_out},
              open(_p(args.out_lemma_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR3 lemma usage analysis", "",
          f"- distinct lemmas used in programs: {len(lemma_out)}",
          f"- useful (>=1 win): {sum(1 for l in lemma_out if l['num_wins']>=1)}", "",
          "| lemma | used_in | wins | class |", "|---|---|---|---|"]
    for l in lemma_out[:40]:
        md.append(f"| `{l['lemma']}` | {l['num_used_in_targets']} | {l['num_wins']} | "
                  f"{l['classification']} |")
    open(_p(args.out_lemma_md), "w").write("\n".join(md) + "\n")

    print(f"[tr3-analyze] families={len(fam_out)} "
          f"(rc_candidate={sum(1 for d in fam_out if d['promotion_recommendation']=='rc_candidate')}, "
          f"experimental={sum(1 for d in fam_out if d['promotion_recommendation']=='experimental')}); "
          f"useful_lemmas={sum(1 for l in lemma_out if l['num_wins']>=1)}")


if __name__ == "__main__":
    main()
