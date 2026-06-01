#!/usr/bin/env python3
"""SF2 Part 5 — cluster-level analysis of the Set deep-dive.

Joins selected_cases, source_context (proof styles), probe_results (live gap
classifications), and the original failure_clusters. For each Set cluster it
aggregates: source proof-style distribution, probe success/failure counts, the
best probe family, the dominant per-theorem gap type, evidence, and a single
recommended next action.

Cluster gap-type rollup rule (conservative):
  - all members tactic_gap/routing_gap -> that type
  - mixed tactic/routing/search -> "mixed"
  - any member solved only by rw/source-inspired -> includes search_depth_gap
  - no member solved AND source proofs are rw-bridges over named lemmas that are
    NOT generic simp lemmas -> candidate "missing_lemma" (flagged for triage)
  - else "needs_more_eval"

Outputs:
  cluster_analysis.json
  cluster_analysis.md
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

CASES = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/selected_cases.json"
SRC = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/source_context.json"
PROBES = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/probe_results.json"
CLUSTERS = "project/evolve/experiments/sf2/out/frontier_expansion/failure_clusters.json"
OUT_JSON = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/cluster_analysis.json"
OUT_MD = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/cluster_analysis.md"


def rollup_gap(gaps, styles, any_solved):
    s = set(gaps)
    if not any_solved:
        # nothing in this cluster solved by probes; lean on source style
        if all(st in ("rw_bridge",) for st in styles) and styles:
            return "missing_lemma"  # candidate, triage will scrutinise
        return "needs_more_eval"
    s.discard(None)
    non_solved = {"needs_deeper_search", "needs_missing_lemma", "trace_insufficient"}
    solved_types = s - non_solved
    if solved_types == {"tactic_gap"}:
        return "tactic_gap"
    if solved_types == {"routing_gap"}:
        return "routing_gap"
    if solved_types == {"search_depth_gap"}:
        return "search_depth_gap"
    if solved_types:
        return "mixed"
    return "needs_more_eval"


def best_family(theorems, probe_results):
    fam = Counter()
    for r in probe_results:
        if r["full_name"] in theorems and r.get("winning_family"):
            fam[r["winning_family"]] += 1
    return fam.most_common(1)[0][0] if fam else None


def recommend(gap):
    return {
        "tactic_gap": "new_probe_family",
        "routing_gap": "NS23_relabel",
        "search_depth_gap": "new_probe_family",
        "missing_lemma": "SF3_candidate_lemma",
        "mixed": "new_probe_family",
        "needs_more_eval": "needs_more_eval",
        "junk": "ignore",
    }.get(gap, "needs_more_eval")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=CASES)
    p.add_argument("--source-context", default=SRC)
    p.add_argument("--probe-results", default=PROBES)
    p.add_argument("--clusters", default=CLUSTERS)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-md", default=OUT_MD)
    args = p.parse_args(argv)

    selected = json.load(open(args.cases))["selected"]
    src = {r["full_name"]: r for r in json.load(open(args.source_context))["cases"]}
    probes = json.load(open(args.probe_results))["results"]
    probe_by = {r["full_name"]: r for r in probes}
    clusters = {c["cluster_id"]: c for c in json.load(open(args.clusters))["clusters"]}

    # group selected theorems by cluster
    by_cluster = {}
    for c in selected:
        by_cluster.setdefault(c["cluster_id"], []).append(c["full_name"])

    out_clusters = []
    for cid, thms in by_cluster.items():
        meta = clusters.get(cid, {})
        styles = Counter()
        for t in thms:
            cl = (src.get(t, {}).get("classification") or {})
            styles[cl.get("proof_style", "n/a")] += 1
        gaps = [probe_by.get(t, {}).get("classification") for t in thms]
        solved = [probe_by.get(t, {}).get("solved_by_probe") for t in thms]
        any_solved = any(solved)
        nsolved = sum(1 for s in solved if s)
        gap = rollup_gap(gaps, list(styles.keys()), any_solved)
        evidence = []
        for t in thms:
            pr = probe_by.get(t, {})
            evidence.append({
                "theorem": t,
                "solved": pr.get("solved_by_probe"),
                "winning_probe": pr.get("winning_probe"),
                "gap": pr.get("classification"),
                "source_style": (src.get(t, {}).get("classification") or {}).get("proof_style"),
                "minimal_sufficient_probe": pr.get("minimal_sufficient_probe"),
            })
        prio = "high" if gap in ("tactic_gap", "search_depth_gap", "missing_lemma", "mixed") else "medium"
        out_clusters.append({
            "cluster_id": cid,
            "cluster_label": meta.get("top_candidate_family"),
            "size": meta.get("size"),
            "selected_theorems": thms,
            "source_proof_styles": dict(styles),
            "probe_successes": nsolved,
            "probe_failures": len(thms) - nsolved,
            "best_probe_family": best_family(thms, probes),
            "per_theorem_gaps": dict(zip(thms, gaps)),
            "likely_gap_type": gap,
            "evidence": evidence,
            "recommended_next_action": recommend(gap),
            "priority": prio,
        })

    out_clusters.sort(key=lambda c: (-(c["probe_successes"]), c["cluster_id"]))
    out = {"num_clusters": len(out_clusters),
           "gap_type_histogram": dict(Counter(c["likely_gap_type"] for c in out_clusters)),
           "total_probe_successes": sum(c["probe_successes"] for c in out_clusters),
           "clusters": out_clusters}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=2)

    L = ["# SF2 Set Cluster — Cluster-Level Analysis", ""]
    L.append(f"- clusters: {out['num_clusters']} | total probe successes: "
             f"{out['total_probe_successes']}")
    L.append(f"- gap-type histogram: `{out['gap_type_histogram']}`")
    L.append("")
    L.append("| cluster | label | size | sel | solved | best family | gap | action | prio |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in out_clusters:
        L.append(f"| `{c['cluster_id'].split('|')[1][:18]}/{c['cluster_id'].split('|')[2]}` "
                 f"| {c['cluster_label']} | {c['size']} | {len(c['selected_theorems'])} "
                 f"| {c['probe_successes']} | {c['best_probe_family']} | "
                 f"**{c['likely_gap_type']}** | {c['recommended_next_action']} | {c['priority']} |")
    L.append("")
    for c in out_clusters:
        L.append(f"## {c['cluster_id']}")
        L.append(f"- gap: **{c['likely_gap_type']}** | action: "
                 f"{c['recommended_next_action']} | styles: {c['source_proof_styles']}")
        for e in c["evidence"]:
            L.append(f"  - `{e['theorem']}` solved={e['solved']} gap={e['gap']} "
                     f"src={e['source_style']} win=`{e['winning_probe']}`")
        L.append("")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[sf2:analyze] clusters={out['num_clusters']} "
          f"gaps={out['gap_type_histogram']} -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
