#!/usr/bin/env python3
"""FLI1 Part 4 — normalize and cluster captured residual goals.

Parses each residual goal pp into (hyps, goal), normalizes the goal (strip universe suffixes /
inaccessible daggers, abstract type vars, α-rename locals), extracts relation symbols + container
operations + constants, and clusters by (namespace, pattern, main-relation, container-op).
Conservative — unrelated goals stay in separate clusters. Pattern is joined from the rerun plan.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = "project/evolve/experiments/fli1/cases/fli1_live_rerun_plan.json"

RELATIONS = ["↔", "⊆", "⊂", "∈", "∉", "=", "≠", "≤", "<", "→", "∃", "∀", "Disjoint"]
CONTAINER_OPS = ["biUnion", "iUnion", "sUnion", "biInter", "iInter", "image", "preimage",
                 "map", "filterMap", "filter", "bind", "card", "toFinset", "toList", "powerset",
                 "singleton", "insert", "erase", "Nonempty", "Subsingleton"]
_UNIV = re.compile(r"\bu_\d+\b|\bType (?:u_\d+|\*)?")
_DAGGER = re.compile(r"✝\d*")
_CONST = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:\.[A-Za-z][A-Za-z0-9]*)+)\b")


def _p(*a):
    return os.path.join(_REPO, *a)


def _split_goal(pp):
    """Return (hyp_block, goal_text) for the FIRST goal in a pp dump."""
    if not pp:
        return "", ""
    # multiple goals separated by blank lines / 'case'; take first goal's ⊢
    if "⊢" not in pp:
        return pp, ""
    # first ⊢ onward, up to the next 'case ' marker or blank-line+case
    head, _, rest = pp.partition("⊢")
    goal = rest.split("\ncase ")[0].strip()
    return head, goal


def _normalize_goal(goal):
    g = _DAGGER.sub("", goal)
    g = _UNIV.sub("Type", g)
    g = re.sub(r"\s+", " ", g).strip()
    return g


def _relations(goal):
    return [r for r in RELATIONS if r in goal]


def _container_ops(goal):
    found = []
    for op in CONTAINER_OPS:
        if re.search(rf"\b{re.escape(op)}\b", goal) or f".{op}" in goal:
            found.append(op)
    return found


def _constants(goal):
    return sorted(set(_CONST.findall(goal)))


def _main_relation(goal):
    for r in ("↔", "⊆", "Disjoint", "∈", "=", "→", "∃", "∀", "≤"):
        if r in goal:
            return r
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual-goals", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    plan = {s["seed_id"]: s for s in json.load(open(_p(PLAN)))["seeds"]}
    rows = [json.loads(l) for l in open(_p(args.residual_goals)) if l.strip()]
    cap = [r for r in rows if r["status"] == "captured" and r.get("residual_goals")]

    enriched = []
    for r in cap:
        seed = plan.get(r["seed_id"], {})
        goal_raw = r["residual_goals"][0] or ""
        _, goal = _split_goal(goal_raw)
        ng = _normalize_goal(goal)
        rels = _relations(goal)
        ops = _container_ops(goal)
        enriched.append({
            "seed_id": r["seed_id"], "theorem": r["theorem"], "namespace": r["namespace"],
            "pattern": seed.get("primary_pattern", "UNKNOWN"),
            "normalized_goal": ng, "main_relation": _main_relation(goal),
            "relations": rels, "container_ops": ops, "constants": _constants(goal),
            "prefix": r.get("last_successful_tactic_prefix", []),
        })

    clusters = defaultdict(list)
    for e in enriched:
        op = (e["container_ops"] or ["none"])[0]
        key = (e["namespace"].split(".")[0], e["pattern"], e["main_relation"], op)
        clusters[key].append(e)

    cluster_out = []
    for i, (key, members) in enumerate(sorted(clusters.items()), 1):
        ns, pat, rel, op = key
        const_counts = Counter(c for m in members for c in m["constants"])
        common = [c for c, n in const_counts.items() if n >= max(2, len(members) // 2)]
        fam = f"{ns}_{pat.split('_')[0].lower()}_{op}_{ {'↔':'iff','⊆':'subset','∈':'mem','=':'eq','→':'imp','Disjoint':'disjoint'}.get(rel, 'rel') }"
        cluster_out.append({
            "cluster_id": f"FLI1-C{i:02d}", "pattern": pat, "namespace": ns,
            "main_relation": rel, "container_op": op, "size": len(members),
            "representative_goal": members[0]["normalized_goal"],
            "common_constants": sorted(common),
            "candidate_lemma_family": fam,
            "seed_ids": [m["seed_id"] for m in members],
            "theorems": [m["theorem"] for m in members],
        })
    cluster_out.sort(key=lambda c: (-c["size"], c["cluster_id"]))

    out = {"generated_by": "scripts/fli1_normalize_and_cluster_goals.py",
           "num_captured": len(cap), "num_clusters": len(cluster_out),
           "clusters": cluster_out, "normalized_goals": enriched}
    with open(_p(args.out_json), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    summary = {"generated_by": "scripts/fli1_normalize_and_cluster_goals.py",
               "num_captured": len(cap), "num_clusters": len(cluster_out),
               "by_pattern": dict(Counter(c["pattern"] for c in cluster_out).most_common()),
               "by_namespace": dict(Counter(c["namespace"] for c in cluster_out).most_common()),
               "multi_member_clusters": sum(1 for c in cluster_out if c["size"] > 1),
               "largest_clusters": [(c["cluster_id"], c["candidate_lemma_family"], c["size"])
                                    for c in cluster_out[:6]]}
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI1 goal cluster summary", "",
          f"- captured goals: {summary['num_captured']} → clusters: {summary['num_clusters']} "
          f"(multi-member: {summary['multi_member_clusters']})",
          f"- by pattern: {summary['by_pattern']}",
          f"- by namespace: {summary['by_namespace']}", "",
          "| cluster | family | ns | rel | op | size | seeds |",
          "|---|---|---|---|---|---|---|"]
    for c in cluster_out:
        md.append(f"| {c['cluster_id']} | {c['candidate_lemma_family']} | {c['namespace']} | "
                  f"{c['main_relation']} | {c['container_op']} | {c['size']} | "
                  f"{','.join(c['seed_ids'])} |")
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli1-cluster] captured={len(cap)} clusters={len(cluster_out)} "
          f"multi={summary['multi_member_clusters']}")


if __name__ == "__main__":
    main()
