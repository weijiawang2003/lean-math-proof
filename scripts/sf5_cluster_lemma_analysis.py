#!/usr/bin/env python3
"""SF5 Part 8 — cluster-level lemma analysis.

For each target cluster, combine retrieval (shared lemmas), live probe outcomes and
attribution to answer, for the large Set iff-equivalence cluster especially: do one
or two existing lemmas recur and close many targets, or does each target need its own
theorem-specific lemma?

Recommendation per cluster:
  add_router_retrieval  most members close via existing-lemma / routing retrieval
  lemma_synthesis       most members are TRUE_MISSING_BRIDGE_LEMMA
  deeper_search         most members are PROOF_DEPTH_GAP
  need_more_data        mixed / too small / no signal
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
    ap.add_argument("--targets", required=True)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--probe-results", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    targets = json.load(open(_p(args.targets)))
    retr = json.load(open(_p(args.retrieval)))
    attr = {r["full_name"]: r for r in json.load(open(_p(args.attribution)))["records"]}
    cluster_shared = retr.get("cluster_shared_lemmas", {})

    by_cluster = {}
    for t in targets:
        by_cluster.setdefault(t.get("cluster_id"), []).append(t["full_name"])

    clusters_out = []
    for cid, members in sorted(by_cluster.items(), key=lambda kv: -len(kv[1])):
        cls_counts = Counter(attr.get(m, {}).get("classification", "UNKNOWN")
                             for m in members)
        win_lemmas = Counter()
        for m in members:
            wl = attr.get(m, {}).get("winning_lemma")
            if wl:
                win_lemmas[wl] += 1
        existing = cls_counts.get("EXISTING_LEMMA_GAP", 0)
        routing = cls_counts.get("RETRIEVAL_ROUTING_GAP", 0)
        true_missing = cls_counts.get("TRUE_MISSING_BRIDGE_LEMMA", 0)
        depth = cls_counts.get("PROOF_DEPTH_GAP", 0)
        retrieval_wins = existing + routing
        size = len(members)

        if retrieval_wins >= max(1, (size + 1) // 2):
            rec = "add_router_retrieval"
        elif true_missing >= max(1, (size + 1) // 2):
            rec = "lemma_synthesis"
        elif depth >= max(1, (size + 1) // 2):
            rec = "deeper_search"
        else:
            rec = "need_more_data"

        clusters_out.append({
            "cluster_id": cid,
            "size": size,
            "targets": sorted(members),
            "shared_retrieved_lemmas": cluster_shared.get(cid, {})
                .get("shared_retrieved_lemmas", []),
            "winning_lemmas": [{"lemma": l, "count": c}
                               for l, c in win_lemmas.most_common()],
            "classification_histogram": dict(cls_counts),
            "retrieval_wins": retrieval_wins,
            "existing_lemma_gap_count": existing,
            "retrieval_routing_gap_count": routing,
            "true_missing_bridge_candidates": true_missing,
            "proof_depth_gap_count": depth,
            "recommendation": rec,
        })

    out = {
        "generated_by": "scripts/sf5_cluster_lemma_analysis.py",
        "num_clusters": len(clusters_out),
        "clusters": clusters_out,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# SF5 cluster lemma analysis", ""]
    for c in clusters_out:
        md += [f"## `{c['cluster_id']}` (size {c['size']}) -> **{c['recommendation']}**",
               "",
               f"- classifications: {c['classification_histogram']}",
               f"- retrieval wins: {c['retrieval_wins']} "
               f"(existing {c['existing_lemma_gap_count']}, routing "
               f"{c['retrieval_routing_gap_count']})",
               f"- true missing-bridge candidates: {c['true_missing_bridge_candidates']}",
               f"- proof-depth gaps: {c['proof_depth_gap_count']}", ""]
        if c["shared_retrieved_lemmas"]:
            md.append("Shared retrieved lemmas (≥2 targets):")
            for s in c["shared_retrieved_lemmas"][:10]:
                md.append(f"- `{s['lemma']}` — {s['appears_in_targets']} targets")
            md.append("")
        if c["winning_lemmas"]:
            md.append("Winning lemmas:")
            for w in c["winning_lemmas"]:
                md.append(f"- `{w['lemma']}` — closed {w['count']} target(s)")
            md.append("")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")

    print(f"[sf5-cluster] {len(clusters_out)} clusters")
    for c in clusters_out:
        print(f"  {c['cluster_id']} (n={c['size']}): {c['recommendation']} "
              f"[{c['classification_histogram']}]")


if __name__ == "__main__":
    main()
