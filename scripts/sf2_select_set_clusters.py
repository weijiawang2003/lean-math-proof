#!/usr/bin/env python3
"""SF2 Part 1 — select target Set clusters from the 18 genuine RC1 failures.

Reads the frontier-expansion failure clusters, the authoritative RC1 eval matrix
(per-theorem `finished` field — the truth-repaired key), and the classified
frontier (tags). Emits `selected_cases.json`: 8-12 representative Set-namespace
failures spanning every high-priority Set cluster, with full per-theorem trace.

Selection rule (deterministic, documented):
  1. Only Set-namespace, genuine (non-junk) failures with a full trace.
  2. Cover every high-priority Set cluster: take the cluster's representatives.
  3. If the union exceeds the 12-cap, defer the least-informative duplicates
     (recorded in `deferred` with a reason) — never silently truncate.

Read-only except for the single output file under sf2/out/set_cluster_deep_dive/.
"""
from __future__ import annotations

import argparse
import json
import os

CLUSTERS = "project/evolve/experiments/sf2/out/frontier_expansion/failure_clusters.json"
EVAL = "project/evolve/experiments/sf1/out/real/eval_matrix_results.json"
FRONTIER = "project/evolve/experiments/sf1/out/real/classified_frontier.jsonl"
OUT = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/selected_cases.json"
CAP = 12

# Theorems deferred to keep within the 8-12 cap: each is subsumed by another
# representative of the same cluster/family, or is a one-off over-specialised goal.
DEFER = {
    "Set.ite_eq_of_subset_right": "duplicate of ite_eq_of_subset_left (same ext+by_cases proof)",
    "Set.ite_inter_compl_self": "subsumed by ite_inter_self family (rw bridge via ite_compl)",
    "Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le": "over-specialised giant simp; iff cluster already represented",
}


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", default=CLUSTERS)
    p.add_argument("--eval", default=EVAL)
    p.add_argument("--frontier", default=FRONTIER)
    p.add_argument("--out", default=OUT)
    p.add_argument("--cap", type=int, default=CAP)
    args = p.parse_args(argv)

    clusters = json.load(open(args.clusters))["clusters"]
    evald = json.load(open(args.eval))
    frontier = {r["decl_name"]: r for r in load_jsonl(args.frontier)}

    # per-theorem trace from authoritative eval (finished == truth)
    trace = {}
    for r in evald["results"]:
        for t in r.get("per_theorem", []):
            trace.setdefault(t["full_name"], t)  # first occurrence

    # build theorem -> cluster (Set, high prio, full trace, non-junk)
    thm_cluster = {}
    for c in clusters:
        if c["namespace"] != "Set":
            continue
        if c.get("priority") != "high" or not c.get("has_full_trace"):
            continue
        if "junk" in c["failure_type"]:
            continue
        for name in c["representative_theorems"]:
            thm_cluster.setdefault(name, c)  # dedupe (e.g. ssubset listed twice)

    selected, deferred = [], []
    for name, c in thm_cluster.items():
        tr = trace.get(name, {})
        fr = frontier.get(name, {})
        rec = {
            "full_name": name,
            "file_path": tr.get("file_path") or fr.get("file_path") or "Mathlib/Data/Set/Basic.lean",
            "cluster_id": c["cluster_id"],
            "cluster_label": c["top_candidate_family"],
            "rc1_status": "failed",
            "last_tried_tactics": [],  # eval records aggregate error, not per-step tactic strings
            "rc1_error_message": tr.get("error_message"),
            "rc1_num_steps": tr.get("num_steps"),
            "final_goal": None,  # filled live by the probe runner (initial_goal)
            "failure_type": c["failure_type"],
            "primary_goal_shape": c["primary_goal_shape"],
            "tags": fr.get("tags", []),
            "selection_reason": (
                f"high-priority Set {c['primary_goal_shape']} cluster "
                f"'{c['top_candidate_family']}'; full trace; "
                f"RC1 {tr.get('error_message', 'failed')}"),
        }
        if name in DEFER:
            rec["deferred_reason"] = DEFER[name]
            deferred.append(rec)
        else:
            selected.append(rec)

    # stable order: by cluster shape then name
    selected.sort(key=lambda r: (r["primary_goal_shape"], r["full_name"]))
    deferred.sort(key=lambda r: r["full_name"])

    if len(selected) > args.cap:
        # should not happen given DEFER, but keep honest
        overflow = selected[args.cap:]
        for r in overflow:
            r["deferred_reason"] = "cap overflow"
        deferred += overflow
        selected = selected[:args.cap]

    out = {
        "selection_rule": "Set-namespace high-priority full-trace genuine failures, "
                          "one+ representative per cluster, <=12 cap, deferrals recorded",
        "num_selected": len(selected),
        "num_deferred": len(deferred),
        "clusters_covered": sorted({r["cluster_id"] for r in selected}),
        "selected": selected,
        "deferred": deferred,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[sf2:select] selected={len(selected)} deferred={len(deferred)} "
          f"clusters={len(out['clusters_covered'])} -> {args.out}")
    for r in selected:
        print(f"  {r['full_name']:55s} {r['primary_goal_shape']:11s} {r['cluster_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
