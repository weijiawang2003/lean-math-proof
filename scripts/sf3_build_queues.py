#!/usr/bin/env python3
"""SF2/SF3 Part 7 — build the updated relabel queue and the SF3 candidate-lemma queue.

- relabel_queue_updated.jsonl  : TACTIC/PROBE candidates (what should go through
  NS23 minimal-sufficient relabeling), drawn from failure clusters with a concrete
  probe family.
- candidate_lemma_queue.jsonl  : MISSING-LEMMA candidates only. After the live
  singleton-iff analysis showed every dependency already exists, this queue holds
  the honest conclusion (no new lemma; the lever is a tactic/routing capability),
  plus any cluster that genuinely lacks an existing closing lemma.

Inputs (graceful if missing): failure_clusters.json, sf3 source_proof_analysis.json,
sf3 probe_results.json, candidate_lemmas.json.
"""

from __future__ import annotations

import argparse
import json
import os

FE = "project/evolve/experiments/sf2/out/frontier_expansion"
SF3 = "project/evolve/experiments/sf3/out/singleton_iff"


def load(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default=f"{FE}/failure_clusters.json")
    ap.add_argument("--sf3-analysis", default=f"{SF3}/source_proof_analysis.json")
    ap.add_argument("--sf3-probes", default=f"{SF3}/probe_results.json")
    ap.add_argument("--candidate-lemmas", default=f"{SF3}/candidate_lemmas.json")
    ap.add_argument("--relabel-out", default=f"{FE}/relabel_queue_updated.jsonl")
    ap.add_argument("--lemma-out", default=f"{SF3}/candidate_lemma_queue.jsonl")
    args = ap.parse_args(argv)

    clusters = (load(args.clusters, {}) or {}).get("clusters", [])
    analysis = load(args.sf3_analysis, {}) or {}
    probes = load(args.sf3_probes, {}) or {}
    n_solved = probes.get("num_solved", 0)

    os.makedirs(os.path.dirname(args.relabel_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.lemma_out), exist_ok=True)

    # ---- relabel (tactic/probe) queue ----
    relabel = []
    # the singleton-iff failure is the anchor entry
    relabel.append({
        "decl_name": "Multiset.toFinset_eq_singleton_iff",
        "namespace": "Multiset",
        "kind": "tactic_probe_candidate",
        "status": "OPEN_FAILURE",
        "probe_family": "split_iff_then_count_extensionality",
        "first_probe_to_try": "constructor <;> intro h <;> simp_all  (FAILS: max_recursion live)",
        "live_result": f"Part-3 ladder closed 0/{probes.get('num_probes', 0)}; "
                       "needs multi-step count-extensionality proof",
        "requires_ns23_relabel": True,
        "why_not_promote": "no single-shot probe closes it; the WX3 induction oracle "
                           "strictly hurts; the open lever is multi-step search + routing",
        "priority": "high",
    })
    for c in clusters:
        if c.get("next_action") != "probe" or c.get("priority") == "low":
            continue
        relabel.append({
            "cluster_id": c["cluster_id"],
            "kind": "tactic_probe_candidate",
            "namespace": c["namespace"],
            "representative_theorems": c["representative_theorems"],
            "probe_family": c.get("candidate_probe_family"),
            "likely_missing_capability": c.get("likely_missing_capability"),
            "requires_ns23_relabel": True,
            "priority": c["priority"],
            "note": "cluster-derived; needs live per-theorem eval + NS23 relabel before any claim",
        })
    with open(args.relabel_out, "w") as fh:
        for r in relabel:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- SF3 candidate-lemma queue (missing-lemma ONLY) ----
    lemma_q = []
    # honest anchor: the singleton failure does NOT need a new lemma
    lemma_q.append({
        "candidate_id": "singleton_iff_NO_MISSING_LEMMA",
        "source_failures": ["Multiset.toFinset_eq_singleton_iff"],
        "cluster_id": "Multiset|wx3_multiset_induction|iff|*",
        "lemma_template": None,
        "why_this_is_not_just_a_tactic": "It IS just a tactic/routing gap — the official "
            "proof uses only existing lemmas (count_*, toFinset_nsmul, toFinset_singleton, "
            "mem_toFinset). Verified live: no missing lemma.",
        "expected_downstream_utility": "n/a (no lemma to invent)",
        "novelty_risk": "duplicate (all deps exist)",
        "first_probe_to_try": "split-iff opener, then multi-step count-extensionality",
        "priority": "low_as_lemma_high_as_routing_fix",
    })
    # any cluster whose capability is genuinely lemma-shaped (none expected from this run)
    for c in clusters:
        if c.get("candidate_lemma_template"):
            lemma_q.append({
                "candidate_id": f"cluster::{c['cluster_id']}",
                "source_failures": c["representative_theorems"],
                "cluster_id": c["cluster_id"],
                "lemma_template": c["candidate_lemma_template"],
                "why_this_is_not_just_a_tactic": "cluster flagged a lemma-shaped gap",
                "expected_downstream_utility": c.get("expected_utility"),
                "novelty_risk": "unverified",
                "first_probe_to_try": c.get("candidate_probe_family"),
                "priority": c.get("priority", "medium"),
            })
    with open(args.lemma_out, "w") as fh:
        for r in lemma_q:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[sf3:queues] relabel(tactic/probe)={len(relabel)} -> {args.relabel_out}")
    print(f"[sf3:queues] candidate_lemma={len(lemma_q)} -> {args.lemma_out}")
    print(f"[sf3:queues] singleton-iff live solved={n_solved} (open failure, no missing lemma)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
