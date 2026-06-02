#!/usr/bin/env python3
"""FLI0 Part 6 — select 20-40 lemma-invention seed cases for FLI1.

Deterministic scoring over the classified failures, then a diversity-capped greedy pick so the
seed set is not dominated by one pattern/namespace. Prioritizes clean + fresh + readable
statement/trace + invention-friendly namespaces + high-signal bridge patterns + "retrieval found
lemmas but didn't close" + "a similar theorem was solved".
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOOD_NS = {"List", "Multiset", "Finset", "Set", "Nat"}
HIGH_VALUE_PATTERNS = {"MEMBERSHIP_BRIDGE", "SINGLETON_CHARACTERIZATION", "DISJOINT_BRIDGE",
                       "SUBSET_BRIDGE", "MAP_FILTER_BIND_BRIDGE", "IFF_SPLIT",
                       "INDUCTION_GENERALIZATION", "EXTENSIONALITY_NEEDED"}
AVOID_PATTERNS = {"UNKNOWN_NAME_OR_IMPORT", "ORDER_STRUCTURE_GAP", "SIMP_LOOP_OR_RECURSION",
                  "NAT_ARITH_GAP", "LOW_SIGNAL", "NEEDS_REVIEW"}
PER_PATTERN_CAP = 8
PER_NS_CAP = 14


def _p(*a):
    return os.path.join(_REPO, *a)


def _score(c):
    s = 0.0
    if c.get("clean_failure"):
        s += 5
    if c.get("freshness_status") in ("strict_fresh", "soft_fresh"):
        s += 2
    if c.get("statement"):
        s += 2
        if len(c["statement"]) <= 240:  # readable, not giant
            s += 1
    if c.get("confidence") == "high":
        s += 3
    elif c.get("confidence") == "medium":
        s += 1.5
    prim = c.get("primary_pattern")
    if prim in HIGH_VALUE_PATTERNS:
        s += 3
    if prim in AVOID_PATTERNS:
        s -= 6
    root_ns = (c.get("namespace") or "").split(".")[0]
    if root_ns in GOOD_NS:
        s += 2
    if c.get("top_retrieved_lemmas"):
        s += 1  # retrieval found candidates but didn't close
    if c.get("similar_solved_theorem"):
        s += 1
    # penalize giant statements / unknown-name traces
    if c.get("statement") and len(c["statement"]) > 320:
        s -= 2
    if any("unknown" in (t or "") for t in c.get("failed_tactic_trace", [])):
        s -= 0.5
    return s


def _candidate_lemma_name(c):
    prim = c.get("primary_pattern")
    short = c["theorem"].split(".")[-1]
    ns = (c.get("namespace") or "").split(".")[0]
    base = {
        "MEMBERSHIP_BRIDGE": f"{ns}.mem_{short}_iff",
        "SINGLETON_CHARACTERIZATION": f"{ns}.{short}_singleton_iff",
        "DISJOINT_BRIDGE": f"{ns}.disjoint_{short}_iff",
        "SUBSET_BRIDGE": f"{ns}.{short}_subset_iff",
        "MAP_FILTER_BIND_BRIDGE": f"{ns}.mem_{short}",
        "IFF_SPLIT": f"{ns}.{short}_helper",
        "INDUCTION_GENERALIZATION": f"{ns}.{short}_induction_aux",
        "EXTENSIONALITY_NEEDED": f"{ns}.{short}_ext",
    }
    return base.get(prim, f"{ns}.{short}_aux")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    ap.add_argument("--target", type=int, default=40)
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(_p(args.patterns)) if l.strip()]
    for c in cases:
        c["_score"] = _score(c)
    # eligible pool: clean, has statement, high-value pattern, good namespace
    pool = [c for c in cases if c.get("clean_failure") and c.get("statement")
            and c.get("primary_pattern") in HIGH_VALUE_PATTERNS
            and (c.get("namespace") or "").split(".")[0] in GOOD_NS]
    pool.sort(key=lambda c: (-c["_score"], c["theorem"]))

    seeds, pat_count, ns_count = [], Counter(), Counter()
    for c in pool:
        if len(seeds) >= args.target:
            break
        prim = c["primary_pattern"]
        root_ns = (c.get("namespace") or "").split(".")[0]
        if pat_count[prim] >= PER_PATTERN_CAP or ns_count[root_ns] >= PER_NS_CAP:
            continue
        seeds.append(c)
        pat_count[prim] += 1
        ns_count[root_ns] += 1
    # backfill if diversity caps left us short of 20
    if len(seeds) < 20:
        for c in pool:
            if len(seeds) >= 20:
                break
            if c not in seeds:
                seeds.append(c)

    seed_records = []
    for i, c in enumerate(sorted(seeds, key=lambda c: (-c["_score"], c["theorem"])), 1):
        seed_records.append({
            "seed_id": f"FLI0-S{i:02d}",
            "theorem": c["theorem"], "namespace": c["namespace"],
            "statement": c["statement"], "source_stage": c["source_stage"],
            "freshness_status": c.get("freshness_status"),
            "failure_pattern": c["pattern_labels"], "primary_pattern": c["primary_pattern"],
            "confidence": c["confidence"],
            "why_selected": (f"clean fresh {c['namespace']} failure, primary {c['primary_pattern']} "
                             f"(conf {c['confidence']}); retrieval found "
                             f"{len(c.get('top_retrieved_lemmas', []))} lemmas but search "
                             f"({c.get('source_stage')}) did not close it"
                             + ("; a similar theorem was solved" if c.get('similar_solved_theorem') else "")),
            "residual_goal": None, "residual_goal_status": "MISSING",
            "top_retrieved_lemmas": c.get("top_retrieved_lemmas", [])[:5],
            "failed_tactics": c.get("failed_tactic_trace", [])[:8],
            "similar_solved_theorem": c.get("similar_solved_theorem"),
            "candidate_lemma_shape_nl": c["candidate_lemma_shape_nl"],
            "candidate_lemma_name_suggested": _candidate_lemma_name(c),
            "recommended_fli1_action": c["recommended_next_probe"],
            "selection_score": round(c["_score"], 2),
        })

    out = {"generated_by": "scripts/fli0_select_seed_cases.py",
           "target": args.target, "pool_size": len(pool), "num_seeds": len(seed_records),
           "seeds": seed_records}
    with open(_p(args.out_json), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    summary = {
        "generated_by": "scripts/fli0_select_seed_cases.py",
        "num_seeds": len(seed_records), "pool_size": len(pool),
        "by_pattern": dict(Counter(s["primary_pattern"] for s in seed_records).most_common()),
        "by_namespace": dict(Counter(s["namespace"] for s in seed_records).most_common()),
        "by_source_stage": dict(Counter(s["source_stage"] for s in seed_records)),
        "by_confidence": dict(Counter(s["confidence"] for s in seed_records)),
        "by_recommended_action": dict(Counter(s["recommended_fli1_action"] for s in seed_records).most_common()),
    }
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI0 seed-case selection summary", "",
          f"- **seeds selected: {summary['num_seeds']}** (target {args.target}) from pool "
          f"{summary['pool_size']}",
          f"- by pattern: {summary['by_pattern']}",
          f"- by namespace: {summary['by_namespace']}",
          f"- by source stage: {summary['by_source_stage']} | confidence: {summary['by_confidence']}",
          f"- recommended FLI1 actions: {summary['by_recommended_action']}", "",
          "## Seeds", "", "| id | theorem | ns | pattern | conf | action |",
          "|---|---|---|---|---|---|"]
    for s in seed_records:
        md.append(f"| {s['seed_id']} | `{s['theorem']}` | {s['namespace']} | "
                  f"{s['primary_pattern']} | {s['confidence']} | {s['recommended_fli1_action']} |")
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli0-seeds] seeds={len(seed_records)} pool={len(pool)} "
          f"patterns={summary['by_pattern']} ns={summary['by_namespace']}")


if __name__ == "__main__":
    main()
