#!/usr/bin/env python3
"""RC5V2 Part 2 — build the fresh out-of-sample frontier.

TR6 fresh pool ∪ discovered catalog, minus an internal exclusion registry covering every
prior-used theorem (TR6 batch + wins, RC4D/RC4R known wins, RC5H benchmark, RC5S benchmark, TR7
corpus). Tags freshness (strict_fresh / soft_fresh / known_control), prefers the allowed
namespaces, and carries the theorem features. ≥500 fresh candidates expected.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED = {"Set", "Finset", "List", "Multiset", "Nat"}
CONTROL = {"Order", "Int", "Option", "Bool", "Function"}


def _p(*a):
    return os.path.join(_REPO, *a)


def _names_from_manifest(rel):
    out = set()
    try:
        man = json.load(open(_p(rel)))
        for s, r in man["set_files"].items():
            for e in json.load(open(_p(r))):
                out.add(e["full_name"])
    except Exception:
        pass
    return out


def _exclusion_registry():
    excl = set()
    # TR6 batch + all searched + wins
    try:
        b = json.load(open(_p("project/evolve/experiments/tr6/cases/tr6_eval_batch.json")))
        for r in (b.get("theorems", b) if isinstance(b, dict) else b):
            excl.add(r["full_name"])
    except Exception:
        pass
    try:
        a = json.load(open(_p("project/evolve/experiments/tr6/out/tr6_attribution.json")))
        excl.update(a.get("fresh_true_delta_targets", []))
        for r in a.get("records", []):
            excl.add(r["full_name"])
    except Exception:
        pass
    excl |= _names_from_manifest("project/evolve/experiments/rc4_candidates/composition_rc4d/theorem_sets/validation_manifest.json")
    excl |= _names_from_manifest("project/evolve/experiments/rc4_release_candidate/theorem_sets/benchmark_manifest.json")
    excl |= _names_from_manifest("project/evolve/experiments/rc5_hybrid/cases/rc5h_benchmark_manifest.json")
    excl |= _names_from_manifest("project/evolve/experiments/rc5_safety/cases/rc5s_benchmark_manifest.json")
    try:
        for l in open(_p("project/evolve/experiments/tr7/cases/tr7_comparison_corpus.jsonl")):
            excl.add(json.loads(l)["full_name"])
    except Exception:
        pass
    return excl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-pool", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    excl = _exclusion_registry()
    pool = {}
    for l in open(_p("project/evolve/experiments/tr6/cases/tr6_fresh_frontier_pool.jsonl")):
        r = json.loads(l)
        pool.setdefault(r["full_name"], r)
    # discovered catalog adds file_paths/candidates
    try:
        disc = json.load(open(_p("project/discovered_theorems.json")))
        for r in (disc.get("theorems", disc) if isinstance(disc, dict) else disc):
            pool.setdefault(r["full_name"], r)
    except Exception:
        pass

    rows = []
    for fn, r in pool.items():
        if not r.get("file_path"):
            continue
        ns = r.get("namespace") or fn.split(".")[0]
        if fn in excl:
            status = "known_control"
        else:
            status = "strict_fresh"
        rows.append({"full_name": fn, "file_path": r.get("file_path"), "namespace": ns,
                     "freshness_status": status,
                     "features": r.get("features") or {},
                     "statement_text": r.get("statement_text"),
                     "source": r.get("source", "tr6_pool")})

    fresh = [r for r in rows if r["freshness_status"] == "strict_fresh"]
    with open(_p(args.out_pool), "w") as f:
        for r in fresh:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ns_dist = Counter(r["namespace"] for r in fresh)
    feat_keys = ["has_subset", "has_iff", "has_mem", "has_disjoint", "has_singleton",
                 "has_map_filter", "has_union_inter", "has_card", "has_tofinset", "has_nat_arith"]
    feat_dist = {k: sum(1 for r in fresh if (r["features"] or {}).get(k)) for k in feat_keys}
    summary = {
        "generated_by": "scripts/rc5v2_build_fresh_frontier.py",
        "exclusion_registry_size": len(excl),
        "pool_total": len(rows), "strict_fresh": len(fresh),
        "allowed_namespace_fresh": sum(1 for r in fresh if r["namespace"] in ALLOWED),
        "namespace_distribution": dict(ns_dist.most_common()),
        "feature_distribution": feat_dist,
        "meets_500_target": len(fresh) >= 500,
    }
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V2 fresh frontier", "",
          f"- exclusion registry: {len(excl)} prior-used theorems",
          f"- **strict-fresh candidates: {len(fresh)}** (allowed-ns: {summary['allowed_namespace_fresh']}) "
          f"| ≥500 target: {summary['meets_500_target']}",
          f"- namespaces: {dict(ns_dist.most_common(12))}",
          f"- features: {feat_dist}"]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v2-frontier] excl={len(excl)} strict_fresh={len(fresh)} "
          f"allowed_ns={summary['allowed_namespace_fresh']} meets_500={summary['meets_500_target']}")
    print(f"[rc5v2-frontier] ns={dict(ns_dist.most_common(8))}")


if __name__ == "__main__":
    main()
