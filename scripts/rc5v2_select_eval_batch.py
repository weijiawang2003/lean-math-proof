#!/usr/bin/env python3
"""RC5V2 Part 3 — select the stratified evaluation batch (~240).

Stratified by namespace (Set 45 / Finset 55 / List 45 / Multiset 45 / Nat 35 / Order-Other 15)
with focused dynamic-tail slices preferred (Finset image/subset/biUnion, Multiset bind/add/
disjoint, List forall/mem/map/filter, Set subset_pair/disjoint/singleton, Nat light arithmetic).
Deterministic ordering (no RNG).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUOTA = {"Set": 45, "Finset": 55, "List": 45, "Multiset": 45, "Nat": 35}
OTHER_QUOTA = 15
# per-namespace dynamic-tail token preferences (name/statement substrings)
TAIL_TOKENS = {
    "Finset": ["image", "subset", "biunion", "disjoint", "map", "filter"],
    "Multiset": ["bind", "add", "disjoint", "map", "filter", "mem"],
    "List": ["forall", "mem", "map", "filter", "all", "bind"],
    "Set": ["subset_pair", "subset", "disjoint", "singleton", "image", "mem"],
    "Nat": ["sqrt", "div", "mod", "dvd", "succ", "le", "lt"],
}


def _p(*a):
    return os.path.join(_REPO, *a)


def _tail_score(r, ns):
    blob = (r["full_name"] + " " + (r.get("statement_text") or "")).lower()
    toks = TAIL_TOKENS.get(ns, [])
    return sum(1 for t in toks if t in blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out-batch", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    ap.add_argument("--batch-size", type=int, default=240)
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(_p(args.pool))]
    by_ns = defaultdict(list)
    for r in pool:
        by_ns[r["namespace"]].append(r)

    chosen, seen = [], set()

    def take(ns, n, candidates):
        # prefer high dynamic-tail score, then lexical (deterministic)
        cands = sorted(candidates, key=lambda r: (-_tail_score(r, ns), r["full_name"]))
        k = 0
        for r in cands:
            if r["full_name"] in seen:
                continue
            seen.add(r["full_name"])
            chosen.append({**r, "stratum": ns, "tail_score": _tail_score(r, ns)})
            k += 1
            if k >= n:
                break
        return k

    for ns, n in QUOTA.items():
        take(ns, n, by_ns.get(ns, []))
    # Order/Other controls
    other = [r for ns in by_ns for r in by_ns[ns] if ns not in QUOTA]
    take("Other", OTHER_QUOTA, other)

    # top up to batch_size from remaining allowed-ns by tail score if short
    if len(chosen) < args.batch_size:
        rest = sorted((r for ns in QUOTA for r in by_ns.get(ns, []) if r["full_name"] not in seen),
                      key=lambda r: (-_tail_score(r, r["namespace"]), r["full_name"]))
        for r in rest:
            if len(chosen) >= args.batch_size:
                break
            seen.add(r["full_name"])
            chosen.append({**r, "stratum": r["namespace"], "tail_score": _tail_score(r, r["namespace"])})

    chosen = chosen[:args.batch_size]
    json.dump({"generated_by": "scripts/rc5v2_select_eval_batch.py", "batch_size": len(chosen),
               "theorems": chosen}, open(_p(args.out_batch), "w"), ensure_ascii=False, indent=2)

    ns_dist = Counter(r["namespace"] for r in chosen)
    feat_keys = ["has_subset", "has_iff", "has_mem", "has_disjoint", "has_singleton",
                 "has_map_filter", "has_union_inter", "has_card", "has_tofinset", "has_nat_arith"]
    feat_dist = {k: sum(1 for r in chosen if (r.get("features") or {}).get(k)) for k in feat_keys}
    with_tail = sum(1 for r in chosen if r["tail_score"] > 0)
    summary = {"generated_by": "scripts/rc5v2_select_eval_batch.py", "batch_size": len(chosen),
               "namespace_distribution": dict(ns_dist.most_common()),
               "feature_distribution": feat_dist, "strict_fresh": len(chosen),
               "with_dynamic_tail_features": with_tail,
               "expected_dynamic_gate_rate": "~allowed-ns fraction; measured after static stage"}
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V2 eval batch", "",
          f"- batch size: {len(chosen)} (all strict-fresh)",
          f"- namespaces: {dict(ns_dist.most_common())}",
          f"- with dynamic-tail features: {with_tail}",
          f"- features: {feat_dist}"]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v2-batch] size={len(chosen)} ns={dict(ns_dist.most_common())} tail={with_tail}")


if __name__ == "__main__":
    main()
