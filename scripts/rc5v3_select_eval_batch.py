#!/usr/bin/env python3
"""RC5V3 Part 3 — select the LARGE stratified evaluation batch (target 600, min useful 400).

Stratified by namespace (Set 100 / Finset 140 / List 120 / Multiset 120 / Nat 80 / Order-Other 40)
with focused dynamic-tail slices preferred per namespace. Deterministic ordering (no RNG).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUOTA = {"Set": 100, "Finset": 140, "List": 120, "Multiset": 120, "Nat": 80}
OTHER_QUOTA = 40
TAIL_TOKENS = {
    "Finset": ["image", "subset", "biunion", "filter", "card", "disjoint", "map", "mem"],
    "Multiset": ["bind", "add", "disjoint", "map", "filter", "mem", "count"],
    "List": ["forall", "mem", "map", "filter", "append", "bind", "all"],
    "Set": ["subset", "disjoint", "singleton", "pair", "inter", "union", "image", "mem"],
    "Nat": ["dvd", "div", "mod", "succ", "le", "lt", "add", "mul", "sub", "sqrt"],
}


def _p(*a):
    return os.path.join(_REPO, *a)


def _tail_score(r, ns):
    blob = (r["full_name"] + " " + (r.get("statement_text") or "")).lower()
    return sum(1 for t in TAIL_TOKENS.get(ns, []) if t in blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out-batch", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    ap.add_argument("--batch-size", type=int, default=600)
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(_p(args.pool))]
    by_ns = defaultdict(list)
    for r in pool:
        by_ns[r["namespace"]].append(r)

    chosen, seen = [], set()

    def take(ns, n, candidates):
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

    # scale quotas to batch size if batch_size != sum(QUOTA)+OTHER (600)
    base_total = sum(QUOTA.values()) + OTHER_QUOTA
    scale = args.batch_size / base_total
    for ns, n in QUOTA.items():
        take(ns, max(1, round(n * scale)), by_ns.get(ns, []))
    other = [r for ns in by_ns for r in by_ns[ns] if ns not in QUOTA]
    take("Other", max(1, round(OTHER_QUOTA * scale)), other)

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
    json.dump({"generated_by": "scripts/rc5v3_select_eval_batch.py", "batch_size": len(chosen),
               "theorems": chosen}, open(_p(args.out_batch), "w"), ensure_ascii=False, indent=2)

    ns_dist = Counter(r["namespace"] for r in chosen)
    feat_keys = ["has_subset", "has_iff", "has_mem", "has_disjoint", "has_singleton",
                 "has_image", "has_map_filter", "has_bind", "has_forall_exists",
                 "has_union_inter", "has_card", "has_tofinset", "has_nat_arith"]
    feat_dist = {k: sum(1 for r in chosen if (r.get("features") or {}).get(k)) for k in feat_keys}
    with_tail = sum(1 for r in chosen if r["tail_score"] > 0)
    allowed = {"Set", "Finset", "List", "Multiset", "Nat"}
    est_eligible = sum(1 for r in chosen if r["namespace"] in allowed)
    summary = {"generated_by": "scripts/rc5v3_select_eval_batch.py", "batch_size": len(chosen),
               "meets_400_min": len(chosen) >= 400, "meets_600_target": len(chosen) >= 600,
               "namespace_distribution": dict(ns_dist.most_common()),
               "feature_distribution": feat_dist, "strict_fresh": len(chosen),
               "with_dynamic_tail_features": with_tail,
               "estimated_dynamic_eligible_allowed_ns": est_eligible,
               "estimated_dynamic_eligibility_rate": round(est_eligible / (len(chosen) or 1), 3)}
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V3 eval batch", "",
          f"- batch size: **{len(chosen)}** (all strict-fresh) | ≥400 min: {summary['meets_400_min']} "
          f"| ≥600 target: {summary['meets_600_target']}",
          f"- namespaces: {dict(ns_dist.most_common())}",
          f"- with dynamic-tail features: {with_tail}",
          f"- est. dynamic-eligible (allowed-ns): {est_eligible} ({summary['estimated_dynamic_eligibility_rate']:.0%})",
          f"- features: {feat_dist}"]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v3-batch] size={len(chosen)} ns={dict(ns_dist.most_common())} tail={with_tail} "
          f"est_eligible={est_eligible}")


if __name__ == "__main__":
    main()
