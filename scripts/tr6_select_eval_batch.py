#!/usr/bin/env python3
"""TR6 Part 4 — select the stratified overnight eval batch.

Stratifies the fresh frontier into Set / Finset / List / Nat / Multiset / Order / Other
with quotas, prioritising within each stratum by proof-search-relevant features (disjoint
& iff & subset rank high for RC4B/RC4C support; card/map_filter/tofinset/mem next), keeping
some Nat-arithmetic hard negatives, and tagging RC4B/RC4C candidacy. Deterministic ordering
(no RNG). Verifies zero overlap with the exclusion registry.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORDER_NS = {"Antitone", "AntitoneOn", "Monotone", "MonotoneOn", "StrictMono", "StrictMonoOn",
            "StrictAnti", "StrictAntiOn", "BddAbove", "BddBelow", "IsGLB", "IsLUB", "IsLeast",
            "IsGreatest", "OrderBot", "OrderTop", "OrderDual", "Preorder", "PartialOrder",
            "LinearOrder", "LE", "LT", "GE", "NoMaxOrder", "NoMinOrder", "ScottContinuous"}
CORE = {"Set", "Finset", "List", "Nat", "Multiset"}
ORDER_FILES = ("Order/",)


def _p(*a):
    return os.path.join(_REPO, *a)


def _stratum(r):
    ns = r["namespace"]
    if ns in CORE:
        return ns
    if ns in ORDER_NS or (r.get("file_path") or "").find("Order/") != -1:
        return "Order"
    return "Other"


def _priority(r):
    f = r.get("features", {})
    s = 0.0
    if f.get("has_disjoint"):
        s += 3.0   # RC4B/RC4C support
    if f.get("has_iff"):
        s += 2.0
    if f.get("has_subset"):
        s += 1.5
    if f.get("has_tofinset"):
        s += 1.5
    if f.get("has_card"):
        s += 1.0
    if f.get("has_map_filter"):
        s += 1.0
    if f.get("has_mem"):
        s += 0.8
    if f.get("has_union_inter"):
        s += 0.8
    # prefer cases that have a statement (richer features) and shorter statements
    if r.get("statement_text"):
        s += 0.5
        s -= min(1.0, len(r["statement_text"]) / 800.0)
    return s


def _tags(r):
    f = r.get("features", {})
    t = []
    if f.get("has_disjoint"):
        t += ["rc4b_candidate", "rc4c_candidate"]
    if f.get("has_iff") or f.get("has_subset") or f.get("has_eq"):
        t.append("d2_simp_aesop_candidate")
    if f.get("has_nat_arith") and not (f.get("has_iff") or f.get("has_subset") or f.get("has_disjoint")):
        t.append("hard_negative_nat")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out-batch", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--exclusion",
                    default="project/evolve/experiments/tr6/cases/tr6_exclusion_registry.json")
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(_p(args.pool)) if l.strip()]
    excl = set(json.load(open(_p(args.exclusion)))["excluded_full_names"])

    # quotas scaled to batch size (defaults for 200)
    base = {"Set": 30, "Finset": 35, "List": 35, "Nat": 35, "Multiset": 25, "Order": 20}
    quotas = dict(base)
    quotas["Other"] = max(0, args.batch_size - sum(base.values()))

    strat = {k: [] for k in list(base) + ["Other"]}
    for r in pool:
        strat[_stratum(r)].append(r)
    for k in strat:
        strat[k].sort(key=lambda r: (-_priority(r), r["full_name"]))

    selected, leftover_pool = [], []
    for k, q in quotas.items():
        picked = strat[k][:q]
        selected.extend(picked)
        leftover_pool.append((k, len(strat[k]), len(picked)))
    # if a stratum is short, backfill from Other / largest strata to reach batch size
    if len(selected) < args.batch_size:
        chosen = {r["full_name"] for r in selected}
        rest = sorted([r for r in pool if r["full_name"] not in chosen],
                      key=lambda r: (-_priority(r), r["full_name"]))
        selected.extend(rest[: args.batch_size - len(selected)])

    for r in selected:
        r["candidate_family_tags"] = _tags(r)

    # zero-overlap verification
    overlap = [r["full_name"] for r in selected if r["full_name"] in excl]

    by_ns = Counter(r["namespace"] for r in selected)
    by_strat = Counter(_stratum(r) for r in selected)
    feat = {k: sum(1 for r in selected if r["features"].get(k)) for k in
            ("has_iff", "has_eq", "has_subset", "has_mem", "has_disjoint", "has_card",
             "has_map_filter", "has_tofinset", "has_order", "has_nat_arith", "has_union_inter")}
    rc4b = [r["full_name"] for r in selected if "rc4b_candidate" in r["candidate_family_tags"]]
    hardneg = [r["full_name"] for r in selected if "hard_negative_nat" in r["candidate_family_tags"]]

    batch = {"generated_by": "scripts/tr6_select_eval_batch.py",
             "batch_size": len(selected), "theorems": selected}
    manifest = {"generated_by": "scripts/tr6_select_eval_batch.py",
                "batch_size": len(selected), "quotas": quotas,
                "by_stratum": dict(by_strat), "by_namespace": dict(by_ns),
                "feature_distribution": feat,
                "rc4b_rc4c_candidate_count": len(rc4b),
                "hard_negative_nat_count": len(hardneg),
                "overlap_with_exclusion": overlap,
                "overlap_is_zero": len(overlap) == 0,
                "expected_runtime_note": "~200 theorems × (open ~6s + 8-step RC2 search); "
                                         "serialized overnight, hard theorems up to ~80s."}
    json.dump(batch, open(_p(args.out_batch), "w"), ensure_ascii=False, indent=2)
    json.dump(manifest, open(_p(args.out_manifest), "w"), ensure_ascii=False, indent=2)
    md = ["# TR6 eval batch", "",
          f"- **{len(selected)} theorems** | overlap with exclusion: **{len(overlap)}** "
          f"(must be 0)",
          f"- by stratum: {dict(by_strat)}",
          f"- by namespace: {dict(by_ns)}",
          f"- feature distribution: {feat}",
          f"- RC4B/RC4C candidates: {len(rc4b)} | hard-negative Nat: {len(hardneg)}",
          "", "## quotas vs available", "",
          "| stratum | available | picked |", "|---|---|---|"]
    for k, avail, pick in leftover_pool:
        md.append(f"| {k} | {avail} | {pick} |")
    open(_p(args.out_summary), "w").write("\n".join(md) + "\n")
    print(f"[tr6-batch] {len(selected)} theorems; by_stratum={dict(by_strat)}; "
          f"overlap={len(overlap)}; rc4b/c_cand={len(rc4b)}; hardneg={len(hardneg)}")


if __name__ == "__main__":
    main()
