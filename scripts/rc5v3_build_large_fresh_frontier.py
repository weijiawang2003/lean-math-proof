#!/usr/bin/env python3
"""RC5V3 Part 2 — build the LARGE fresh out-of-sample frontier.

Union of (TR6 fresh pool ∪ RC5V2 fresh pool ∪ discovered catalog), minus an exclusion registry
covering EVERY prior-used theorem: RC4D/RC4R known wins, all TR6 wins + batch, RC5H/RC5S/RC5V2 true
wins + batches, the RC5V2 eval batch + dynamic-examples corpus, and TR7's comparison corpus. Tags
freshness, prefers the allowed namespaces, carries features. Target ≥800 strict-fresh candidates.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED = {"Set", "Finset", "List", "Multiset", "Nat"}
CONTROL = {"Order", "Int", "Option", "Bool", "Function"}

FEATURE_TOKENS = {
    "has_subset": ["subset", "⊆", "ssubset"],
    "has_iff": ["iff", "↔"],
    "has_mem": ["mem", "∈"],
    "has_disjoint": ["disjoint"],
    "has_singleton": ["singleton", "{", "insert"],
    "has_image": ["image", "map"],
    "has_map_filter": ["map", "filter"],
    "has_bind": ["bind"],
    "has_forall_exists": ["forall", "∀", "exists", "∃", "bex", "ball"],
    "has_card": ["card"],
    "has_tofinset": ["tofinset", "tofinset"],
    "has_union_inter": ["union", "inter", "∪", "∩"],
    "has_nat_arith": ["add", "mul", "sub", "div", "mod", "succ", "dvd", "le", "lt"],
    "has_order": ["order", "mono", "bound", "lattice"],
}


def _p(*a):
    return os.path.join(_REPO, *a)


def _names_from_manifest(rel):
    out = set()
    try:
        man = json.load(open(_p(rel)))
        for _s, r in man.get("set_files", {}).items():
            for e in json.load(open(_p(r))):
                out.add(e["full_name"])
    except Exception:
        pass
    return out


def _names_from_attr(rel):
    out = set()
    try:
        a = json.load(open(_p(rel)))
        out.update(a.get("fresh_true_delta_targets", []))
        for r in a.get("records", []):
            if r.get("full_name"):
                out.add(r["full_name"])
    except Exception:
        pass
    return out


def _names_from_batch(rel):
    out = set()
    try:
        b = json.load(open(_p(rel)))
        for r in (b.get("theorems", b) if isinstance(b, dict) else b):
            if isinstance(r, dict) and r.get("full_name"):
                out.add(r["full_name"])
    except Exception:
        pass
    return out


def _names_from_jsonl(rel, key="full_name"):
    out = set()
    try:
        for l in open(_p(rel)):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            v = r.get(key) or r.get("target") or r.get("theorem")
            if v:
                out.add(v)
    except Exception:
        pass
    return out


def _exclusion_registry():
    excl = set()
    # TR6
    excl |= _names_from_batch("project/evolve/experiments/tr6/cases/tr6_eval_batch.json")
    excl |= _names_from_attr("project/evolve/experiments/tr6/out/tr6_attribution.json")
    # RC4D / RC4R known wins (manifests + comparison)
    excl |= _names_from_manifest("project/evolve/experiments/rc4_candidates/composition_rc4d/theorem_sets/validation_manifest.json")
    excl |= _names_from_manifest("project/evolve/experiments/rc4_release_candidate/theorem_sets/benchmark_manifest.json")
    # RC5H / RC5S benchmark manifests + true wins
    excl |= _names_from_manifest("project/evolve/experiments/rc5_hybrid/cases/rc5h_benchmark_manifest.json")
    excl |= _names_from_manifest("project/evolve/experiments/rc5_safety/cases/rc5s_benchmark_manifest.json")
    excl |= _names_from_attr("project/evolve/experiments/rc5_hybrid/out/rc5h_attribution.json")
    excl |= _names_from_attr("project/evolve/experiments/rc5_safety/out/rc5s_attribution.json")
    # RC5V2 eval batch + true wins + dynamic-examples corpus
    excl |= _names_from_batch("project/evolve/experiments/rc5_v2/cases/rc5v2_eval_batch.json")
    excl |= _names_from_attr("project/evolve/experiments/rc5_v2/out/rc5v2_attribution.json")
    excl |= _names_from_jsonl("project/evolve/experiments/rc5_v2/data/rc5v2_dynamic_examples.jsonl")
    # TR7 comparison corpus / gap examples
    excl |= _names_from_jsonl("project/evolve/experiments/tr7/cases/tr7_comparison_corpus.jsonl")
    excl |= _names_from_jsonl("project/evolve/experiments/tr7/data/tr7_fresh_delta_gap_examples.jsonl")
    return excl


def _derive_features(fn, stmt, given):
    if given:
        return given
    blob = (fn + " " + (stmt or "")).lower()
    return {k: any(tok in blob for tok in toks) for k, toks in FEATURE_TOKENS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-pool", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    ap.add_argument("--target", type=int, default=800)
    args = ap.parse_args()

    excl = _exclusion_registry()

    pool = {}
    pool_sources = [
        ("tr6_pool", "project/evolve/experiments/tr6/cases/tr6_fresh_frontier_pool.jsonl"),
        ("rc5v2_pool", "project/evolve/experiments/rc5_v2/cases/rc5v2_fresh_frontier_pool.jsonl"),
    ]
    for src, rel in pool_sources:
        try:
            for l in open(_p(rel)):
                r = json.loads(l)
                r.setdefault("source", src)
                pool.setdefault(r["full_name"], r)
        except Exception:
            pass
    # discovered catalog
    try:
        disc = json.load(open(_p("project/discovered_theorems.json")))
        for r in (disc.get("theorems", disc) if isinstance(disc, dict) else disc):
            if isinstance(r, dict) and r.get("full_name"):
                r.setdefault("source", "discovered")
                pool.setdefault(r["full_name"], r)
    except Exception:
        pass

    rows = []
    for fn, r in pool.items():
        if not r.get("file_path"):
            continue
        ns = r.get("namespace") or fn.split(".")[0]
        stmt = r.get("statement_text")
        status = "known_control" if fn in excl else "strict_fresh"
        rows.append({
            "full_name": fn, "file_path": r.get("file_path"), "namespace": ns,
            "freshness_status": status,
            "features": _derive_features(fn, stmt, r.get("features")),
            "statement_text": stmt,
            "source": r.get("source", "tr6_pool"),
            "excluded_reason": "prior_used" if status == "known_control" else None,
        })

    fresh = [r for r in rows if r["freshness_status"] == "strict_fresh"]
    with open(_p(args.out_pool), "w") as f:
        for r in fresh:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ns_dist = Counter(r["namespace"] for r in fresh)
    allowed_fresh = [r for r in fresh if r["namespace"] in ALLOWED]
    feat_keys = list(FEATURE_TOKENS.keys())
    feat_dist = {k: sum(1 for r in fresh if (r["features"] or {}).get(k)) for k in feat_keys}
    summary = {
        "generated_by": "scripts/rc5v3_build_large_fresh_frontier.py",
        "exclusion_registry_size": len(excl),
        "pool_total": len(rows), "strict_fresh": len(fresh),
        "allowed_namespace_fresh": len(allowed_fresh),
        "namespace_distribution": dict(ns_dist.most_common()),
        "allowed_namespace_distribution": dict(Counter(r["namespace"] for r in allowed_fresh).most_common()),
        "feature_distribution": feat_dist,
        "meets_800_target": len(fresh) >= args.target,
        "meets_800_allowed_ns": len(allowed_fresh) >= args.target,
    }
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V3 large fresh frontier", "",
          f"- exclusion registry: **{len(excl)}** prior-used theorems",
          f"- pool total (file-path'd): {len(rows)}",
          f"- **strict-fresh candidates: {len(fresh)}** (allowed-ns: {len(allowed_fresh)}) "
          f"| ≥{args.target} target: {summary['meets_800_target']}",
          f"- namespaces (all): {dict(ns_dist.most_common(12))}",
          f"- namespaces (allowed): {summary['allowed_namespace_distribution']}",
          f"- features: {feat_dist}"]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v3-frontier] excl={len(excl)} strict_fresh={len(fresh)} "
          f"allowed_ns={len(allowed_fresh)} meets_{args.target}={summary['meets_800_target']}")
    print(f"[rc5v3-frontier] ns(allowed)={summary['allowed_namespace_distribution']}")


if __name__ == "__main__":
    main()
