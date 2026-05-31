#!/usr/bin/env python3
"""RC4A Part 4 — build validation theorem sets for the def_unfold_simp candidate.

Sets: known_wins / same_cluster_holdout (gate fires, not a known win) /
fresh_frontier_holdout (frontier/discovered cases naming an allowlisted def) /
negative_controls (confirmed RC2 failures naming NO allowlisted def — gate must not
fire) / canonical_smoke (demo_v1 + nat_defs_medium + nat_defs_large_v5 samples).
Plus a validation_manifest with sizes, TR3 overlap, leakage notes and expected gate
behaviour.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4a_gate as G  # noqa: E402

POLICY = "project/evolve/experiments/rc4_candidates/def_unfold_simp/def_unfold_simp_policy.json"


def _p(*a):
    return os.path.join(_REPO, *a)


def _entry(full_name, file_path, namespace, goal_text, policy, expected):
    fires, tactic, defs = G.gate_fires(policy, goal_text, full_name)
    return {"full_name": full_name, "file_path": file_path, "namespace": namespace,
            "goal_text": goal_text, "gate_should_fire": fires,
            "candidate_tactic": tactic, "matched_defs": defs,
            "expected_behavior": expected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-wins", required=True)
    ap.add_argument("--tr3-pool", required=True)
    ap.add_argument("--tr3-confirmation", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frontier",
                    default="project/evolve/experiments/sf1/out/real/frontier_with_paths.jsonl")
    ap.add_argument("--max-fresh", type=int, default=30)
    ap.add_argument("--max-neg", type=int, default=20)
    ap.add_argument("--smoke-medium", type=int, default=15)
    ap.add_argument("--smoke-large", type=int, default=15)
    args = ap.parse_args()

    policy = G.load_policy(POLICY)
    allow = policy["validated_def_allowlist"]
    known = json.load(open(_p(args.known_wins)))
    known_names = {w["full_name"] for w in known}
    conf = json.load(open(_p(args.tr3_confirmation)))["results"]
    conf_by = {r["full_name"]: r for r in conf}
    out_dir = _p(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1) known_wins (enriched with gate info)
    kw = []
    for w in known:
        r = conf_by.get(w["full_name"], {})
        kw.append(_entry(w["full_name"], w["file_path"], w["namespace"],
                         r.get("goal_text"), policy, "gate fires; expected NEW WIN"))

    # 2) same_cluster_holdout: TR3 cases whose goal names an allowlist def, not a known win
    sch = []
    for r in conf:
        if r["full_name"] in known_names:
            continue
        fires, _, _ = G.gate_fires(policy, r.get("goal_text"), r["full_name"])
        if fires:
            exp = ("gate fires; RC2 already solved (additive no-op)"
                   if r["classification"] == "RC2_SOLVED"
                   else "gate fires; win NOT guaranteed (tests generalization)")
            sch.append(_entry(r["full_name"], r.get("file_path"), r.get("namespace"),
                              r.get("goal_text"), policy, exp))

    # 3) fresh_frontier_holdout: frontier + discovered cases naming an allowlist def,
    #    excluding everything already chosen
    chosen = known_names | {e["full_name"] for e in sch}
    fresh = []
    frontier = [json.loads(l) for l in open(_p(args.frontier))] if os.path.exists(_p(args.frontier)) else []
    discovered = (json.load(open(_p("project/discovered_theorems.json"))) or {}).get("theorems", [])
    seen_fresh = set()
    for row in frontier + discovered:
        fn = row.get("name") or row.get("full_name") or row.get("decl_name")
        fp = row.get("file_path")
        if not fn or not fp or fn in chosen or fn in seen_fresh:
            continue
        stmt = G.statement_from_source(fp, fn)
        fires, _, _ = G.gate_fires(policy, stmt, fn)
        if fires:
            seen_fresh.add(fn)
            ns = fn.split(".")[0]
            fresh.append(_entry(fn, fp, ns, stmt, policy,
                                "fresh holdout; gate fires; RC2 status unknown"))
        if len(fresh) >= args.max_fresh:
            break

    # 4) negative_controls: confirmed RC2 failures naming NO allowlist def (gate must NOT fire)
    neg = []
    pref_ns = ("Nat", "Multiset", "List")
    cand = [r for r in conf if r["classification"] == "CONFIRMED_RC2_FAILURE"
            and not G.gate_fires(policy, r.get("goal_text"), r["full_name"])[0]]
    cand.sort(key=lambda r: (0 if r.get("namespace") in pref_ns else 1, r["full_name"]))
    for r in cand[: args.max_neg]:
        neg.append(_entry(r["full_name"], r.get("file_path"), r.get("namespace"),
                          r.get("goal_text"), policy, "gate must NOT fire (off-gate check)"))

    # 5) canonical_smoke
    from tasks import get_theorems
    smoke = []

    def _add_smoke(setname, n):
        try:
            ts = get_theorems(setname)
        except Exception:
            return
        sel = ts if (n is None or len(ts) <= n) else ts[:n]
        for t in sel:
            stmt = G.statement_from_source(t.file_path, t.full_name)
            smoke.append(_entry(t.full_name, t.file_path,
                                t.full_name.split(".")[0], stmt, policy,
                                f"canonical floor ({setname}); gate generally must not fire"))
    _add_smoke("demo_v1", None)
    _add_smoke("nat_defs_medium", args.smoke_medium)
    _add_smoke("nat_defs_large_v5", args.smoke_large)

    sets = {
        "known_wins": kw, "same_cluster_holdout": sch,
        "fresh_frontier_holdout": fresh, "negative_controls": neg,
        "canonical_smoke": smoke,
    }
    for name, data in sets.items():
        json.dump(data, open(os.path.join(out_dir, f"{name}.json"), "w"),
                  ensure_ascii=False, indent=2)

    tr3_names = set(conf_by)
    manifest = {
        "generated_by": "scripts/rc4a_build_validation_sets.py",
        "policy": POLICY, "allowlist": allow,
        "sizes": {k: len(v) for k, v in sets.items()},
        "gate_fire_counts": {k: sum(1 for e in v if e["gate_should_fire"])
                             for k, v in sets.items()},
        "overlap_with_tr3": {k: sum(1 for e in v if e["full_name"] in tr3_names)
                             for k, v in sets.items()},
        "leakage_notes": [
            "known_wins are the 5 TR3 def_unfold_simp wins (in-sample by construction).",
            "same_cluster_holdout are TR3 confirmed/solved cases that ALSO name an "
            "allowlisted def but were NOT credited wins — tests generalization, NOT "
            "in-sample wins.",
            "fresh_frontier_holdout come from the SF1 frontier / discovered catalog "
            "(may overlap TR3 if the same theorem was confirmed there).",
            "negative_controls and canonical_smoke must show 0 gate emissions / no "
            "regressions.",
        ],
        "expected_behavior": {
            "known_wins": "gate fires on all; expected to reproduce TR3 wins",
            "same_cluster_holdout": "gate fires; wins NOT guaranteed",
            "fresh_frontier_holdout": "gate fires; wins NOT guaranteed",
            "negative_controls": "0 gate emissions",
            "canonical_smoke": "≈0 gate emissions; floors preserved",
        },
        "set_files": {k: os.path.relpath(os.path.join(out_dir, f"{k}.json"), _REPO)
                      for k in sets},
    }
    json.dump(manifest, open(os.path.join(out_dir, "validation_manifest.json"), "w"),
              ensure_ascii=False, indent=2)

    print(f"[rc4a-sets] sizes={manifest['sizes']}")
    print(f"           gate_fire_counts={manifest['gate_fire_counts']}")


if __name__ == "__main__":
    main()
