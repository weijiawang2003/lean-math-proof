#!/usr/bin/env python3
"""RC5H Part 3 — build the hybrid benchmark sets.

Seven sets: (1) TR6 dynamic-tail replay (TR6 fresh wins RC4 did NOT reproduce — the decisive
recovery test), (2) TR6 static-covered controls, (3) RC4R fresh no-delta cases, (4) a fresh
dynamic-candidate frontier (not used in TR6/TR7/RC4R), (5) multi-namespace hard negatives
(Nat/Order/root), (6) canonical floors, (7) off-gate controls. Sized so the live dynamic stage
(static failures only) stays tractable. Records sizes, overlap, fresh status, namespace
distribution, and the dynamic-eligible (allowed-namespace) rate per set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4d_gate as G  # noqa: E402

TR6 = "project/evolve/experiments/tr6"
TR7 = "project/evolve/experiments/tr7"
RC4R = "project/evolve/experiments/rc4_release_candidate"
ALLOWED_NS = {"Set", "Finset", "List", "Multiset", "Nat"}
POLICY = "project/evolve/experiments/rc4_candidates/composition_rc4d/rc4d_composition_policy.json"


def _p(*a):
    return os.path.join(_REPO, *a)


def _j(*a):
    return json.load(open(_p(*a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    ap.add_argument("--max-fresh", type=int, default=70)
    ap.add_argument("--max-rc4r-nodelta", type=int, default=28)
    ap.add_argument("--max-hard-neg", type=int, default=22)
    ap.add_argument("--max-offgate", type=int, default=20)
    args = ap.parse_args()

    policy = G.load_policy(POLICY)
    out_dir = os.path.dirname(_p(args.out_manifest))
    os.makedirs(out_dir, exist_ok=True)

    # corpus row lookup (file_path/goal/ns/features) from TR7 + RC4R + TR6 pool
    meta = {}
    for r in (json.loads(l) for l in open(_p(TR7, "cases/tr7_comparison_corpus.jsonl"))):
        meta[r["full_name"]] = r
    for setname, rel in _j(RC4R, "theorem_sets/benchmark_manifest.json")["set_files"].items():
        for e in _j(rel):
            meta.setdefault(e["full_name"], e)
    pool = {}
    for l in open(_p(TR6, "cases/tr6_fresh_frontier_pool.jsonl")):
        r = json.loads(l)
        pool[r["full_name"]] = r

    def entry(fn, set_tag, expected, src=None):
        m = meta.get(fn) or pool.get(fn) or {}
        ns = m.get("namespace") or fn.split(".")[0]
        goal = m.get("goal_text") or m.get("statement_text")
        fp = m.get("file_path")
        fires, em = G.gate_fires(policy, ns, goal, fn)
        return {"full_name": fn, "file_path": fp, "namespace": G.namespace_of(ns, fn),
                "goal_text": goal, "statement_text": m.get("statement_text") or goal,
                "features": m.get("features") or {},
                "rc4_static_gate_fires": fires,
                "dynamic_eligible_namespace": G.namespace_of(ns, fn) in ALLOWED_NS,
                "set_tag": set_tag, "expected_behavior": expected, "source": src}

    # used-name exclusion for the fresh frontier
    used = set(meta)
    tr7corp = {r["full_name"]: r for r in (json.loads(l) for l in open(_p(TR7, "cases/tr7_comparison_corpus.jsonl")))}

    sets = {}

    # (1) TR6 dynamic-tail replay = TR6 fresh wins RC4 did NOT reproduce
    replay = _j(TR7, "out/tr7_rc4_replay_on_tr6_wins.json")["records"]
    tail = [r["full_name"] for r in replay if r["classification"] != "RC4_REPRODUCES_TR6_WIN"]
    sets["TR6_dynamic_tail_replay"] = [entry(fn, "TR6_dynamic_tail_replay",
                                             "RC4 fails; dynamic stage should recover", "tr7_replay")
                                       for fn in tail]
    # (2) TR6 static-covered controls
    covered = [r["full_name"] for r in replay if r["classification"] == "RC4_REPRODUCES_TR6_WIN"]
    sets["TR6_static_covered_controls"] = [entry(fn, "TR6_static_covered_controls",
                                                 "RC4 solves; dynamic gated out", "tr7_replay")
                                           for fn in covered]

    # (3) RC4R fresh no-delta = RC4R fresh frontier where RC4 did not beat RC2
    cmp_rows = {r["full_name"]: r for r in _j(RC4R, "out/rc4_vs_rc2_comparison.json")["rows"]}
    fresh_entries = _j(RC4R, "theorem_sets/fresh_out_of_sample_frontier.json")
    nodelta, firing_taken, nonfiring_taken = [], 0, 0
    for e in fresh_entries:
        fn = e["full_name"]
        row = cmp_rows.get(fn, {})
        if row.get("classification") in ("BOTH_FAILED", "RC4_REGRESSION"):  # RC4 didn't win
            en = entry(fn, "RC4R_fresh_no_delta_set", "RC4 fails on fresh; dynamic eligible", "rc4r_fresh")
            # balance gate-firing and non-firing
            if en["rc4_static_gate_fires"] and firing_taken < args.max_rc4r_nodelta // 2:
                nodelta.append(en); firing_taken += 1
            elif not en["rc4_static_gate_fires"] and nonfiring_taken < args.max_rc4r_nodelta // 2:
                nodelta.append(en); nonfiring_taken += 1
        if len(nodelta) >= args.max_rc4r_nodelta:
            break
    sets["RC4R_fresh_no_delta_set"] = nodelta

    # (4) fresh dynamic-candidate frontier: pool theorems not used anywhere, allowed namespace
    fresh, ns_count = [], Counter()
    for fn, r in pool.items():
        if fn in used or fn in tr7corp:
            continue
        ns = r.get("namespace") or fn.split(".")[0]
        if ns not in ALLOWED_NS or not r.get("file_path"):
            continue
        if ns_count[ns] >= args.max_fresh // 4:   # balance across the 5 allowed namespaces
            continue
        ns_count[ns] += 1
        fresh.append(entry(fn, "Fresh_dynamic_candidate_frontier", "fresh out-of-sample; dynamic test", "tr6_pool"))
        if len(fresh) >= args.max_fresh:
            break
    sets["Fresh_dynamic_candidate_frontier"] = fresh

    # (5) multi-namespace hard negatives: Nat / Order / root theorems from the pool
    hard, hn = [], 0
    for fn, r in pool.items():
        if fn in used or fn in tr7corp or fn in ns_count:
            continue
        ns = r.get("namespace") or fn.split(".")[0]
        is_hard = ns in ("Nat", "Int") or "order" in (r.get("statement_text") or "").lower() \
            or ns in ("Order", "Monotone", "Antitone") or "." not in fn
        if not is_hard or not r.get("file_path"):
            continue
        hard.append(entry(fn, "Multi_namespace_hard_negatives", "hard; dynamic likely no-win", "tr6_pool"))
        hn += 1
        if hn >= args.max_hard_neg:
            break
    sets["Multi_namespace_hard_negatives"] = hard

    # (6) canonical floors (reuse RC4R floor sets)
    floors = []
    for fl in ("canonical_demo_v1", "canonical_nat_defs_medium", "canonical_nat_defs_large_v5"):
        for e in _j(RC4R, "theorem_sets/" + fl + ".json"):
            floors.append(entry(e["full_name"], "canonical_floors", f"floor {fl}; dynamic gated out", "rc4r_floor"))
    sets["canonical_floors"] = floors

    # (7) off-gate controls (reuse RC4R offgate)
    sets["offgate_controls"] = [entry(e["full_name"], "offgate_controls",
                                      "dynamic must not run/fire unsafe", "rc4r_offgate")
                                for e in _j(RC4R, "theorem_sets/offgate_controls.json")][:args.max_offgate]

    # write + manifest
    set_files, sizes, ns_dist, fresh_cnt, dyn_elig, gate_fire = {}, {}, {}, {}, {}, {}
    overlap = {}
    for name, entries in sets.items():
        entries = [e for e in entries if e["file_path"]]
        path = os.path.join(out_dir, name + ".json")
        json.dump(entries, open(path, "w"), ensure_ascii=False, indent=2)
        set_files[name] = os.path.relpath(path, _REPO)
        sizes[name] = len(entries)
        ns_dist[name] = dict(Counter(e["namespace"] for e in entries))
        fresh_cnt[name] = sum(1 for e in entries if e["full_name"] not in used)
        dyn_elig[name] = sum(1 for e in entries if e["dynamic_eligible_namespace"])
        gate_fire[name] = sum(1 for e in entries if e["rc4_static_gate_fires"])
        overlap[name] = sum(1 for e in entries if e["full_name"] in tr7corp)

    uniq = set()
    for entries in sets.values():
        for e in entries:
            if e["file_path"]:
                uniq.add(e["full_name"])
    manifest = {"generated_by": "scripts/rc5h_build_benchmark_sets.py",
                "set_files": set_files, "sizes": sizes, "total": sum(sizes.values()),
                "unique_total": len(uniq)}
    json.dump(manifest, open(_p(args.out_manifest), "w"), ensure_ascii=False, indent=2)
    summary = {"generated_by": "scripts/rc5h_build_benchmark_sets.py",
               "sizes": sizes, "total": sum(sizes.values()), "unique_total": len(uniq),
               "namespace_distribution": ns_dist, "fresh_counts": fresh_cnt,
               "dynamic_eligible_namespace_counts": dyn_elig,
               "static_gate_firing_counts": gate_fire, "overlap_with_tr7": overlap}
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5H benchmark sets", "",
          f"- total: {manifest['total']} | unique: {len(uniq)}", "",
          "| set | size | fresh | dyn-eligible | gate-fires |", "|---|---|---|---|---|"]
    for name in sizes:
        md.append(f"| {name} | {sizes[name]} | {fresh_cnt[name]} | {dyn_elig[name]} | {gate_fire[name]} |")
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5h-sets] sizes={sizes}")
    print(f"[rc5h-sets] total={manifest['total']} unique={len(uniq)} dyn_eligible={dyn_elig}")


if __name__ == "__main__":
    main()
