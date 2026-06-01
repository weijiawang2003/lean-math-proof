#!/usr/bin/env python3
"""RC5H Part 4 — static-stage benchmark (the frozen RC4R wrapper).

Reuse-first: RC4R RC4 benchmark + RC4D schema smoke (wrapper side) + RC4D floor-bench RC4 + RC4B
candidate. The RC4 wrapper is purely additive, so non-gate-firing theorems are forced RC4 ≡ RC2
(reusing the RC5H RC2 baseline); only gate-firing theorems not previously measured run live.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4r_bench_common as C  # noqa: E402
import rc4d_gate as G  # noqa: E402

RC4R = "project/evolve/experiments/rc4_release_candidate/out"
RC4D = "project/evolve/experiments/rc4_candidates/composition_rc4d/out"
POLICY = "project/evolve/experiments/rc4_candidates/composition_rc4d/rc4d_composition_policy.json"
FLOOR_REUSE = {"canonical_demo_v1": RC4D + "/floor_bench/rc4d_demo_v1.json",
               "canonical_nat_defs_medium": RC4D + "/floor_bench/rc4d_nat_defs_medium.json",
               "canonical_nat_defs_large_v5": RC4D + "/floor_bench/rc4d_nat_defs_large_v5.json"}
COMP_OF = {}
for _c, _ts in {
    "RC4A": ["simp [Finset.disjUnion]", "simp [Monotone, MonotoneOn]", "simp [Antitone, AntitoneOn]",
             "simp [StrictMono, StrictMonoOn]", "simp [StrictAnti, StrictAntiOn]"],
    "RC4B": ["simp [Set.disjoint_left]", "simp [Set.disjoint_left] <;> aesop",
             "simp [Multiset.disjoint_left]", "simp [Multiset.disjoint_left] <;> aesop"],
    "RC4C_residue": ["simp [Multiset.disjoint_right]", "simp [Multiset.disjoint_right] <;> aesop",
                     "simp [Set.subset_pair_iff_eq]", "simp [Set.subset_pair_iff_eq] <;> aesop",
                     "simp [List.forall_iff_forall_mem]", "simp [List.forall_iff_forall_mem] <;> aesop"],
}.items():
    for _t in _ts:
        COMP_OF[_t] = _c


def _p(*a):
    return os.path.join(_REPO, *a)


def _reuse():
    m = {}
    rb = _p(RC4R, "rc4_benchmark_results.json")
    if os.path.exists(rb):
        for r in json.load(open(rb))["results"]:
            m.setdefault(r["full_name"], {"status": r["status"], "winning_tactic": r.get("winning_tactic"),
                                          "provenance": "reused:rc4r_rc4_bench"})
    sm = _p(RC4D, "schema_wrapper_smoke.json")
    if os.path.exists(sm):
        for r in json.load(open(sm))["results"]:
            m.setdefault(r["full_name"], {"status": "solved" if r.get("wrapper_finished") else "failed",
                                          "winning_tactic": r.get("winning_tactic"),
                                          "provenance": "reused:rc4d_schema_smoke"})
    for rel in FLOOR_REUSE.values():
        if os.path.exists(_p(rel)):
            solved = set(json.load(open(_p(rel))).get("solved_names", []))
            for fn in solved:
                m.setdefault(fn, {"status": "solved", "provenance": "reused:rc4d_floor_bench_rc4"})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--cases-json")
    ap.add_argument("--set-label", default="rc4")
    ap.add_argument("--manifest")
    ap.add_argument("--rc4-wrapper", default="project/evolve/experiments/rc4_release_candidate/rc4_release_candidate_wrapper.json")
    ap.add_argument("--route-config", default="project/evolve/routing/ns24_router.json")
    ap.add_argument("--rc2-results", default="project/evolve/experiments/rc5_hybrid/out/rc5h_rc2_baseline_results.json")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--out-dir", default="project/evolve/experiments/rc5_hybrid/out/rc4_bench")
    ap.add_argument("--checkpoint", default="project/evolve/experiments/rc5_hybrid/out/rc4_bench_checkpoint.json")
    ap.add_argument("--wrapper")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--hard-timeout", type=int, default=1800)
    args = ap.parse_args()
    if args.worker:
        C.run_worker(args.worker_out, args.cases_json, args.out_dir, args.wrapper,
                     args.route_config, "hybrid_evolved", args.top_k, args.max_steps, args.set_label)
        return

    manifest = json.load(open(_p(args.manifest)))
    policy = G.load_policy(POLICY)
    rc2 = {}
    if os.path.exists(_p(args.rc2_results)):
        rc2 = {r["full_name"]: r for r in json.load(open(_p(args.rc2_results)))["results"]}
    # gate map per theorem
    gatefire = {}
    for setname, rel in manifest["set_files"].items():
        for e in json.load(open(_p(rel))):
            fires, _ = G.gate_fires(policy, e.get("namespace"), e.get("goal_text") or e.get("statement_text"),
                                    e["full_name"])
            gatefire[e["full_name"]] = fires

    def skip_predicate(entry):
        if gatefire.get(entry["full_name"]):
            return None
        r2 = rc2.get(entry["full_name"])
        if not r2:
            return None
        return {"status": r2["status"], "winning_tactic": r2.get("winning_tactic"),
                "provenance": "rc4_equals_rc2_nonfiring"}

    recs, _ = C.run_benchmark(manifest, _p(args.rc4_wrapper), _p(args.route_config), args.out_dir,
                              args.checkpoint, _reuse(), os.path.abspath(__file__),
                              top_k=args.top_k, max_steps=args.max_steps, chunk_size=args.chunk_size,
                              hard_timeout=args.hard_timeout, skip_predicate=skip_predicate, label="rc4")
    for r in recs:
        r["rc4_component"] = COMP_OF.get(r.get("winning_tactic"))
    roll = C.rollup(recs, manifest)
    out = {"generated_by": "scripts/rc5h_run_static_stage.py", "wrapper": args.rc4_wrapper,
           "num_theorems": len(recs), **roll, "results": recs}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5H static stage (RC4R)", "", f"- theorems: {len(recs)} | {roll['status_histogram']}", "",
          "| set | n | solved |", "|---|---|---|"]
    for s, d in roll["by_set"].items():
        md.append(f"| {s} | {d['n']} | {d['solved']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5h-static] {roll['status_histogram']}")
    print(f"[rc5h-static] by_set={ {s: d['solved'] for s, d in roll['by_set'].items()} }")


if __name__ == "__main__":
    main()
