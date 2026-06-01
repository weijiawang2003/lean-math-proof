#!/usr/bin/env python3
"""RC4R Part 5 — RC4 release-candidate benchmark over the benchmark manifest.

The RC4 wrapper is purely additive (gated priority tactics): on any theorem whose name matches
no RC4 gate prefix the search is byte-identical to RC2, so RC4 ≡ RC2 there by construction. This
runner therefore:
  * reuses the canonical-floor + known-win RC4 results from RC4D (the RC4 release wrapper has the
    same actions/gates as the validated RC4D wrapper — only metadata differs, which the loader
    ignores);
  * for any non-gate-firing theorem, FORCES the RC4 result = the RC2 benchmark result (no live
    run — provably identical);
  * runs live ONLY the gate-firing fresh out-of-sample theorems (where a fresh RC4 delta can occur).
Records component-of-winning-tactic where the RC4 winning tactic is a known component action.
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

RC4D = "project/evolve/experiments/rc4_candidates/composition_rc4d/out"
FLOOR_REUSE = {"canonical_demo_v1": RC4D + "/floor_bench/rc4d_demo_v1.json",
               "canonical_nat_defs_medium": RC4D + "/floor_bench/rc4d_nat_defs_medium.json",
               "canonical_nat_defs_large_v5": RC4D + "/floor_bench/rc4d_nat_defs_large_v5.json"}
# winning-tactic -> component (the 15 RC4 actions)
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


def _floor_reuse(manifest):
    m = {}
    for setname, rel in FLOOR_REUSE.items():
        if not os.path.exists(_p(rel)):
            continue
        fr = json.load(open(_p(rel)))
        solved = set(fr.get("solved_names", []))
        for e in json.load(open(_p(manifest["set_files"][setname]))):
            fn = e["full_name"]
            m.setdefault(fn, {"status": "solved" if fn in solved else "failed",
                              "provenance": "reused:rc4d_floor_bench_rc4"})
    return m


def _schema_reuse():
    p = _p(RC4D, "schema_wrapper_smoke.json")
    if not os.path.exists(p):
        return {}
    m = {}
    for r in json.load(open(p))["results"]:
        m[r["full_name"]] = {"status": "solved" if r.get("wrapper_finished") else "failed",
                             "winning_tactic": r.get("winning_tactic"),
                             "provenance": "reused:rc4d_schema_smoke"}
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
    ap.add_argument("--rc2-results", default="project/evolve/experiments/rc4_release_candidate/out/rc2_benchmark_results.json")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--out-dir", default="project/evolve/experiments/rc4_release_candidate/out/rc4_bench")
    ap.add_argument("--checkpoint", default="project/evolve/experiments/rc4_release_candidate/out/rc4_bench_checkpoint.json")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--hard-timeout", type=int, default=1800)
    ap.add_argument("--wrapper")
    args = ap.parse_args()

    if args.worker:
        C.run_worker(args.worker_out, args.cases_json, args.out_dir, args.wrapper,
                     args.route_config, "hybrid_evolved", args.top_k, args.max_steps, args.set_label)
        return

    manifest = json.load(open(_p(args.manifest)))
    # RC2 benchmark results for the additive-equivalence force on non-firing theorems
    rc2 = {}
    if os.path.exists(_p(args.rc2_results)):
        rc2 = {r["full_name"]: r for r in json.load(open(_p(args.rc2_results)))["results"]}

    reuse = {}
    reuse.update(_schema_reuse())
    reuse.update(_floor_reuse(manifest))  # floors authoritative for RC4

    def skip_predicate(entry):
        # RC4 ≡ RC2 on any non-gate-firing theorem (purely additive wrapper).
        if entry.get("rc4_gate_fires"):
            return None
        r2 = rc2.get(entry["full_name"])
        if not r2:
            return None  # no RC2 result yet -> run live (shouldn't happen post-RC2)
        return {"status": r2["status"], "winning_tactic": r2.get("winning_tactic"),
                "provenance": "rc4_equals_rc2_nonfiring"}

    recs, membership = C.run_benchmark(
        manifest, _p(args.rc4_wrapper), _p(args.route_config), args.out_dir, args.checkpoint,
        reuse, os.path.abspath(__file__), top_k=args.top_k, max_steps=args.max_steps,
        chunk_size=args.chunk_size, hard_timeout=args.hard_timeout,
        skip_predicate=skip_predicate, label="rc4")

    # tag component of winning tactic
    for r in recs:
        r["rc4_component"] = COMP_OF.get(r.get("winning_tactic"))
    roll = C.rollup(recs, manifest)
    out = {"generated_by": "scripts/rc4r_run_rc4_benchmark.py", "wrapper": args.rc4_wrapper,
           "route_config": args.route_config, "policy_type": "hybrid_evolved",
           "top_k": args.top_k, "max_steps": args.max_steps, "num_theorems": len(recs),
           **roll, "results": recs}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC4 release-candidate benchmark", "",
          f"- theorems: {len(recs)} | status: {roll['status_histogram']}", "",
          "## By set", "", "| set | n | solved | failed | flake | path_err |", "|---|---|---|---|---|---|"]
    for s, d in roll["by_set"].items():
        md.append(f"| {s} | {d['n']} | {d['solved']} | {d['failed']} | {d['flake']} | {d['path_error']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4r-rc4] {roll['status_histogram']}")
    print(f"[rc4r-rc4] by_set={ {s: d['solved'] for s, d in roll['by_set'].items()} }")


if __name__ == "__main__":
    main()
