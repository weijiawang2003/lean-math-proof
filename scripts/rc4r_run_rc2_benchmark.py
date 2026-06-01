#!/usr/bin/env python3
"""RC4R Part 4 — RC2 baseline benchmark over the benchmark manifest.

Reuse-first at the exact RC2 config (rc2_release wrapper, ns24, hybrid_evolved, top-k 8,
max-steps 8): canonical floors reuse the RC4D full-floor-benchmark RC2 solved-name lists; the
23 known wins + negative/offgate controls reuse RC4D literal RC2 + schema smoke; TR
confirmations fill remaining known statuses. Only the fresh out-of-sample frontier (not seen at
this config) runs live in chunked worker subprocesses under an OS hard timeout.
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
RESULT_FILES = [RC4D + "/literal_rc2_results.json"]
CONF_FILES = [
    "project/evolve/experiments/tr6/out/tr6_rc2_confirmation.json",
    "project/evolve/experiments/tr3/out/tr3_rc2_confirmation.json",
    "project/evolve/experiments/tr5/out/tr5_rc2_confirmation.json",
    "project/evolve/experiments/sf4/out/rc2_failure_confirmation.json",
]
FLOOR_REUSE = {"canonical_demo_v1": RC4D + "/floor_bench/rc2_demo_v1.json",
               "canonical_nat_defs_medium": RC4D + "/floor_bench/rc2_nat_defs_medium.json",
               "canonical_nat_defs_large_v5": RC4D + "/floor_bench/rc2_nat_defs_large_v5.json"}


def _p(*a):
    return os.path.join(_REPO, *a)


def _floor_reuse(manifest):
    """{fn: {status}} from RC4D floor-benchmark RC2 solved-name lists."""
    m = {}
    for setname, rel in FLOOR_REUSE.items():
        if not os.path.exists(_p(rel)):
            continue
        fr = json.load(open(_p(rel)))
        solved = set(fr.get("solved_names", []))
        for e in json.load(open(_p(manifest["set_files"][setname]))):
            fn = e["full_name"]
            m.setdefault(fn, {"status": "solved" if fn in solved else "failed",
                              "provenance": "reused:rc4d_floor_bench"})
    return m


def _schema_reuse():
    """{fn: {status}} RC2 side from the RC4D schema smoke per-theorem records."""
    p = _p(RC4D, "schema_wrapper_smoke.json")
    if not os.path.exists(p):
        return {}
    m = {}
    for r in json.load(open(p))["results"]:
        m[r["full_name"]] = {"status": "solved" if r.get("rc2_finished") else "failed",
                             "provenance": "reused:rc4d_schema_smoke"}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--cases-json")
    ap.add_argument("--set-label", default="rc2")
    ap.add_argument("--manifest")
    ap.add_argument("--rc2-wrapper", default="project/evolve/experiments/rc2_release/rc2_production_wrapper.json")
    ap.add_argument("--route-config", default="project/evolve/routing/ns24_router.json")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--out-dir", default="project/evolve/experiments/rc4_release_candidate/out/rc2_bench")
    ap.add_argument("--checkpoint", default="project/evolve/experiments/rc4_release_candidate/out/rc2_bench_checkpoint.json")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--hard-timeout", type=int, default=1800)
    # worker-mode passthrough
    ap.add_argument("--wrapper")
    args = ap.parse_args()

    if args.worker:
        C.run_worker(args.worker_out, args.cases_json, args.out_dir,
                     args.wrapper, args.route_config, "hybrid_evolved", args.top_k, args.max_steps,
                     args.set_label)
        return

    manifest = json.load(open(_p(args.manifest)))
    reuse = C.build_reuse_map([_p(x) for x in RESULT_FILES], [_p(x) for x in CONF_FILES])
    reuse.update({k: v for k, v in _schema_reuse().items() if k not in reuse})
    reuse.update({k: v for k, v in _floor_reuse(manifest).items()})  # floors authoritative

    recs, membership = C.run_benchmark(
        manifest, _p(args.rc2_wrapper), _p(args.route_config), args.out_dir, args.checkpoint,
        reuse, os.path.abspath(__file__), top_k=args.top_k, max_steps=args.max_steps,
        chunk_size=args.chunk_size, hard_timeout=args.hard_timeout, label="rc2")
    roll = C.rollup(recs, manifest)
    out = {"generated_by": "scripts/rc4r_run_rc2_benchmark.py", "wrapper": args.rc2_wrapper,
           "route_config": args.route_config, "policy_type": "hybrid_evolved",
           "top_k": args.top_k, "max_steps": args.max_steps, "num_theorems": len(recs),
           **roll, "results": recs}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC2 baseline benchmark", "",
          f"- theorems: {len(recs)} | status: {roll['status_histogram']}", "",
          "## By set", "", "| set | n | solved | failed | flake | path_err |", "|---|---|---|---|---|---|"]
    for s, d in roll["by_set"].items():
        md.append(f"| {s} | {d['n']} | {d['solved']} | {d['failed']} | {d['flake']} | {d['path_error']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4r-rc2] {roll['status_histogram']}")
    print(f"[rc4r-rc2] by_set={ {s: d['solved'] for s, d in roll['by_set'].items()} }")


if __name__ == "__main__":
    main()
