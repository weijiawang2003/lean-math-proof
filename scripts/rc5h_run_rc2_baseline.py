#!/usr/bin/env python3
"""RC5H Part 5 — literal RC2 baseline over the hybrid benchmark (for TRUE_HYBRID_DELTA).

Reuse-first at the exact RC2 config: RC4R RC2 benchmark + RC4D literal RC2 + TR confirmations +
RC4D floor-bench RC2 solved-names + the TR6 RC2 confirmation; the rest run live via
rc4r_bench_common. Identical config to every prior RC2 run.
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

RC4R = "project/evolve/experiments/rc4_release_candidate/out"
RC4D = "project/evolve/experiments/rc4_candidates/composition_rc4d/out"
RESULT_FILES = [RC4R + "/rc2_benchmark_results.json", RC4D + "/literal_rc2_results.json"]
CONF_FILES = ["project/evolve/experiments/tr6/out/tr6_rc2_confirmation.json",
              "project/evolve/experiments/tr3/out/tr3_rc2_confirmation.json",
              "project/evolve/experiments/tr5/out/tr5_rc2_confirmation.json",
              "project/evolve/experiments/sf4/out/rc2_failure_confirmation.json"]
FLOOR_REUSE = {"canonical_demo_v1": RC4D + "/floor_bench/rc2_demo_v1.json",
               "canonical_nat_defs_medium": RC4D + "/floor_bench/rc2_nat_defs_medium.json",
               "canonical_nat_defs_large_v5": RC4D + "/floor_bench/rc2_nat_defs_large_v5.json"}
# rc5h floor sets are named canonical_floors (merged); map via solved-name lists from RC4D
_TR6_CONF = "project/evolve/experiments/tr6/out/tr6_rc2_confirmation.json"


def _p(*a):
    return os.path.join(_REPO, *a)


def _floor_reuse():
    """{fn: status} from RC4D floor-bench RC2 solved-name lists (covers the canonical_floors set)."""
    m = {}
    for rel in FLOOR_REUSE.values():
        if not os.path.exists(_p(rel)):
            continue
        fr = json.load(open(_p(rel)))
        solved = set(fr.get("solved_names", []))
        all_names = solved | set()
        # we only know solved; failures inferred at rollup. Seed solved=solved; the rest unknown.
        for fn in solved:
            m[fn] = {"status": "solved", "provenance": "reused:rc4d_floor_bench"}
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
    ap.add_argument("--out-dir", default="project/evolve/experiments/rc5_hybrid/out/rc2_bench")
    ap.add_argument("--checkpoint", default="project/evolve/experiments/rc5_hybrid/out/rc2_bench_checkpoint.json")
    ap.add_argument("--wrapper")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--hard-timeout", type=int, default=1800)
    args = ap.parse_args()
    if args.worker:
        C.run_worker(args.worker_out, args.cases_json, args.out_dir, args.wrapper,
                     args.route_config, "hybrid_evolved", args.top_k, args.max_steps, args.set_label)
        return

    manifest = json.load(open(_p(args.manifest)))
    reuse = C.build_reuse_map([_p(x) for x in RESULT_FILES], [_p(x) for x in CONF_FILES])
    # RC4R schema smoke RC2 side
    sm = _p(RC4D, "schema_wrapper_smoke.json")
    if os.path.exists(sm):
        for r in json.load(open(sm))["results"]:
            reuse.setdefault(r["full_name"], {"status": "solved" if r.get("rc2_finished") else "failed",
                                              "provenance": "reused:rc4d_schema_smoke"})
    reuse.update(_floor_reuse())

    recs, _ = C.run_benchmark(manifest, _p(args.rc2_wrapper), _p(args.route_config), args.out_dir,
                              args.checkpoint, reuse, os.path.abspath(__file__),
                              top_k=args.top_k, max_steps=args.max_steps,
                              chunk_size=args.chunk_size, hard_timeout=args.hard_timeout, label="rc2")
    roll = C.rollup(recs, manifest)
    out = {"generated_by": "scripts/rc5h_run_rc2_baseline.py", "wrapper": args.rc2_wrapper,
           "num_theorems": len(recs), **roll, "results": recs}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5H RC2 baseline", "", f"- theorems: {len(recs)} | {roll['status_histogram']}", "",
          "| set | n | solved |", "|---|---|---|"]
    for s, d in roll["by_set"].items():
        md.append(f"| {s} | {d['n']} | {d['solved']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5h-rc2] {roll['status_histogram']}")
    print(f"[rc5h-rc2] by_set={ {s: d['solved'] for s, d in roll['by_set'].items()} }")


if __name__ == "__main__":
    main()
