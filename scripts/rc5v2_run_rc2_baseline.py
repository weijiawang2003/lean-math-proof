#!/usr/bin/env python3
"""RC5V2 Part 4 — literal RC2 baseline over the fresh eval batch.

All-fresh batch (no prior exact-config cache), so it runs live via rc4r_bench_common at the
exact RC2 config. Chunked worker subprocesses under an OS hard timeout.
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


def _p(*a):
    return os.path.join(_REPO, *a)


def _manifest_from_batch(batch_path, set_dir, set_name):
    batch = json.load(open(_p(batch_path)))
    theorems = batch.get("theorems", batch)
    os.makedirs(_p(set_dir), exist_ok=True)
    set_path = _p(set_dir, set_name + ".json")
    json.dump([{"full_name": t["full_name"], "file_path": t.get("file_path"),
                "namespace": t.get("namespace"), "goal_text": t.get("statement_text"),
                "statement_text": t.get("statement_text")} for t in theorems],
              open(set_path, "w"), ensure_ascii=False, indent=2)
    return {"set_files": {set_name: os.path.relpath(set_path, _REPO)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--cases-json")
    ap.add_argument("--set-label", default="rc2")
    ap.add_argument("--batch")
    ap.add_argument("--rc2-wrapper", default="project/evolve/experiments/rc2_release/rc2_production_wrapper.json")
    ap.add_argument("--route-config", default="project/evolve/routing/ns24_router.json")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--out-dir", default="project/evolve/experiments/rc5_v2/out/rc2_bench")
    ap.add_argument("--checkpoint", default="project/evolve/experiments/rc5_v2/out/rc2_bench_checkpoint.json")
    ap.add_argument("--wrapper")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--hard-timeout", type=int, default=900)
    args = ap.parse_args()
    if args.worker:
        C.run_worker(args.worker_out, args.cases_json, args.out_dir, args.wrapper,
                     args.route_config, "hybrid_evolved", args.top_k, args.max_steps, args.set_label)
        return

    manifest = _manifest_from_batch(args.batch, "project/evolve/experiments/rc5_v2/cases/_sets", "rc5v2_batch")
    recs, _ = C.run_benchmark(manifest, _p(args.rc2_wrapper), _p(args.route_config), args.out_dir,
                              args.checkpoint, {}, os.path.abspath(__file__),
                              top_k=args.top_k, max_steps=args.max_steps, chunk_size=args.chunk_size,
                              hard_timeout=args.hard_timeout, label="rc2")
    roll = C.rollup(recs, manifest)
    out = {"generated_by": "scripts/rc5v2_run_rc2_baseline.py", "wrapper": args.rc2_wrapper,
           "num_theorems": len(recs), **roll, "results": recs}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V2 RC2 baseline", "", f"- theorems: {len(recs)} | {roll['status_histogram']}", "",
          "## By namespace", "", "| ns | n | solved |", "|---|---|---|"]
    for ns, d in roll["by_namespace"].items():
        md.append(f"| {ns} | {d['n']} | {d['solved']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v2-rc2] {roll['status_histogram']}")


if __name__ == "__main__":
    main()
