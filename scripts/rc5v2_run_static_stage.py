#!/usr/bin/env python3
"""RC5V2 Part 5 — RC4R static stage over the fresh eval batch.

RC4R is purely additive (gated priority tactics), so non-gate-firing theorems are forced
RC4 ≡ RC2 (reuse the RC2 baseline); only gate-firing theorems run live. Reports RC4 new wins
over RC2, regressions (additive ⇒ 0), gate emissions, by namespace/component.
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
import rc5v2_run_rc2_baseline as R  # reuse _manifest_from_batch

RC4D_POLICY = "project/evolve/experiments/rc4_candidates/composition_rc4d/rc4d_composition_policy.json"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--cases-json")
    ap.add_argument("--set-label", default="rc4")
    ap.add_argument("--batch")
    ap.add_argument("--rc4-wrapper", default="project/evolve/experiments/rc4_release_candidate/rc4_release_candidate_wrapper.json")
    ap.add_argument("--route-config", default="project/evolve/routing/ns24_router.json")
    ap.add_argument("--rc2-results", default="project/evolve/experiments/rc5_v2/out/rc5v2_rc2_baseline_results.json")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--out-dir", default="project/evolve/experiments/rc5_v2/out/rc4_bench")
    ap.add_argument("--checkpoint", default="project/evolve/experiments/rc5_v2/out/rc4_bench_checkpoint.json")
    ap.add_argument("--wrapper")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--hard-timeout", type=int, default=900)
    args = ap.parse_args()
    if args.worker:
        C.run_worker(args.worker_out, args.cases_json, args.out_dir, args.wrapper,
                     args.route_config, "hybrid_evolved", args.top_k, args.max_steps, args.set_label)
        return

    manifest = R._manifest_from_batch(args.batch, "project/evolve/experiments/rc5_v2/cases/_sets", "rc5v2_batch")
    policy = G.load_policy(RC4D_POLICY)
    rc2 = {}
    if os.path.exists(_p(args.rc2_results)):
        rc2 = {r["full_name"]: r for r in json.load(open(_p(args.rc2_results)))["results"]}
    # gate map
    gatefire = {}
    for setname, rel in manifest["set_files"].items():
        for e in json.load(open(_p(rel))):
            f, _ = G.gate_fires(policy, e.get("namespace"), e.get("goal_text") or e.get("statement_text"),
                                e["full_name"])
            gatefire[e["full_name"]] = f

    def skip_predicate(entry):
        if gatefire.get(entry["full_name"]):
            return None
        r2 = rc2.get(entry["full_name"])
        if not r2:
            return None
        return {"status": r2["status"], "winning_tactic": r2.get("winning_tactic"),
                "provenance": "rc4_equals_rc2_nonfiring"}

    recs, _ = C.run_benchmark(manifest, _p(args.rc4_wrapper), _p(args.route_config), args.out_dir,
                              args.checkpoint, {}, os.path.abspath(__file__), top_k=args.top_k,
                              max_steps=args.max_steps, chunk_size=args.chunk_size,
                              hard_timeout=args.hard_timeout, skip_predicate=skip_predicate, label="rc4")
    for r in recs:
        r["rc4_component"] = COMP_OF.get(r.get("winning_tactic"))
    roll = C.rollup(recs, manifest)
    rc4_solved = {r["full_name"] for r in recs if r["status"] == "solved"}
    rc2_solved = {fn for fn, r in rc2.items() if r.get("status") == "solved"}
    new_over_rc2 = sorted(rc4_solved - rc2_solved)
    regressions = sorted(rc2_solved - rc4_solved)
    gate_emissions = sum(1 for v in gatefire.values() if v)
    out = {"generated_by": "scripts/rc5v2_run_static_stage.py", "wrapper": args.rc4_wrapper,
           "num_theorems": len(recs), **roll, "new_over_rc2": new_over_rc2,
           "num_new_over_rc2": len(new_over_rc2), "regressions": regressions,
           "gate_emissions": gate_emissions, "results": recs}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V2 RC4 static stage", "",
          f"- theorems: {len(recs)} | {roll['status_histogram']}",
          f"- RC4 new wins over RC2: {len(new_over_rc2)} {new_over_rc2} | regressions: {len(regressions)}",
          f"- gate emissions: {gate_emissions}", "",
          "## By namespace", "", "| ns | n | solved |", "|---|---|---|"]
    for ns, d in roll["by_namespace"].items():
        md.append(f"| {ns} | {d['n']} | {d['solved']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v2-static] {roll['status_histogram']} new_over_rc2={len(new_over_rc2)} {new_over_rc2} "
          f"regr={len(regressions)} gate_emissions={gate_emissions}")


if __name__ == "__main__":
    main()
