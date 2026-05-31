#!/usr/bin/env python3
"""SF4 Part 7 — apply the SX4 attribution gate to candidate-probe results.

Inputs are CONFIRMED RC2 failures (baseline_finished == false by construction) plus
the live probe results. Each candidate probe outcome is classified; credit only
TRUE_DELTA.

Categories:
  TRUE_DELTA          confirmed RC2 failure; a gated probe solves; controls fail; depth-1 sub-controls fail; generic
  PRODUCTION_SUBSUMED literal RC2 actually solves it (should be RARE on a failure-first pool -> flag stale/flake)
  BASELINE_DUPLICATE  a bare control (simp/simp_all/aesop/classical <;> aesop) solves
  DEPTH1_DUPLICATE    a depth-1 sub-tactic of the sequence solves
  SOURCE_SPECIFIC     the winning probe is a source-specific rw bridge (never credited)
  TRACE_INSUFFICIENT  not live / no usable record
  FAILED_CANDIDATE    no probe solved
  NEEDS_REVIEW        otherwise

Consistent with project/evolve/experiments/sx4/sx4_methodology.md: never credit a
sequence on depth-(k-1) controls; require a genuine win over literal production.
"""
from __future__ import annotations

import argparse
import json
import os

BASELINE_CONTROLS = {"simp", "simp_all", "aesop", "classical <;> aesop"}


def _classify(conf_rec, probe_rec):
    fn = probe_rec["full_name"]
    # baseline (literal RC2) must be a confirmed failure
    baseline_finished = conf_rec.get("rc2_finished") if conf_rec else None
    if baseline_finished is True:
        return "PRODUCTION_SUBSUMED", False, "literal RC2 solves it (stale baseline or flake — investigate)"

    if not probe_rec.get("live"):
        return "TRACE_INSUFFICIENT", False, probe_rec.get("setup_error") or "no live record"

    ctl_solved = {c["tactic"] for c in probe_rec.get("controls", []) if c.get("solved")}
    sub_solved = {c["tactic"] for c in probe_rec.get("depth1_subcontrols", []) if c.get("solved")}
    probe_wins = [p for p in probe_rec.get("probes_tried", []) if p.get("solved")]
    src_specific = any(p.get("source_specific") for p in probe_wins)

    if ctl_solved:
        return "BASELINE_DUPLICATE", False, f"bare control solves: {sorted(ctl_solved)}"
    if src_specific:
        return "SOURCE_SPECIFIC", False, "winning probe is source-specific"
    if probe_wins and sub_solved:
        return "DEPTH1_DUPLICATE", False, f"depth-1 sub-tactic solves: {sorted(sub_solved)}"
    if probe_wins and not sub_solved:
        seqs = [p["tactic"] for p in probe_wins]
        return "TRUE_DELTA", True, f"gated probe solves a confirmed RC2 failure; controls+depth-1 fail: {seqs}"
    return "FAILED_CANDIDATE", False, "no gated probe solved"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--confirmation", required=True)
    p.add_argument("--probe-results", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args(argv)

    conf = {r["full_name"]: r for r in json.load(open(args.confirmation)).get("results", [])}
    probes = json.load(open(args.probe_results)).get("results", [])

    records = []
    for pr in probes:
        fn = pr["full_name"]
        cls, credit, reason = _classify(conf.get(fn), pr)
        winning = [p["tactic"] for p in pr.get("probes_tried", []) if p.get("solved")]
        win_fams = sorted({p.get("family") for p in pr.get("probes_tried", []) if p.get("solved")})
        records.append({"full_name": fn, "classification": cls, "credit": credit,
                        "reason": reason, "winning_probes": winning, "winning_families": win_fams,
                        "rc2_baseline_finished": (conf.get(fn) or {}).get("rc2_finished")})

    hist = {}
    for r in records:
        hist[r["classification"]] = hist.get(r["classification"], 0) + 1
    true_delta = sorted(r["full_name"] for r in records if r["classification"] == "TRUE_DELTA")
    out = {"confirmation_input": args.confirmation, "probe_results_input": args.probe_results,
           "num_candidates": len(records), "classification_histogram": hist,
           "true_delta_wins": true_delta, "num_true_delta": len(true_delta),
           "credit_policy": "Only TRUE_DELTA is credited.",
           "production_subsumed_note": "On a failure-first pool, PRODUCTION_SUBSUMED should be ~0; any occurrence flags a stale baseline or open-flake.",
           "records": records}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# SF4 SX4 attribution", "",
         f"- candidates classified: **{len(records)}**",
         f"- **TRUE_DELTA (credited): {len(true_delta)}** {true_delta}",
         f"- histogram: {hist}", "",
         "| theorem | class | credit | winning families | reason |", "|---|---|---|---|---|"]
    for r in records:
        L.append(f"| `{r['full_name']}` | **{r['classification']}** | {'✅' if r['credit'] else '—'} | "
                 f"{r['winning_families']} | {r['reason']} |")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[sf4-sx4] candidates={len(records)} true_delta={len(true_delta)} hist={hist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
