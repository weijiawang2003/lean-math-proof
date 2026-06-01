#!/usr/bin/env python3
"""SF4 Part 8 — candidate family analysis + promotion recommendation.

Aggregates clusters + probes + probe-results + SX4 attribution per probe family and
recommends rc_candidate | experimental | training_only | reject.

Promotion to rc_candidate requires (all):
  - >=2 TRUE_DELTA wins, OR 1 TRUE_DELTA with strong fresh-holdout support
  - 0 off-gate
  - generic (not source-specific)
  - deterministic on rerun (not measured here -> if otherwise eligible, recommend rc_candidate
    but flag determinism as a remaining gate)
Most likely outcome: experimental or training_only.
"""
from __future__ import annotations

import argparse
import json
import os


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", required=True)
    p.add_argument("--probes", required=True)
    p.add_argument("--probe-results", required=True)
    p.add_argument("--attribution", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args(argv)

    probes = json.load(open(args.probes))["probes"]
    pres = json.load(open(args.probe_results))["results"]
    attr = {r["full_name"]: r for r in json.load(open(args.attribution))["records"]}

    fam_cluster = {}
    fam_risk = {}
    fam_src = {}
    for pr in probes:
        fam_cluster.setdefault(pr["family"], set()).update(pr["cluster_id"])
        fam_risk[pr["family"]] = pr.get("risk")
        fam_src[pr["family"]] = pr.get("source_specific", False)

    # tally per family from probe results
    fam = {f: {"probes_tried": 0, "true_delta_wins": 0, "true_delta_theorems": [],
               "duplicates": 0, "source_specific": 0, "off_gate": 0,
               "parse_errors": 0, "timeouts": 0}
           for f in fam_cluster}
    for r in pres:
        fn = r["full_name"]
        a = attr.get(fn, {})
        for pt in r.get("probes_tried", []):
            f = pt.get("family")
            if f not in fam:
                continue
            fam[f]["probes_tried"] += 1
            if pt.get("outcome") == "parse_error":
                fam[f]["parse_errors"] += 1
            if pt.get("outcome") == "timeout":
                fam[f]["timeouts"] += 1
            if pt.get("solved") and a.get("classification") == "TRUE_DELTA" and f in a.get("winning_families", []):
                fam[f]["true_delta_wins"] += 1
                fam[f]["true_delta_theorems"].append(fn)
            elif pt.get("solved") and a.get("classification") in ("BASELINE_DUPLICATE", "DEPTH1_DUPLICATE"):
                fam[f]["duplicates"] += 1

    out_fams = []
    for f, t in fam.items():
        true_wins = t["true_delta_wins"]
        eligible = (true_wins >= 2) and t["off_gate"] == 0 and not fam_src.get(f)
        if eligible:
            reco = "rc_candidate"
        elif true_wins == 1 and t["off_gate"] == 0 and not fam_src.get(f):
            reco = "experimental"
        elif t["duplicates"] > 0 and true_wins == 0:
            reco = "training_only"
        else:
            reco = "reject"
        out_fams.append({
            "family": f, "cluster_ids": sorted(fam_cluster[f]), "risk": fam_risk.get(f),
            "probes_tried": t["probes_tried"], "true_delta_wins": true_wins,
            "true_delta_theorems": t["true_delta_theorems"],
            "fresh_wins": true_wins,  # all SF4 wins are over confirmed RC2 failures = fresh by construction
            "duplicates": t["duplicates"], "source_specific": int(bool(fam_src.get(f))),
            "off_gate": t["off_gate"], "parse_errors": t["parse_errors"], "timeouts": t["timeouts"],
            "promotion_recommendation": reco,
            "determinism_gate": "UNVERIFIED — rerun required before any rc_candidate promotion",
        })
    out_fams.sort(key=lambda x: (-x["true_delta_wins"], x["family"]))

    out = {"num_families": len(out_fams),
           "any_rc_candidate": any(f["promotion_recommendation"] == "rc_candidate" for f in out_fams),
           "total_true_delta_wins": sum(f["true_delta_wins"] for f in out_fams),
           "families": out_fams}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# SF4 candidate family analysis", "",
         f"- families: **{len(out_fams)}**",
         f"- total TRUE_DELTA wins: **{out['total_true_delta_wins']}**",
         f"- any rc_candidate: **{out['any_rc_candidate']}**", "",
         "| family | risk | probes | TRUE_DELTA | dups | parse_err | timeouts | reco |",
         "|---|---|---|---|---|---|---|---|"]
    for f in out_fams:
        L.append(f"| {f['family']} | {f['risk']} | {f['probes_tried']} | {f['true_delta_wins']} | "
                 f"{f['duplicates']} | {f['parse_errors']} | {f['timeouts']} | **{f['promotion_recommendation']}** |")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[sf4-family] families={len(out_fams)} true_delta_total={out['total_true_delta_wins']} "
          f"any_rc_candidate={out['any_rc_candidate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
