#!/usr/bin/env python3
"""SF4 Part 9 — missing-lemma triage.

Identifies confirmed-RC2-failure clusters where probes ALSO failed, and the repeated
goal shape suggests a reusable lemma / a search-depth gap / a parser limit. Produces
candidate directions + rationale only — does NOT invent lemmas.

Categories:
  LIKELY_EXISTING_MATHLIB_LEMMA   a known lemma probably exists; this is a retrieval/routing gap
  POSSIBLE_MISSING_BRIDGE_LEMMA   repeated shape, no generic tactic closes -> reusable bridge candidate
  PROOF_SEARCH_DEPTH_GAP          a bare control/probe closed it in isolation but literal RC2 search did not
  PARSER_LIMITATION               probes failed only via parse errors
  SOURCE_SPECIFIC_ONLY            only a source-specific rw chain would close it
  NEEDS_MORE_DATA                 cluster too small / signal unclear
"""
from __future__ import annotations

import argparse
import json
import os


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--confirmed", required=True)
    p.add_argument("--clusters", required=True)
    p.add_argument("--probe-results", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args(argv)

    clusters = json.load(open(args.clusters))["clusters"]
    pres = {r["full_name"]: r for r in json.load(open(args.probe_results))["results"]}

    triage = []
    for c in clusters:
        members = c["members"]
        # gather per-member probe outcome
        unresolved, depth_gap, parse_only, baseline_solved = [], [], [], []
        for fn in members:
            r = pres.get(fn)
            if not r:
                unresolved.append(fn); continue
            ctl_solved = [x["tactic"] for x in r.get("controls", []) if x.get("solved")]
            probe_wins = [x for x in r.get("probes_tried", []) if x.get("solved")]
            outcomes = {x.get("outcome") for x in r.get("probes_tried", [])}
            if ctl_solved:
                baseline_solved.append(fn)     # solved in isolation but RC2 search missed -> depth gap
                depth_gap.append(fn)
            elif probe_wins:
                pass  # a candidate win (handled by attribution, not a missing lemma)
            elif outcomes and outcomes <= {"parse_error"}:
                parse_only.append(fn)
            else:
                unresolved.append(fn)

        n = len(members)
        if depth_gap and len(depth_gap) == n:
            cat = "PROOF_SEARCH_DEPTH_GAP"
            rationale = ("every member is closed in isolation by a bare control/probe but literal RC2 "
                         "search did not reach it — depth/ordering gap, not a missing lemma")
        elif parse_only and len(parse_only) >= max(1, n // 2):
            cat = "PARSER_LIMITATION"
            rationale = "probes failed predominantly via parse errors (run_transition single-line limits)"
        elif len(unresolved) == n and n >= 2:
            cat = "POSSIBLE_MISSING_BRIDGE_LEMMA"
            rationale = ("repeated goal shape with NO generic tactic/sequence closing any member -> "
                         "reusable bridge-lemma candidate (verify a Mathlib lemma does not already exist)")
        elif len(unresolved) == n and n == 1:
            cat = "NEEDS_MORE_DATA"
            rationale = "single unresolved theorem; insufficient repetition to infer a lemma"
        elif unresolved:
            cat = "LIKELY_EXISTING_MATHLIB_LEMMA"
            rationale = ("partial: some members unresolved while the shape is standard -> probably an "
                         "existing Mathlib lemma not retrieved/routed")
        else:
            cat = "NEEDS_MORE_DATA"
            rationale = "no clear signal"

        triage.append({
            "cluster_id": c["cluster_id"], "namespace": c["namespace"],
            "goal_shape": c["goal_shape"], "size": n, "members": members,
            "category": cat, "rationale": rationale,
            "unresolved": unresolved, "depth_gap": depth_gap, "parse_only": parse_only,
            "common_goal_features": c.get("common_goal_features"),
            "do_not_invent": True,
        })

    hist = {}
    for t in triage:
        hist[t["category"]] = hist.get(t["category"], 0) + 1
    out = {"num_clusters": len(triage), "category_histogram": hist,
           "note": "Candidate directions only — no lemmas invented. Verify existence in Mathlib before any SF5 synthesis.",
           "triage": triage}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# SF4 missing-lemma triage", "",
         f"- clusters triaged: **{len(triage)}**",
         f"- histogram: {hist}", "",
         "| cluster | ns | shape | size | category |", "|---|---|---|---|---|"]
    for t in triage:
        L.append(f"| `{t['cluster_id']}` | {t['namespace']} | {t['goal_shape']} | {t['size']} | **{t['category']}** |")
    L += ["", "## Detail", ""]
    for t in triage:
        L.append(f"### `{t['cluster_id']}` — {t['category']}")
        L.append(f"- members: {t['members']}")
        L.append(f"- rationale: {t['rationale']}")
        L.append("")
    L += ["", f"> {out['note']}"]
    open(args.out_md, "w").write("\n".join(L))
    print(f"[sf4-lemma] clusters={len(triage)} hist={hist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
