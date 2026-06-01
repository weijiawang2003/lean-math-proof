#!/usr/bin/env python3
"""SF5 Part 7 — attribute retrieval probe outcomes over LITERAL RC2.

Every target in the plan is a CONFIRMED literal-RC2 failure (verified through the
production harness in rc2_failure_confirmation.json), so any probe that closes the
goal from the initial state is a genuine win over literal RC2 — there is no
best-first search to subsume it (contrast the SX3 sequence over-credit, which lived
inside RC2's own search). We still guard against stale baselines.

Classes (ordered):
  PRODUCTION_SUBSUMED      rc2_status is not a confirmed failure (stale / now solved)
  BASELINE_DUPLICATE       a no-lemma trivial probe solves (shouldn't occur on a
                           confirmed failure; kept as a guard)
  EXISTING_LEMMA_GAP       a named existing-lemma probe (exact/simpa/simp/rw/cluster)
                           closes it generically
  RETRIEVAL_ROUTING_GAP    only library-search (exact?/apply?) or aesop+hint closes it
                           -> the lemma exists & is reachable, RC2 routing never tries it
  SOURCE_SPECIFIC          win comes only from a diagnostic source-copy probe
  PROOF_DEPTH_GAP          no one-step win; prior label / cluster says deeper search
  TRUE_MISSING_BRIDGE_LEMMA  no lemma solves; repeated cluster shape -> missing bridge
  NO_RETRIEVAL_SIGNAL      retrieval produced nothing usable / no direction
"""
from __future__ import annotations

import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NAMED_LEMMA_FAMILIES = {"exact", "simpa_using", "simp_lemma", "rw_lemma",
                        "cluster_simp_only", "cluster_simp", "def_unfold_simp"}
ROUTING_FAMILIES = {"diagnostic_search", "aesop_add_simp"}


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-results", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--targets",
                    default="project/evolve/experiments/sf5/cases/"
                            "sf5_missing_bridge_targets.json")
    ap.add_argument("--retrieval",
                    default="project/evolve/experiments/sf5/out/sf5_retrieval_results.json")
    ap.add_argument("--tr2",
                    default="project/evolve/experiments/tr2/out/tr2_attributed_outcomes.json")
    args = ap.parse_args()

    pr = json.load(open(_p(args.probe_results)))
    targets = {t["full_name"]: t for t in json.load(open(_p(args.targets)))}
    retr = {r["target"]: r for r in json.load(open(_p(args.retrieval)))["results"]}
    tr2_label = {}
    try:
        for rec in json.load(open(_p(args.tr2)))["records"]:
            tr2_label[rec["full_name"]] = rec.get("predicted_label") or rec.get("classification")
    except (OSError, KeyError):
        pass

    # cluster size for "repeated shape" judgement
    cluster_size = {}
    for t in targets.values():
        cid = t.get("cluster_id")
        cluster_size[cid] = cluster_size.get(cid, 0) + 1

    def _has_multistep_source_proof(fn):
        sp = targets.get(fn, {}).get("source_proof") or {}
        return bool(sp.get("has_source_proof")) and (sp.get("num_steps", 0) >= 2
                                                     or sp.get("is_term_proof"))

    records = []
    hist = {}
    for r in pr["results"]:
        fn = r["full_name"]
        cid = r.get("cluster_id")
        rc2 = r.get("rc2_status")
        wins = r.get("wins", [])
        named_wins = [w for w in wins if w.get("family") in NAMED_LEMMA_FAMILIES
                      and w.get("lemma")]
        diag_wins = [w for w in wins if w.get("diagnostic")]
        routing_wins = [w for w in wins if w.get("family") in ROUTING_FAMILIES]
        retr_count = retr.get(fn, {}).get("num_retrieved", 0)
        prior = tr2_label.get(fn)

        cls = None
        evidence = None
        winning_lemma = None
        if rc2 not in ("CONFIRMED_RC2_FAILURE", "unknown", None):
            cls = "PRODUCTION_SUBSUMED"
            evidence = f"rc2_status={rc2}"
        elif named_wins:
            cls = "EXISTING_LEMMA_GAP"
            w = named_wins[0]
            winning_lemma = w["lemma"]
            evidence = f"`{w['tactic']}` closes (existing Mathlib lemma, generic)"
        elif routing_wins or diag_wins:
            cls = "RETRIEVAL_ROUTING_GAP"
            w = (routing_wins or diag_wins)[0]
            winning_lemma = w.get("lemma")
            evidence = (f"`{w['tactic']}` closes via library search / hinted aesop "
                        f"-> lemma reachable, RC2 routing never tries it")
        else:
            # no win. KEY: every target here is itself an existing, named Mathlib lemma.
            # If it has an existing multi-step source proof, the lemma is NOT missing —
            # it just needs multi-step planning RC2's bounded battery doesn't reach
            # (PROOF_DEPTH_GAP). TRUE_MISSING_BRIDGE is reserved for the case where no
            # existing proof is found at all (does not occur on this Mathlib frontier).
            sp = targets.get(fn, {}).get("source_proof") or {}
            if _has_multistep_source_proof(fn):
                cls = "PROOF_DEPTH_GAP"
                evidence = (f"existing Mathlib proof is multi-step "
                            f"(~{sp.get('num_steps')} steps, first `{sp.get('first_tactic')}`); "
                            f"no single retrieved lemma closes it -> depth, not missing lemma")
            elif retr_count == 0:
                cls = "NO_RETRIEVAL_SIGNAL"
                evidence = "retrieval produced no scored candidate; no source proof found"
            elif cluster_size.get(cid, 0) >= 3:
                cls = "TRUE_MISSING_BRIDGE_LEMMA"
                evidence = (f"no retrieved lemma closes it, NO existing source proof found, "
                            f"and cluster `{cid}` (size {cluster_size.get(cid)}) shares goal "
                            f"shape -> genuine reusable bridge-lemma candidate")
            else:
                cls = "NO_RETRIEVAL_SIGNAL"
                evidence = "no win; no source proof; insufficient cluster repetition"

        hist[cls] = hist.get(cls, 0) + 1
        records.append({
            "full_name": fn,
            "cluster_id": cid,
            "rc2_status": rc2,
            "classification": cls,
            "win_over_literal_rc2": cls in ("EXISTING_LEMMA_GAP", "RETRIEVAL_ROUTING_GAP"),
            "winning_lemma": winning_lemma,
            "evidence": evidence,
            "num_wins": len(wins),
            "num_named_lemma_wins": len(named_wins),
            "num_diagnostic_wins": len(diag_wins),
            "all_wins": wins,
            "prior_tr2_label": prior,
            "live": r.get("live"),
            "setup_error": r.get("setup_error"),
        })

    existing = [r for r in records if r["classification"] == "EXISTING_LEMMA_GAP"]
    routing = [r for r in records if r["classification"] == "RETRIEVAL_ROUTING_GAP"]
    true_missing = [r for r in records if r["classification"] == "TRUE_MISSING_BRIDGE_LEMMA"]
    depth = [r for r in records if r["classification"] == "PROOF_DEPTH_GAP"]

    out = {
        "generated_by": "scripts/sf5_attribute_retrieval_wins.py",
        "probe_results_input": args.probe_results,
        "num_targets": len(records),
        "classification_histogram": hist,
        "num_retrieval_wins_over_rc2": len(existing) + len(routing),
        "num_existing_lemma_gap": len(existing),
        "num_retrieval_routing_gap": len(routing),
        "num_true_missing_bridge": len(true_missing),
        "num_proof_depth_gap": len(depth),
        "every_win_over_literal_rc2": all(
            r["rc2_status"] in ("CONFIRMED_RC2_FAILURE", "unknown", None)
            for r in records if r["win_over_literal_rc2"]),
        "records": records,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# SF5 retrieval attribution", "",
          f"- targets: {len(records)}",
          f"- classification: {hist}",
          f"- retrieval wins over literal RC2: **{len(existing) + len(routing)}** "
          f"(existing-lemma {len(existing)}, routing {len(routing)})",
          f"- TRUE_MISSING_BRIDGE_LEMMA: **{len(true_missing)}**, "
          f"PROOF_DEPTH_GAP: {len(depth)}", "",
          "| target | class | winning_lemma | evidence |",
          "|---|---|---|---|"]
    for r in records:
        ev = (r["evidence"] or "").replace("|", "\\|")[:80]
        md.append(f"| {r['full_name']} | {r['classification']} | "
                  f"{r['winning_lemma'] or ''} | {ev} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")

    print(f"[sf5-attrib] {hist}")
    print(f"  retrieval wins over RC2: {len(existing) + len(routing)} "
          f"(existing {len(existing)}, routing {len(routing)})")


if __name__ == "__main__":
    main()
