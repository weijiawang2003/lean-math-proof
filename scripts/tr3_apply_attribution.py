#!/usr/bin/env python3
"""TR3 Part 8 — SX4-style attribution of depth-program wins over literal RC2.

Every target ran is a CONFIRMED literal-RC2 failure, so a program that closes the
goal single-shot from the initial state genuinely beats literal RC2 (there is no
best-first search to subsume an independent tactic — contrast the SX3 sequence
over-credit which lived *inside* RC2's search). Two guards keep credit honest:
  - if a bare control (simp/simp_all/aesop/classical <;> aesop) solves, the win is a
    BASELINE_DUPLICATE / routing artifact, not a TR3 contribution (no credit);
  - if the case is not actually a confirmed RC2 failure, it is PRODUCTION_SUBSUMED.

Credited classes: TRUE_RETRIEVAL_ONLY_DELTA / TRUE_RETRIEVAL_DEPTH_DELTA /
TRUE_DEPTH_ONLY_DELTA.
"""
from __future__ import annotations

import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RETRIEVAL_ONLY_FAMILIES = {"def_unfold_simp", "d1_exact", "d1_simpa_using",
                           "d1_simp_lemma", "d1_simpa_lemma", "d1_rw_lemma"}
RETRIEVAL_DEPTH_FAMILIES = {"d2_simp_aesop", "d2_simp_simpall", "d2_rw_aesop",
                            "d2_rw_simpall", "d2_constructor_simpa", "d2_ext_simp",
                            "d3_ext_simp_aesop", "d3_simp_try"}
DEPTH_ONLY_FAMILIES = {"d2_ext_aesop", "d3_antisymm_aesop", "d3_constructor_aesop",
                       "d3_constructor_simp_all", "d1_omega", "d1_nlinarith",
                       "d1_tofinset_simp", "d1_aesop", "d1_simp_all", "d1_tauto"}


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirmation", required=True)
    ap.add_argument("--program-results", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    conf = {r["full_name"]: r for r in json.load(open(_p(args.confirmation)))["results"]}
    pr = json.load(open(_p(args.program_results)))

    records = []
    hist = {}
    for r in pr["results"]:
        fn = r["full_name"]
        c = conf.get(fn, {})
        rc2_cls = c.get("classification")
        wins = r.get("wins", [])
        control_wins = r.get("control_wins", [])
        best = r.get("best_win")
        retrieved_any = True  # plan only built for failures that had retrieval

        cls = None
        evidence = None
        win_tactic = best["tactic"] if best else None
        win_family = best["family"] if best else None
        if not r.get("live") or r.get("setup_error"):
            cls = "OPEN_FLAKE"
            evidence = r.get("setup_error") or "not live"
        elif rc2_cls != "CONFIRMED_RC2_FAILURE":
            cls = "PRODUCTION_SUBSUMED"
            evidence = f"rc2 status = {rc2_cls}"
        elif control_wins:
            cls = "BASELINE_DUPLICATE"
            evidence = f"bare control solves: {control_wins}"
        elif wins:
            # pick the credited class from the best win's family (prefer named-lemma win)
            fam = win_family
            if fam in RETRIEVAL_ONLY_FAMILIES:
                cls = "TRUE_RETRIEVAL_ONLY_DELTA"
            elif fam in RETRIEVAL_DEPTH_FAMILIES:
                cls = "TRUE_RETRIEVAL_DEPTH_DELTA"
            elif fam in DEPTH_ONLY_FAMILIES:
                cls = "TRUE_DEPTH_ONLY_DELTA"
            else:
                cls = "NEEDS_REVIEW"
            evidence = f"`{win_tactic}` [{fam}] closes; controls fail; beats literal RC2"
        else:
            # no win
            cls = "PROOF_DEPTH_GAP" if retrieved_any else "NO_RETRIEVAL_SIGNAL"
            evidence = "retrieval found lemmas but no depth<=3 program closed it"

        credited = cls in ("TRUE_RETRIEVAL_ONLY_DELTA", "TRUE_RETRIEVAL_DEPTH_DELTA",
                           "TRUE_DEPTH_ONLY_DELTA")
        hist[cls] = hist.get(cls, 0) + 1
        records.append({
            "full_name": fn, "namespace": r.get("namespace"), "cluster_id": r.get("cluster_id"),
            "classification": cls, "credited": credited,
            "win_over_literal_rc2": credited,
            "winning_program": win_tactic, "winning_family": win_family,
            "winning_depth": best["depth"] if best else None,
            "winning_lemmas": best.get("lemmas", []) if best else [],
            "all_wins": wins, "control_wins": control_wins,
            "rc2_status": rc2_cls, "evidence": evidence,
        })

    credited = [r for r in records if r["credited"]]
    out = {
        "generated_by": "scripts/tr3_apply_attribution.py",
        "num_targets": len(records),
        "classification_histogram": hist,
        "num_true_delta": len(credited),
        "true_delta_by_class": {
            k: sum(1 for r in credited if r["classification"] == k)
            for k in ("TRUE_RETRIEVAL_ONLY_DELTA", "TRUE_RETRIEVAL_DEPTH_DELTA",
                      "TRUE_DEPTH_ONLY_DELTA")},
        "every_win_over_literal_rc2": all(
            r["rc2_status"] == "CONFIRMED_RC2_FAILURE" for r in credited),
        "true_delta_targets": [r["full_name"] for r in credited],
        "records": records,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# TR3 attribution", "",
          f"- targets: {len(records)}",
          f"- classification: {hist}",
          f"- **TRUE_DELTA total: {len(credited)}** by class {out['true_delta_by_class']}",
          f"- every win over literal RC2: {out['every_win_over_literal_rc2']}", "",
          "| target | class | program | depth | lemmas |", "|---|---|---|---|---|"]
    for r in records:
        if r["credited"]:
            md.append(f"| `{r['full_name']}` | {r['classification']} | "
                      f"`{r['winning_program']}` | {r['winning_depth']} | "
                      f"{r['winning_lemmas']} |")
    md.append("")
    md.append("### Non-credited")
    md.append("| target | class | evidence |")
    md.append("|---|---|---|")
    for r in records:
        if not r["credited"]:
            md.append(f"| `{r['full_name']}` | {r['classification']} | "
                      f"{(r['evidence'] or '')[:70]} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr3-attrib] {hist} | TRUE_DELTA={len(credited)} {out['true_delta_by_class']}")


if __name__ == "__main__":
    main()
