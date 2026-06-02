#!/usr/bin/env python3
"""FLI3 Part 7 — attribution over candidate eval (RC-style)."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def classify(r):
    if r["rc2_status"] == "solved":
        return "BASELINE_DUPLICATE"  # gate fired on RC2-solved theorem (offgate-on-solved)
    if not r["gate"]:
        return "GATE_NO_FIRE"
    if r.get("setup_error"):
        return "NEEDS_REVIEW"
    if r["candidate_win"]:
        if r.get("control_solved"):
            return "CONTROL_DUPLICATE"
        if r.get("robust"):
            return "TRUE_FLI3_DELTA"
        return "FLAKE"
    # no win
    if any(a.get("status") == "unknown_name" for a in r.get("actions", [])):
        return "UNKNOWN_NAME_OR_IMPORT_GAP"
    if r.get("control_solved"):
        return "CONTROL_DUPLICATE"
    return "NO_DELTA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-results", required=True)
    ap.add_argument("--rc2-results", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    cand = json.load(open(_p(args.candidate_results)))["results"]
    recs = []
    for r in cand:
        cls = classify(r)
        rec = {"theorem": r["theorem"], "set": r["set"], "namespace": r["namespace"],
               "candidate_family": r["candidate_family"], "lemma": r.get("lemma"),
               "tactic": r.get("winning_tactic"), "classification": cls,
               "controls_solved": r.get("control_solved"), "robust": r.get("robust"),
               "rc2_status": r.get("rc2_status"),
               "explanation": ""}
        if cls == "TRUE_FLI3_DELTA":
            rec["explanation"] = (f"literal RC2 failed; gate fired ({r['candidate_family']}); "
                                  f"`{r.get('winning_tactic')}` deploying `{r.get('lemma')}` solves; "
                                  f"controls {r.get('control_solved')} failed; robust; non-vacuous")
            rec["suggests_reusable_rule"] = True
        recs.append(rec)

    hist = Counter(r["classification"] for r in recs)
    true_delta = [r for r in recs if r["classification"] == "TRUE_FLI3_DELTA"]
    by_set = {s: dict(Counter(r["classification"] for r in recs if r["set"] == s))
              for s in sorted({r["set"] for r in recs})}
    by_family_true = Counter(r["candidate_family"] for r in true_delta)
    out = {"generated_by": "scripts/fli3_attribution.py",
           "num_items": len(recs), "classification_histogram": dict(hist.most_common()),
           "true_fli3_delta": len(true_delta),
           "true_delta_theorems": sorted(r["theorem"] for r in true_delta),
           "true_delta_by_family": dict(by_family_true),
           "true_delta_by_set": dict(Counter(r["set"] for r in true_delta)),
           "rescue_replay_reproduced": sum(1 for r in true_delta if r["set"] == "rescue_replay"),
           "family_holdout_wins": sum(1 for r in true_delta if r["set"] == "family_holdout"),
           "control_duplicates": hist.get("CONTROL_DUPLICATE", 0),
           "unknown_name": hist.get("UNKNOWN_NAME_OR_IMPORT_GAP", 0),
           "flakes": hist.get("FLAKE", 0), "by_set": by_set, "records": recs}
    with open(_p(args.out_json), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    md = ["# FLI3 attribution", "",
          f"- items: {out['num_items']} | classes: {out['classification_histogram']}",
          f"- **TRUE_FLI3_DELTA: {out['true_fli3_delta']}** by family {out['true_delta_by_family']} "
          f"by set {out['true_delta_by_set']}",
          f"- rescue_replay reproduced: {out['rescue_replay_reproduced']}/6 | "
          f"family_holdout wins: {out['family_holdout_wins']}",
          f"- control-duplicates: {out['control_duplicates']} | unknown-name: {out['unknown_name']} | "
          f"flakes: {out['flakes']}", "",
          "## TRUE_FLI3_DELTA", "", "| theorem | family | lemma | tactic | set |",
          "|---|---|---|---|---|"]
    for r in true_delta:
        md.append(f"| `{r['theorem']}` | {r['candidate_family']} | `{r['lemma']}` | "
                  f"`{r['tactic']}` | {r['set']} |")
    with open(_p(args.out_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli3-attrib] TRUE_FLI3_DELTA={len(true_delta)} "
          f"(rescue {out['rescue_replay_reproduced']}/6, holdout {out['family_holdout_wins']}) "
          f"classes={dict(hist)}")


if __name__ == "__main__":
    main()
