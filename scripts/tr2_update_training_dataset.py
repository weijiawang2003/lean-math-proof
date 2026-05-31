#!/usr/bin/env python3
"""TR2 Part 9 — incremental training-dataset update (non-destructive).

Folds TR2's verified label candidates into a NEW dataset file alongside (never
overwriting) the TR1 examples. Because every TR2 case is already a TR1 example
(fresh frontier exhausted), the update is dominated by *re-confirmations* and
*label-revision candidates* rather than net-new rows:

  net_new            full_name absent from TR1  -> appended as a new verified example
  reconfirmation     in TR1 and new_label == prior label -> confidence corroborated, no new row
  revision_candidate in TR1 and new_label != prior label -> flagged (NOT silently overwritten)
  excluded           usable_for_tr1 == false (PRODUCTION_SUBSUMED / OPEN_FLAKE / NEEDS_REVIEW / no label)

Outputs the added rows, the combined TR1+TR2 set, and a delta summary.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

LABEL_TYPE = {"BASELINE_DUPLICATE": "negative", "NO_CHEAP_ACTION": "triage",
              "MISSING_BRIDGE_LEMMA_CANDIDATE": "triage", "PROOF_SEARCH_DEPTH_GAP": "triage",
              "SET_ITE_SIMP": "positive", "WX3_MULTISET_INDUCTION": "positive",
              "MX2_TOFINSET_AESOP": "positive", "TRUE_DELTA": "positive"}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tr1-examples", required=True)
    p.add_argument("--outcomes", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pool", default="project/evolve/experiments/tr2/cases/tr2_candidate_pool.jsonl")
    args = p.parse_args(argv)

    tr1 = [json.loads(l) for l in open(args.tr1_examples) if l.strip()]
    tr1_by = {e["full_name"]: e for e in tr1}
    att = json.load(open(args.outcomes))
    pool = {r["full_name"]: r for r in (json.loads(l) for l in open(args.pool) if l.strip())} \
        if os.path.exists(args.pool) else {}

    net_new, reconfirm, revisions, excluded = [], [], [], []
    for lc in att["label_candidates"]:
        fn = lc["full_name"]
        nl = lc["new_label"]
        if not lc["usable_for_tr1"] or not nl:
            excluded.append({"full_name": fn, "reason": lc["reason"][:80], "new_label": nl})
            continue
        prior = tr1_by.get(fn, {}).get("label")
        if fn not in tr1_by:
            prow = pool.get(fn, {})
            net_new.append({
                "example_id": f"tr2_{len(net_new):03d}",
                "full_name": fn, "file_path": prow.get("file_path"),
                "namespace": prow.get("namespace"),
                "goal_text": None, "last_error": None, "trace_symptoms": [],
                "source_artifact": "tr2_active_probing", "source_surface": prow.get("sf4_cluster"),
                "rc2_status": prow.get("known_rc2_status"),
                "candidate_family": None, "label": nl, "label_type": LABEL_TYPE.get(nl, "triage"),
                "label_confidence": lc["label_confidence"], "sx4_credit": (nl == "TRUE_DELTA"),
                "minimal_relabel_class": None, "features": (prow.get("goal_features") and {}),
                "provenance": "tr2_live_verified"})
        elif nl == prior:
            reconfirm.append({"full_name": fn, "label": nl, "confidence": lc["label_confidence"]})
        else:
            revisions.append({"full_name": fn, "prior_label": prior, "new_label": nl,
                              "confidence": lc["label_confidence"], "reason": lc["reason"][:100]})

    os.makedirs(args.out_dir, exist_ok=True)
    added_path = os.path.join(args.out_dir, "tr2_added_examples.jsonl")
    with open(added_path, "w") as f:
        for r in net_new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    combined = tr1 + net_new
    comb_path = os.path.join(args.out_dir, "tr1_plus_tr2_examples.jsonl")
    with open(comb_path, "w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    before = collections.Counter(e["label"] for e in tr1)
    after = collections.Counter(e["label"] for e in combined)
    underrep = ["WX3_MULTISET_INDUCTION", "MX2_TOFINSET_AESOP", "PROOF_SEARCH_DEPTH_GAP"]
    underrep_delta = {l: {"before": before.get(l, 0), "after": after.get(l, 0)} for l in underrep}
    new_pos = [r["full_name"] for r in net_new if r["label_type"] == "positive"]
    new_neg = [r["full_name"] for r in net_new if r["label_type"] == "negative"]

    summary = {
        "tr1_examples": len(tr1), "tr2_label_candidates": len(att["label_candidates"]),
        "net_new_examples": len(net_new), "reconfirmations": len(reconfirm),
        "revision_candidates": len(revisions), "excluded": len(excluded),
        "combined_examples": len(combined),
        "label_distribution_before": dict(before), "label_distribution_after": dict(after),
        "underrepresented_label_delta": underrep_delta,
        "underrepresented_improved": any(underrep_delta[l]["after"] > underrep_delta[l]["before"] for l in underrep),
        "new_positive_labels": new_pos, "new_negative_labels": new_neg,
        "revision_candidate_details": revisions,
        "excluded_details": excluded[:20],
        "reconfirmation_details": reconfirm,
        "interpretation": ("Fresh frontier exhausted: every probed case already exists in TR1, so TR2 adds "
                           "%d net-new rows. Its value here is corroboration (%d reconfirmations) and %d "
                           "label-revision candidates flagged for human review — NOT dataset growth. To move "
                           "TR1 off PILOT_ONLY_NEEDS_MORE_DATA, a genuinely fresh multi-namespace frontier "
                           "must be sourced." % (len(net_new), len(reconfirm), len(revisions))),
        "files": {"added": added_path, "combined": comb_path},
        "note": "TR1 dataset file is NOT modified.",
    }
    json.dump(summary, open(os.path.join(args.out_dir, "tr2_dataset_delta_summary.json"), "w"), indent=2)

    L = ["# TR2 dataset delta", "",
         f"- TR1 examples: **{len(tr1)}**  ·  combined: **{len(combined)}**",
         f"- **net-new: {len(net_new)}**  ·  reconfirmations: {len(reconfirm)}  ·  "
         f"revision candidates: {len(revisions)}  ·  excluded: {len(excluded)}",
         f"- underrepresented improved: **{summary['underrepresented_improved']}**",
         f"- new positive: {new_pos}  ·  new negative: {new_neg}", "",
         f"> {summary['interpretation']}", "",
         "## Label distribution before → after", "", "| label | before | after |", "|---|---|---|"]
    for lab in sorted(set(before) | set(after)):
        L.append(f"| {lab} | {before.get(lab,0)} | {after.get(lab,0)} |")
    if revisions:
        L += ["", "## Revision candidates (flagged, not applied)", "",
              "| theorem | prior | new | reason |", "|---|---|---|---|"]
        for rv in revisions:
            L.append(f"| `{rv['full_name']}` | {rv['prior_label']} | {rv['new_label']} | {rv['reason'][:60]} |")
    open(os.path.join(args.out_dir, "tr2_dataset_delta_summary.md"), "w").write("\n".join(L))
    print(f"[tr2-dataset] net_new={len(net_new)} reconfirm={len(reconfirm)} "
          f"revisions={len(revisions)} excluded={len(excluded)} combined={len(combined)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
