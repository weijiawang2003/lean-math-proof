#!/usr/bin/env python3
"""TR2 Part 7 — SX4 attribution + training-label candidates.

Consumes the RC2 confirmation (literal-production baseline) and the live probe
results (reuse-merged), and assigns each case a verified outcome class, crediting
only genuine deltas. Same discipline as scripts/sf4_apply_sx4_attribution.py and
project/evolve/experiments/sx4/sx4_methodology.md: never credit a sequence on its
depth-(k-1) sub-tactics; require a win over literal production.

Classes (Part 7 spec):
  TRUE_DELTA                     confirmed failure; gated probe solves; controls + depth-1 fail; generic
  BASELINE_DUPLICATE             a bare control (or depth-1 sub-tactic) closes it
  PRODUCTION_SUBSUMED            literal RC2 solves it with no single cheap control (search-only)
  NO_CHEAP_ACTION                confirmed failure; minimal controls all fail (verified negative)
  MISSING_BRIDGE_LEMMA_CANDIDATE confirmed failure; retrieval probe (exact?) fails -> SF5 target
  PROOF_SEARCH_DEPTH_GAP         confirmed failure; bounded depth battery fails -> deeper search needed
  SOURCE_SPECIFIC                winning probe is a source-specific rw bridge (never credited)
  OPEN_FLAKE                     not live / Dojo error
  NEEDS_REVIEW                   otherwise

Each case also yields a training-label candidate {new_label, label_confidence,
usable_for_tr1, reason}.
"""
from __future__ import annotations

import argparse
import json
import os

BASELINE_CONTROLS = {"simp", "simp_all", "aesop", "classical <;> aesop"}
USEFUL = {"TRUE_DELTA", "BASELINE_DUPLICATE", "NO_CHEAP_ACTION",
          "MISSING_BRIDGE_LEMMA_CANDIDATE", "PROOF_SEARCH_DEPTH_GAP"}
# map outcome class -> TR1 training label vocabulary
TR1_LABEL = {
    "BASELINE_DUPLICATE": "BASELINE_DUPLICATE",
    "NO_CHEAP_ACTION": "NO_CHEAP_ACTION",
    "MISSING_BRIDGE_LEMMA_CANDIDATE": "MISSING_BRIDGE_LEMMA_CANDIDATE",
    "PROOF_SEARCH_DEPTH_GAP": "PROOF_SEARCH_DEPTH_GAP",
}
# winning probe family -> action label (for TRUE_DELTA)
FAMILY_ACTION = {"multiset_induction": "WX3_MULTISET_INDUCTION",
                 "tofinset_aesop": "MX2_TOFINSET_AESOP", "set_ite_sanity": "SET_ITE_SIMP"}


def _classify(conf_rec, pr):
    fn = pr["full_name"]
    finished = (conf_rec or {}).get("rc2_finished")
    fam = pr.get("probe_family")
    if not pr.get("live") and not pr.get("probes_tried") and not pr.get("controls"):
        return "OPEN_FLAKE", False, pr.get("setup_error") or "no live/reused record", None

    ctl_solved = sorted({c["tactic"] for c in pr.get("controls", []) if c.get("solved")})
    sub_solved = sorted({c["tactic"] for c in pr.get("depth1_subcontrols", []) if c.get("solved")})
    probe_wins = [p for p in pr.get("probes_tried", []) if p.get("solved")]
    src_specific = any(p.get("source_specific") for p in probe_wins)

    # RC2 actually solved it (a solved-case control check)
    if finished is True:
        if ctl_solved:
            return ("BASELINE_DUPLICATE", False,
                    f"RC2-solved; bare control also closes it -> routing/depth gap: {ctl_solved}",
                    "BASELINE_DUPLICATE")
        return ("PRODUCTION_SUBSUMED", False,
                "RC2 solves it via multi-step search; no single cheap control -> not a new label", None)

    # confirmed RC2 failure
    if ctl_solved:
        return ("BASELINE_DUPLICATE", False,
                f"confirmed RC2 failure but a bare control closes it (RC2 search missed): {ctl_solved}",
                "BASELINE_DUPLICATE")
    if src_specific:
        return "SOURCE_SPECIFIC", False, "winning probe is source-specific (never credited)", None
    if probe_wins and sub_solved:
        return ("BASELINE_DUPLICATE", False,
                f"probe wins but a depth-1 sub-tactic already closes it: {sub_solved}", "BASELINE_DUPLICATE")
    if probe_wins and not sub_solved:
        wfam = sorted({p.get("family") for p in probe_wins})
        action = next((FAMILY_ACTION[f] for f in [p.get("family") for p in probe_wins] if f in FAMILY_ACTION), None)
        seqs = [p["tactic"] for p in probe_wins]
        return ("TRUE_DELTA", True,
                f"gated probe beats literal RC2; controls + depth-1 fail: {seqs} (families {wfam})",
                action or "TRUE_DELTA")
    # no probe win -> triage by family
    if fam == "retrieval":
        return ("MISSING_BRIDGE_LEMMA_CANDIDATE", False,
                "controls + exact? retrieval fail -> likely-existing Mathlib bridge lemma (SF5 target)",
                "MISSING_BRIDGE_LEMMA_CANDIDATE")
    if fam == "depth_gap_bounded":
        return ("PROOF_SEARCH_DEPTH_GAP", False,
                "bounded depth-2/3 battery + controls fail -> needs deeper search / lemma, not a cheap fix",
                "PROOF_SEARCH_DEPTH_GAP")
    if fam in ("minimal_controls", "controls"):
        return ("NO_CHEAP_ACTION", False,
                "minimal controls all fail on a confirmed RC2 failure -> verified no cheap action",
                "NO_CHEAP_ACTION")
    if fam == "set_ite_sanity":
        return ("PRODUCTION_SUBSUMED", False, "RC2 already owns simp [Set.ite] -> sanity negative", None)
    return "NEEDS_REVIEW", False, "no clear outcome", None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--confirmation", required=True)
    p.add_argument("--probe-results", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--tr1-examples", default="project/evolve/experiments/tr1/data/tr1_examples.jsonl")
    args = p.parse_args(argv)

    conf = {r["full_name"]: r for r in json.load(open(args.confirmation)).get("results", [])}
    probes = json.load(open(args.probe_results)).get("results", [])
    tr1_label = {}
    if os.path.exists(args.tr1_examples):
        for l in open(args.tr1_examples):
            if l.strip():
                e = json.loads(l); tr1_label[e["full_name"]] = e.get("label")

    records, label_candidates = [], []
    for pr in probes:
        fn = pr["full_name"]
        cls, credit, reason, new_label = _classify(conf.get(fn), pr)
        useful = cls in USEFUL
        # confidence: probe-win deltas / control proofs are verified; triage negatives are strong
        if cls == "TRUE_DELTA" or (cls == "BASELINE_DUPLICATE"):
            conf_lvl = "verified"
        elif cls in ("NO_CHEAP_ACTION", "MISSING_BRIDGE_LEMMA_CANDIDATE", "PROOF_SEARCH_DEPTH_GAP"):
            conf_lvl = "strong"
        else:
            conf_lvl = "weak"
        prior = tr1_label.get(fn)
        matches_prior = (new_label == prior) if new_label else None
        usable = bool(new_label) and useful
        rec = {"full_name": fn, "classification": cls, "credit": credit, "reason": reason,
               "probe_family": pr.get("probe_family"), "predicted_label": pr.get("predicted_label"),
               "winning_probes": pr.get("winning_probes"),
               "controls_solved": sorted({c["tactic"] for c in pr.get("controls", []) if c.get("solved")}),
               "rc2_finished": (conf.get(fn) or {}).get("rc2_finished"),
               "useful_label": useful, "num_live": pr.get("num_live"), "num_reused": pr.get("num_reused"),
               "tr1_prior_label": prior, "matches_prior_label": matches_prior}
        records.append(rec)
        label_candidates.append({"full_name": fn, "new_label": new_label,
                                 "label_confidence": conf_lvl, "usable_for_tr1": usable,
                                 "matches_prior_label": matches_prior, "reason": reason})

    import collections
    hist = collections.Counter(r["classification"] for r in records)
    true_delta = sorted(r["full_name"] for r in records if r["classification"] == "TRUE_DELTA")
    n_useful = sum(1 for r in records if r["useful_label"])
    out = {"confirmation_input": args.confirmation, "probe_results_input": args.probe_results,
           "num_cases": len(records), "classification_histogram": dict(hist),
           "true_delta_wins": true_delta, "num_true_delta": len(true_delta),
           "num_useful_labels": n_useful,
           "credit_policy": "Only TRUE_DELTA is credited; useful labels also include verified "
                            "missing-bridge / depth-gap / no-cheap-action / baseline-duplicate negatives.",
           "records": records, "label_candidates": label_candidates}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# TR2 SX4 attribution", "",
         f"- cases: **{len(records)}**  ·  **TRUE_DELTA: {len(true_delta)}** {true_delta}",
         f"- useful labels: **{n_useful}**",
         f"- histogram: {dict(hist)}", "",
         "| theorem | class | credit | useful | family | matches TR1? | reason |",
         "|---|---|---|---|---|---|---|"]
    for r in records:
        L.append(f"| `{r['full_name']}` | **{r['classification']}** | {'✅' if r['credit'] else '—'} | "
                 f"{'✓' if r['useful_label'] else '—'} | {r['probe_family']} | {r['matches_prior_label']} | "
                 f"{r['reason'][:60]} |")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[tr2-sx4] cases={len(records)} true_delta={len(true_delta)} useful={n_useful} hist={dict(hist)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
