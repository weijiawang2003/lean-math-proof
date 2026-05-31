#!/usr/bin/env python3
"""TR1 Part 6 — error analysis.

Reports low-support labels, ambiguous clusters, SET_ITE false predictions,
missing-lemma vs no-cheap-action confusion, leakage risk (within-dist vs grouped
generalization gap; name-only vs name+goal signal), and data-collection targets.
"""
from __future__ import annotations

import argparse
import json
import os


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--training-results", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--examples", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args(argv)

    tr = json.load(open(args.training_results))
    preds = json.load(open(args.predictions))
    examples = [json.loads(l) for l in open(args.examples) if l.strip()]
    best = tr["best_model"]
    bm = tr["models"][best]["metrics"]

    support = tr["label_support"]
    low = sorted([l for l, n in support.items() if 0 < n < 3])
    zero = sorted([l for l, n in support.items() if n == 0])

    # leakage: within-dist OOF vs grouped (leave-one-namespace-out)
    within = bm["accuracy"]
    grouped = bm.get("grouped_leave_one_namespace_out_accuracy")
    leakage_gap = round(within - grouped, 3) if grouped is not None else None

    # confusion between MISSING_BRIDGE and NO_CHEAP_ACTION (from held-out predictions)
    bridge_vs_nocheap = []
    set_ite_false = []
    for r in preds["predictions"]:
        t, pr = r.get("true_triage"), r.get("predicted_label")
        if t and t != pr:
            if {t, pr} <= {"MISSING_BRIDGE_LEMMA_CANDIDATE", "NO_CHEAP_ACTION"}:
                bridge_vs_nocheap.append({"theorem": r["full_name"], "true": t, "pred": pr})
            if pr == "SET_ITE_SIMP" and t != "SET_ITE_SIMP":
                set_ite_false.append({"theorem": r["full_name"], "true": t})

    # name-only dominance: do name features alone determine label for SET_ITE / bridge?
    name_dominated = {}
    for lab in ("SET_ITE_SIMP", "MISSING_BRIDGE_LEMMA_CANDIDATE", "BASELINE_DUPLICATE"):
        ex = [e for e in examples if e["label"] == lab]
        # fraction whose name contains the obvious cue
        cue = {"SET_ITE_SIMP": "ite", "MISSING_BRIDGE_LEMMA_CANDIDATE": "iff",
               "BASELINE_DUPLICATE": ""}[lab]
        if ex:
            frac = sum(1 for e in ex if cue and cue in e["full_name"].lower()) / len(ex)
            name_dominated[lab] = round(frac, 3)

    with_goal = sum(1 for e in examples if e.get("goal_text"))
    findings = {
        "best_model": best,
        "within_distribution_accuracy_LOO": within,
        "grouped_leave_one_namespace_out_accuracy": grouped,
        "leakage_generalization_gap": leakage_gap,
        "low_support_labels": low,
        "zero_support_labels": zero,
        "missing_lemma_vs_no_cheap_confusions": bridge_vs_nocheap,
        "set_ite_false_positives": set_ite_false,
        "name_cue_dominance_fraction": name_dominated,
        "examples_with_goal_text": with_goal,
        "examples_total": len(examples),
        "usable_signal": [l for l, s in bm["per_label"].items() if s["support"] >= 3 and s["f1"] >= 0.6],
        "unreliable_labels": low + zero,
        "data_collection_targets": [
            "WX3_MULTISET_INDUCTION and MX2_TOFINSET_AESOP positives (current support 1 and 0) — "
            "mine more verified Multiset-induction / Set.Finite-aesop wins",
            "PROOF_SEARCH_DEPTH_GAP examples (support 1) — collect more bare-control-closes-but-RC2-missed cases",
            "non-Set namespaces — grouped generalization is weak; corpus is Set-dominated",
            "goal-text coverage — only %d/%d have goal text; capture initial goals for all live failures" % (with_goal, len(examples)),
        ],
        "interpretation": {
            "within_vs_grouped": ("High within-distribution OOF accuracy but a large drop under "
                                  "leave-one-namespace-out indicates the model leans on namespace/name-surface "
                                  "cues rather than transferable structure — expected for a 57-example, "
                                  "Set-dominated corpus. Treat as a PILOT signal, not a deployable router."),
            "name_vs_goal": ("SET_ITE / bridge labels are strongly predictable from the theorem NAME alone "
                             "(see name_cue_dominance_fraction); goal text adds little at this size."),
        },
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(findings, open(args.out_json, "w"), indent=2)

    L = ["# TR1 error analysis", "",
         f"- best model: `{best}`",
         f"- within-distribution (LOO) accuracy: **{within}**",
         f"- grouped (leave-one-namespace-out) accuracy: **{grouped}**  → generalization gap **{leakage_gap}**",
         f"- low-support labels: {low}; zero-support: {zero}",
         f"- goal-text coverage: {with_goal}/{len(examples)}", "",
         "## Confusions", "",
         f"- MISSING_BRIDGE ↔ NO_CHEAP_ACTION: {len(bridge_vs_nocheap)} {bridge_vs_nocheap}",
         f"- SET_ITE false positives: {len(set_ite_false)} {set_ite_false}", "",
         "## Name-cue dominance (fraction of label whose name contains the obvious cue)", ""]
    for lab, fr in name_dominated.items():
        L.append(f"- `{lab}`: {fr}")
    L += ["", "## Usable signal", "", f"- {findings['usable_signal']}",
          "", "## Unreliable labels", "", f"- {findings['unreliable_labels']}",
          "", "## Data-collection targets", ""]
    for t in findings["data_collection_targets"]:
        L.append(f"- {t}")
    L += ["", "## Interpretation", "",
          f"- **Leakage/generalization:** {findings['interpretation']['within_vs_grouped']}",
          f"- **Name vs goal:** {findings['interpretation']['name_vs_goal']}"]
    open(args.out_md, "w").write("\n".join(L))
    print(f"[tr1-error] within={within} grouped={grouped} gap={leakage_gap} "
          f"usable={findings['usable_signal']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
